"""Train/evaluate the Risk Double DQN agent from a YAML configuration."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dqn import DQNAgent
from env import RiskEnvFlat
from reporting import create_output_graphs

DEFAULT_CONFIG = Path(__file__).with_name("training_config.yaml")


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping")
    for key in (
        "num_episodes",
        "max_actions",
        "save_interval",
        "batch_size",
        "optimize_ratio",
        "target_update_freq",
        "num_eval",
    ):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("decay_type") not in ("lin", "geo", "osc"):
        raise ValueError("decay_type must be lin, geo, or osc")
    if not 0 <= config["epsilon_min"] <= config["epsilon_max"] <= 1:
        raise ValueError("Require 0 <= epsilon_min <= epsilon_max <= 1")
    name = config.get("experiment_name")
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("experiment_name must be a single directory name")
    return config


def episode_result(env):
    if env.agent.territory_count == len(env.territories):
        return "win"
    if env.agent.territory_count == 0:
        return "lose"
    return "draw"


def evaluate(agent, env, max_actions=500, num_eval=1, show_board=False):
    """Greedy evaluation does not change the agent's exploration rate."""
    if max_actions < 1 or num_eval < 1:
        raise ValueError("Evaluation counts must be positive")
    results = []
    for _ in range(num_eval):
        state, _ = env.reset()
        total_reward = 0
        illegal_moves = 0
        for actions in range(1, max_actions + 1):
            action = agent.act(state, eval=True)
            if show_board:
                env.show_board()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            illegal_moves += int(info["illegal_action"])
            if terminated or truncated:
                break
        result = dict(
            reward=total_reward,
            actions=actions,
            illegal_move_ratio=illegal_moves / actions,
            result=episode_result(env),
        )
        results.append(result)
        print(
            f"Evaluation reward: {total_reward:.3f}; illegal moves: "
            f"{result['illegal_move_ratio']:.3f}; result: {result['result']}"
        )
    return results


def exploration_rate(config, episode):
    start, end = config["epsilon_max"], config["epsilon_min"]
    progress = (episode + 1) / config["num_episodes"]
    if config["decay_type"] == "lin":
        return max(end, start - progress * (start - end))
    amplitude = (1 - progress) * (start - end)
    phase = episode / config["num_episodes"] * config["num_oscillations"] * 2 * np.pi
    return float(np.clip(end + amplitude * abs(np.sin(phase)), end, start))


def train_episode(agent, env, max_actions, optimize_ratio):
    state, _ = env.reset()
    counts = np.zeros(env.action_space.n, dtype=int)
    rewards, losses = [], []
    illegal_moves = 0
    started = time.perf_counter()
    for actions in range(1, max_actions + 1):
        action = agent.act(state)
        counts[action] += 1
        next_state, reward, terminated, truncated, info = env.step(action)
        truncated = bool(truncated or (actions == max_actions and not terminated))
        agent.remember(state, action, reward, next_state, terminated, truncated)
        state = next_state
        rewards.append(reward)
        illegal_moves += int(info["illegal_action"])
        if actions % agent.batch_size == 0:
            for _ in range(optimize_ratio):
                loss = agent.optimize_network()
                if loss is not None:
                    losses.append(loss)
        if terminated or truncated:
            break
    return {
        "average_reward": float(np.mean(rewards)),
        "cumulative_reward": float(sum(rewards)),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "illegal_move_ratio": illegal_moves / actions,
        "actions": actions,
        "turns": env.turns_passed + 1,
        "seconds": time.perf_counter() - started,
        "action_std": float(np.std(counts)),
        "skip_ratio": float(counts[-1] / actions),
        "map_owned": env.agent.territory_count / len(env.territories),
        "result": episode_result(env),
        "truncated": truncated,
    }


def main(
    env_name="RiskEnvFlat-v0",
    num_episodes=None,
    save_interval=None,
    load_model=False,
    *,
    config_path=DEFAULT_CONFIG,
    output_dir=None,
    on_episode=None,
):
    """Shared training entry point; env_name is retained for old callers."""
    if env_name != "RiskEnvFlat-v0":
        raise ValueError("Only RiskEnvFlat-v0 is supported")
    config = load_config(config_path)
    for key, value in (
        ("num_episodes", num_episodes),
        ("save_interval", save_interval),
    ):
        if value is not None:
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be positive")
            config[key] = value
    seed = config.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    env = RiskEnvFlat(config)
    env.reset(seed=seed)
    agent = DQNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        epsilon=config["epsilon_max"],
        epsilon_min=config["epsilon_min"],
        epsilon_decay=config["epsilon_decay"]
        if config["decay_type"] == "geo"
        else None,
        learning_rate=config["learning_rate"],
        tau=config["tau"],
        num_layers=config["num_layers"],
        hidden_dim_max=config["hidden_dim_max"],
        hidden_dim_min=config["hidden_dim_min"],
        target_update_freq=config["target_update_freq"],
    )
    output = (
        Path(output_dir)
        if output_dir is not None
        else (Path(__file__).parent / "experiment_results" / config["experiment_name"])
    )
    checkpoints = output / "checkpoints"
    eval_only = config.get("eval_only", False)
    if (
        not eval_only
        and not load_model
        and checkpoints.exists()
        and any(checkpoints.iterdir())
        and not config.get("overwrite", False)
    ):
        raise FileExistsError(
            f"{checkpoints} already contains a run; choose a new experiment name "
            "or set overwrite: true explicitly"
        )
    checkpoint = checkpoints / config.get("load_checkpoint", "dqn_model_best.pth")
    if load_model or eval_only:
        agent.load(checkpoint)
    records = []
    evaluations = []
    try:
        if not eval_only:
            checkpoints.mkdir(parents=True, exist_ok=True)
            (output / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            best_reward = -np.inf
            for episode in range(config["num_episodes"]):
                if config["decay_type"] != "geo":
                    agent.epsilon = exploration_rate(config, episode)
                record = train_episode(
                    agent, env, config["max_actions"], config["optimize_ratio"]
                )
                records.append(record)
                if (episode + 1) % config["save_interval"] == 0:
                    agent.save(checkpoints / f"dqn_model_{episode}.pth")
                if record["cumulative_reward"] > best_reward:
                    best_reward = record["cumulative_reward"]
                    agent.save(checkpoints / "dqn_model_best.pth")
                print(
                    f"Episode {episode + 1}/{config['num_episodes']}: "
                    f"reward={record['cumulative_reward']:.2f}, loss={record['loss']:.5f}, "
                    f"actions={record['actions']}, epsilon={agent.epsilon:.3f}"
                )
                if on_episode is not None:
                    on_episode(episode + 1, record)
            agent.save(checkpoints / "dqn_model_final.pth")
            (output / "metrics.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
            if config.get("save_plots", True):
                keys = (
                    "average_reward",
                    "cumulative_reward",
                    "loss",
                    "illegal_move_ratio",
                    "actions",
                    "turns",
                    "seconds",
                    "action_std",
                    "skip_ratio",
                    "map_owned",
                )
                series = [[record[key] for record in records] for key in keys]
                create_output_graphs(
                    output,
                    len(records),
                    env.num_players,
                    env.bot_types,
                    env.board_size,
                    series,
                )
        if eval_only or config.get("render_final_results", False):
            if not eval_only:
                agent.load(checkpoint)
            evaluations = evaluate(
                agent,
                env,
                config["max_actions"],
                config["num_eval"],
                config.get("show_board", False),
            )
            (output / "evaluation.json").write_text(
                json.dumps(evaluations, indent=2), encoding="utf-8"
            )
        return {
            "output_dir": str(output),
            "episodes": records,
            "evaluation": evaluations,
        }
    finally:
        env.close()


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    main(
        config_path=args.config, output_dir=args.output_dir, load_model=args.load_model
    )


if __name__ == "__main__":
    cli()
