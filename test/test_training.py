"""Numerical learning regressions and a small end-to-end training run."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from dqn import DQN, DQNAgent
from env import RiskEnvFlat
from train import (
    DEFAULT_CONFIG,
    evaluate,
    exploration_rate,
    load_config,
    main,
    train_episode,
)


class LearningRegressions(unittest.TestCase):
    def agent(self, **kwargs):
        return DQNAgent(
            3,
            2,
            num_layers=2,
            hidden_dim_max=4,
            hidden_dim_min=4,
            batch_size=1,
            **kwargs,
        )

    def test_target_starts_as_online_copy(self):
        agent = self.agent()
        for online, target in zip(
            agent.model.parameters(), agent.target_model.parameters()
        ):
            self.assertTrue(torch.equal(online, target))
            self.assertFalse(target.requires_grad)

    def test_double_dqn_target_and_terminal_mask(self):
        for terminated, truncated, expected_loss in (
            (False, False, 4),
            (False, True, 4),
            (True, False, 1),
        ):
            agent = self.agent(gamma=1, learning_rate=0, target_update_freq=100)
            with torch.no_grad():
                for model in (agent.model, agent.target_model):
                    for parameter in model.parameters():
                        parameter.zero_()
                agent.model.layers[-1].bias.copy_(
                    torch.tensor([1.0, 2.0], device=agent.device)
                )
                agent.target_model.layers[-1].bias.copy_(
                    torch.tensor([10.0, 3.0], device=agent.device)
                )
            agent.remember(np.zeros(3), 0, 0, np.zeros(3), terminated, truncated)
            self.assertAlmostEqual(agent.optimize_network(), expected_loss)
            self.assertTrue(
                all(p.grad is None for p in agent.target_model.parameters())
            )

    def test_replay_advances_without_optimizer_and_copies_observations(self):
        agent = self.agent()
        agent.memory_limit = 3
        state = np.zeros(3)
        for reward in range(7):
            agent.remember(state, 0, reward, state, False, False)
        state[0] = 99
        self.assertEqual(sorted(row[2] for row in agent.memory), [4, 5, 6])
        self.assertTrue(all(row[0][0] == 0 for row in agent.memory))

    def test_soft_update_frequency_and_epsilon_floor(self):
        agent = self.agent(
            tau=1,
            target_update_freq=2,
            epsilon=0.11,
            epsilon_min=0.1,
            epsilon_decay=0.5,
        )
        before = [parameter.clone() for parameter in agent.target_model.parameters()]
        agent.remember(np.zeros(3), 0, 100, np.zeros(3), True, False)
        agent.optimize_network()
        self.assertEqual(agent.epsilon, 0.1)
        self.assertTrue(
            all(
                torch.equal(old, new)
                for old, new in zip(before, agent.target_model.parameters())
            )
        )
        agent.optimize_network()
        self.assertTrue(
            all(
                torch.equal(a, b)
                for a, b in zip(
                    agent.model.parameters(), agent.target_model.parameters()
                )
            )
        )

    def test_checkpoint_round_trip_and_legacy_imports(self):
        from env import DQN as LegacyDQN
        from env import DQNAgent as LegacyAgent

        self.assertIs(LegacyDQN, DQN)
        self.assertIs(LegacyAgent, DQNAgent)
        original, loaded = self.agent(), self.agent()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            original.save(path)
            loaded.load(path)
        for a, b in zip(original.model.parameters(), loaded.model.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_evaluation_preserves_epsilon(self):
        env = RiskEnvFlat()
        agent = DQNAgent(
            env.observation_space.shape[0],
            env.action_space.n,
            num_layers=2,
            hidden_dim_max=4,
            hidden_dim_min=4,
            epsilon=0.6,
        )
        results = evaluate(agent, env, max_actions=3)
        self.assertEqual(agent.epsilon, 0.6)
        self.assertEqual(results[0]["actions"], 3)

    def test_action_limit_is_saved_as_truncation(self):
        env = RiskEnvFlat()
        agent = DQNAgent(
            env.observation_space.shape[0],
            env.action_space.n,
            num_layers=2,
            hidden_dim_max=4,
            hidden_dim_min=4,
        )
        record = train_episode(agent, env, max_actions=1, optimize_ratio=1)
        self.assertTrue(record["truncated"])
        self.assertEqual(agent.memory[-1][-2:], (False, True))


class TrainingSmokeTests(unittest.TestCase):
    def config(self):
        config = load_config(DEFAULT_CONFIG)
        config.update(
            num_episodes=2,
            max_actions=12,
            save_interval=100,
            batch_size=2,
            num_layers=2,
            hidden_dim_max=8,
            hidden_dim_min=4,
            num_eval=1,
            save_plots=False,
            render_final_results=True,
            experiment_name="smoke",
            size=0,
            seed=11,
        )
        return config

    def test_short_run_saves_best_checkpoint_and_evaluates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config = self.config()
            config_path.write_text(yaml.safe_dump(config))
            result = main(config_path=config_path, output_dir=root / "run")
            self.assertEqual(len(result["episodes"]), 2)
            self.assertEqual(len(result["evaluation"]), 1)
            self.assertTrue((root / "run/checkpoints/dqn_model_best.pth").exists())
            self.assertTrue((root / "run/checkpoints/dqn_model_final.pth").exists())
            metrics = json.loads((root / "run/metrics.json").read_text())
            self.assertEqual(len(metrics), 2)
            self.assertTrue(all(np.isfinite(record["loss"]) for record in metrics))
            with self.assertRaises(FileExistsError):
                main(config_path=config_path, output_dir=root / "run")
            config.update(eval_only=True, render_final_results=False)
            config_path.write_text(yaml.safe_dump(config))
            result = main(config_path=config_path, output_dir=root / "run")
            self.assertEqual(len(result["episodes"]), 0)
            self.assertEqual(len(result["evaluation"]), 1)

    def test_epsilon_schedules_stay_in_bounds(self):
        config = self.config()
        config["num_episodes"] = 100
        for decay in ("lin", "osc"):
            config["decay_type"] = decay
            rates = [exploration_rate(config, episode) for episode in range(100)]
            self.assertGreaterEqual(min(rates), config["epsilon_min"])
            self.assertLessEqual(max(rates), config["epsilon_max"])

    def test_zero_action_limit_rejected(self):
        config = self.config()
        config["max_actions"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "max_actions"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
