"""Optional hardware monitoring around the shared training loop.

Metrics stay local unless --wandb-project explicitly enables W&B logging.
"""

import argparse
import cProfile
import json
from pathlib import Path

from train import DEFAULT_CONFIG
from train import main as train_main


def log_system_metrics():
    import psutil

    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
    except (ImportError, OSError):
        gpus = []
    if gpus:
        metrics.update(
            gpu_percent=gpus[0].load * 100,
            gpu_memory_percent=gpus[0].memoryUtil * 100,
            gpu_temperature=gpus[0].temperature,
        )
    return metrics


def main(
    env_name="RiskEnvFlat-v0",
    num_episodes=None,
    save_interval=None,
    load_model=False,
    *,
    config_path=DEFAULT_CONFIG,
    output_dir=None,
    wandb_project=None,
    monitor=None,
    monitor_interval=None,
):
    records = []
    run = None
    if wandb_project:
        import wandb

        run = wandb.init(project=wandb_project)

    def on_episode(episode, record):
        metrics = dict(episode=episode, **log_system_metrics())
        records.append(metrics)
        if run is not None:
            run.log(dict(record, **metrics), step=episode)

    try:
        result = train_main(
            env_name,
            num_episodes,
            save_interval,
            load_model,
            config_path=config_path,
            output_dir=output_dir,
            on_episode=on_episode,
            monitor=monitor,
            monitor_interval=monitor_interval,
        )
        (Path(result["output_dir"]) / "hardware_metrics.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        return result
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-project")
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--monitor", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--monitor-interval", type=float)
    args = parser.parse_args()
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    try:
        main(
            config_path=args.config,
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            monitor=args.monitor,
            monitor_interval=args.monitor_interval,
        )
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(args.profile))
