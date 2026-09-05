"""Serve saved and active experiments in a local TensorBoard dashboard."""

import argparse
import threading
import webbrowser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logdir", type=Path, default=Path(__file__).parent / "experiment_results"
    )
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument(
        "--open", action="store_true", help="Open the dashboard in your browser"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        from tensorboard import program
    except ImportError:
        parser.error(
            "Install requirements-monitoring.txt before starting the dashboard"
        )
    board = program.TensorBoard()
    board.configure(
        argv=[
            "tensorboard",
            "--logdir",
            str(args.logdir.resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--reload_interval",
            "2",
        ]
    )
    url = board.launch()
    print(
        f"RiskRevival live dashboard: {url}\nPress Ctrl+C to stop the dashboard.",
        flush=True,
    )
    if args.open:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
