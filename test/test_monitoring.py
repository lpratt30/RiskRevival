"""Verify that live records are readable before completion and survive failures."""

import importlib.util
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from monitoring import ExperimentMonitor, system_metrics
from train import DEFAULT_CONFIG, load_config, main


class MonitoringUnitTests(unittest.TestCase):
    def test_snapshot_retries_transient_windows_reader_lock(self):
        import os

        with tempfile.TemporaryDirectory() as directory:
            monitor = ExperimentMonitor(directory, self.config())
            monitor.run_dir = Path(directory)
            real_replace = os.replace
            attempts = 0

            def replace(source, target):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("file briefly open by reader")
                real_replace(source, target)

            with (
                patch("monitoring.os.replace", side_effect=replace),
                patch("monitoring.time.sleep"),
            ):
                monitor._write_snapshot()
            self.assertEqual(attempts, 3)
            self.assertEqual(
                json.loads((Path(directory) / "status.json").read_text())["status"],
                "running",
            )

    def config(self):
        return {"experiment_name": "monitor-test", "num_episodes": 2, "max_actions": 5}

    def test_disabled_monitor_has_no_dependencies_or_files(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("monitoring._writer_type") as writer,
        ):
            with ExperimentMonitor(directory, self.config()) as monitor:
                monitor.update(actions=1)
                monitor.record("train", 1, {"reward": 2})
            writer.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_interval_validation(self):
        for value in (0, -1, float("nan"), float("inf"), True, "2"):
            with self.assertRaises(ValueError):
                ExperimentMonitor("unused", self.config(), interval=value)

    def test_missing_tensorboard_has_actionable_message(self):
        from monitoring import _writer_type

        with patch.dict("sys.modules", {"torch.utils.tensorboard": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-monitoring.txt"):
                _writer_type()

    def test_heartbeat_continues_without_actions(self):
        sampled = threading.Event()
        calls = 0

        def sample():
            nonlocal calls
            calls += 1
            if calls >= 2:
                sampled.set()
            return {"cpu_percent": 12.5}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("monitoring._writer_type", return_value=Mock()),
            patch("monitoring.system_metrics", side_effect=sample),
        ):
            with ExperimentMonitor(
                directory, self.config(), enabled=True, interval=0.01
            ) as monitor:
                monitor.update(stage="training", actions=3)
                self.assertTrue(sampled.wait(timeout=2))
            status = json.loads((monitor.run_dir / "status.json").read_text())
            self.assertEqual(status["actions"], 3)
            self.assertEqual(status["hardware"]["cpu_percent"], 12.5)
            self.assertEqual(status["status"], "completed")
            self.assertFalse(monitor._thread.is_alive())

    def test_failures_and_interrupts_keep_records_and_close_writer(self):
        for exception, expected in (
            (ValueError("bad training"), "failed"),
            (KeyboardInterrupt(), "interrupted"),
        ):
            writer = Mock()
            with (
                tempfile.TemporaryDirectory() as directory,
                patch(
                    "monitoring._writer_type", return_value=Mock(return_value=writer)
                ),
                patch("monitoring.system_metrics", return_value={}),
            ):
                with self.assertRaises(type(exception)):
                    with ExperimentMonitor(
                        directory, self.config(), enabled=True
                    ) as monitor:
                        monitor.record("train", 1, {"cumulative_reward": 3})
                        raise exception
                status = json.loads((monitor.run_dir / "status.json").read_text())
                self.assertEqual(status["status"], expected)
                self.assertIn(type(exception).__name__, status["error"])
                record = json.loads((monitor.run_dir / "records.jsonl").read_text())
                self.assertEqual(record["cumulative_reward"], 3)
                writer.close.assert_called_once()
                self.assertTrue(monitor._records.closed)
                self.assertFalse(monitor._thread.is_alive())

    def test_nvidia_query_times_out_without_breaking_monitor(self):
        with (
            patch("monitoring.shutil.which", return_value="nvidia-smi"),
            patch(
                "monitoring.subprocess.run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 1),
            ) as query,
        ):
            metrics = system_metrics()
        self.assertFalse(any(key.startswith("gpu_") for key in metrics))
        self.assertEqual(query.call_args.kwargs["timeout"], 1)

    def test_multiple_gpus_and_unavailable_readings(self):
        result = Mock(stdout="50, 100, 1000, 45\n[N/A], 200, 2000, [N/A]\n")
        with (
            patch("monitoring.shutil.which", return_value="nvidia-smi"),
            patch("monitoring.subprocess.run", return_value=result),
        ):
            metrics = system_metrics()
        self.assertEqual(metrics["gpu_0_percent"], 50)
        self.assertEqual(metrics["gpu_1_memory_used_mb"], 200)
        self.assertNotIn("gpu_1_temperature", metrics)


@unittest.skipUnless(
    importlib.util.find_spec("tensorboard"),
    "Install requirements-dev.txt for TensorBoard integration tests",
)
class LiveIntegrationTests(unittest.TestCase):
    def config(self):
        config = load_config(DEFAULT_CONFIG)
        config.update(
            num_episodes=2,
            max_actions=5,
            save_interval=100,
            batch_size=2,
            num_layers=2,
            hidden_dim_max=8,
            hidden_dim_min=4,
            num_eval=1,
            save_plots=False,
            render_final_results=True,
            experiment_name="live-smoke",
            live_monitoring=True,
            monitor_interval_seconds=0.01,
            size=0,
        )
        return config

    def test_event_files_are_readable_while_writer_is_open(self):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("monitoring.system_metrics", return_value={}),
        ):
            with ExperimentMonitor(directory, self.config(), enabled=True) as monitor:
                monitor.record("train", 1, {"loss": 3.5, "result": "win"})
                accumulator = EventAccumulator(str(monitor.run_dir)).Reload()
                self.assertEqual(accumulator.Scalars("train/loss")[0].value, 3.5)
                self.assertEqual(accumulator.Scalars("train/win")[0].value, 1)
                self.assertEqual(
                    json.loads((monitor.run_dir / "status.json").read_text())["status"],
                    "running",
                )

    def test_training_and_eval_only_get_separate_live_sessions(self):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("monitoring.system_metrics", return_value={}),
        ):
            config = self.config()
            path = Path(directory) / "config.yaml"
            output = Path(directory) / "run"
            path.write_text(yaml.safe_dump(config))

            def during_episode(episode, record):
                logs = next((output / "monitoring").iterdir())
                lines = (logs / "records.jsonl").read_text().splitlines()
                self.assertEqual(len(lines), episode)
                self.assertEqual(
                    json.loads((logs / "status.json").read_text())["status"], "running"
                )

            result = main(
                config_path=path, output_dir=output, on_episode=during_episode
            )
            logs = Path(result["monitor_dir"])
            accumulator = EventAccumulator(str(logs)).Reload()
            self.assertEqual(len(accumulator.Scalars("train/loss")), 2)
            self.assertEqual(len(accumulator.Scalars("evaluation/reward")), 1)
            status = json.loads((logs / "status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["completed_episodes"], 2)
            config["eval_only"] = True
            path.write_text(yaml.safe_dump(config))
            evaluated = main(config_path=path, output_dir=output)
            self.assertNotEqual(result["monitor_dir"], evaluated["monitor_dir"])
            self.assertEqual(len(list((output / "monitoring").iterdir())), 2)

    def test_training_failure_is_recorded(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("monitoring.system_metrics", return_value={}),
        ):
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(self.config()))
            with patch(
                "train.train_episode", side_effect=RuntimeError("training exploded")
            ):
                with self.assertRaisesRegex(RuntimeError, "training exploded"):
                    main(config_path=path, output_dir=directory)
            status_path = next((Path(directory) / "monitoring").glob("*/status.json"))
            self.assertEqual(json.loads(status_path.read_text())["status"], "failed")

    def test_monitoring_preserves_seeded_training_results(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("monitoring.system_metrics", return_value={}),
        ):
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(self.config()))
            plain = main(
                config_path=path, output_dir=Path(directory) / "plain", monitor=False
            )
            live = main(
                config_path=path, output_dir=Path(directory) / "live", monitor=True
            )
            for expected, actual in zip(plain["episodes"], live["episodes"]):
                self.assertEqual(
                    {k: v for k, v in expected.items() if k != "seconds"},
                    {k: v for k, v in actual.items() if k != "seconds"},
                )


if __name__ == "__main__":
    unittest.main()
