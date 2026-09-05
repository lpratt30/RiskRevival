"""Local, opt-in TensorBoard logging with durable progress and heartbeat files."""

import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path


def system_metrics():
    """Sample without blocking training; unavailable sensors are omitted."""
    metrics = {}
    try:
        import psutil

        metrics.update(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            process_memory_mb=psutil.Process().memory_info().rss / 1024**2,
        )
    except (ImportError, OSError):
        pass
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            for index, line in enumerate(result.stdout.splitlines()):
                values = line.split(",")
                for name, value in zip(
                    ("percent", "memory_used_mb", "memory_total_mb", "temperature"),
                    values,
                ):
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if math.isfinite(number):
                        metrics[f"gpu_{index}_{name}"] = number
        except (OSError, subprocess.SubprocessError):
            pass
    return metrics


def _writer_type():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "Live monitoring needs TensorBoard. Install requirements-monitoring.txt "
            "or run without --monitor."
        ) from error
    return SummaryWriter


class ExperimentMonitor:
    """Each invocation gets an independent run, including evaluations/resumes.

    Training only updates lightweight snapshots between actions. A background
    sampler writes heartbeat/hardware/progress at a bounded rate. Episode and
    evaluation records are flushed immediately and survive interrupted runs.
    """

    def __init__(self, output_dir, config, *, enabled=False, interval=2.0):
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError(
                "monitor_interval_seconds must be a finite positive number"
            )
        self.enabled = enabled
        self.interval = interval
        self.run_dir = None
        self._config = dict(config)
        self._output = Path(output_dir)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._writer = None
        self._records = None
        self._started = time.monotonic()
        self._sample = 0
        self._state = {
            "experiment": config["experiment_name"],
            "status": "running",
            "stage": "starting",
            "episode": 0,
            "completed_episodes": 0,
            "total_episodes": config["num_episodes"],
            "actions": 0,
            "max_actions": config["max_actions"],
        }

    def __enter__(self):
        if not self.enabled:
            return self
        writer_type = _writer_type()
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self.run_dir = self._output / "monitoring" / run_id
        self.run_dir.mkdir(parents=True)
        self._writer = writer_type(str(self.run_dir), flush_secs=self.interval)
        try:
            self._records = (self.run_dir / "records.jsonl").open("a", encoding="utf-8")
            self._writer.add_text(
                "run/config",
                "```json\n" + json.dumps(self._config, indent=2) + "\n```",
                0,
            )
            self._writer.add_text("run/status", "running", 0)
            self._write_snapshot()
            self._thread = threading.Thread(
                target=self._sample_loop, name="experiment-monitor", daemon=True
            )
            self._thread.start()
        except BaseException:
            self._writer.close()
            if self._records:
                self._records.close()
            raise
        print(f"Live monitoring logs: {self.run_dir.resolve()}")
        return self

    def update(self, **values):
        if self.enabled:
            with self._lock:
                if "stage" in values and values["stage"] != self._state["stage"]:
                    self._writer.add_text("run/stage", values["stage"], self._sample)
                self._state.update(values)

    def _scalars(self, prefix, values, step):
        for key, value in values.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                self._writer.add_scalar(f"{prefix}/{key}", value, step)

    def record(self, kind, step, values):
        if not self.enabled:
            return
        with self._lock:
            self._scalars(kind, values, step)
            if "result" in values:
                for outcome in ("win", "lose", "draw"):
                    self._writer.add_scalar(
                        f"{kind}/{outcome}", int(values["result"] == outcome), step
                    )
            self._records.write(
                json.dumps({"kind": kind, "step": step, **values}, allow_nan=False)
                + "\n"
            )
            self._records.flush()
            self._write_snapshot()
            self._writer.flush()

    def _write_snapshot(self):
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._state["elapsed_seconds"] = time.monotonic() - self._started
        target = self.run_dir / "status.json"
        temporary = self.run_dir / ".status.json.tmp"
        temporary.write_text(
            json.dumps(self._state, indent=2, allow_nan=False), encoding="utf-8"
        )
        # On Windows, a reader can briefly prevent replacement of an open file.
        # Keep the last complete snapshot visible while retrying the atomic swap.
        for attempt in range(5):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01)

    def _sample_loop(self):
        try:
            while not self._stop.is_set():
                hardware = system_metrics()
                with self._lock:
                    self._sample += 1
                    self._state["hardware"] = hardware
                    self._scalars("hardware", hardware, self._sample)
                    self._scalars("live", self._state, self._sample)
                    self._write_snapshot()
                    self._writer.flush()
                self._stop.wait(self.interval)
        except Exception as error:
            with self._lock:
                self._state["monitor_error"] = str(error)
            warnings.warn(f"Live monitoring sampler stopped: {error}", RuntimeWarning)

    def __exit__(self, exc_type, error, traceback):
        if not self.enabled:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        status = (
            "completed"
            if error is None
            else "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "failed"
        )
        try:
            with self._lock:
                self._state["status"] = status
                if error is not None:
                    self._state["error"] = f"{type(error).__name__}: {error}"
                self._writer.add_text("run/status", status, self._sample)
                if error is not None:
                    self._writer.add_text(
                        "run/error", self._state["error"], self._sample
                    )
                self._writer.add_scalar(
                    "run/status_code",
                    {"completed": 1, "failed": -1, "interrupted": -2}[status],
                    self._sample,
                )
                self._write_snapshot()
        except Exception:
            if error is None:
                raise
        finally:
            self._writer.close()
            self._records.close()
        return False
