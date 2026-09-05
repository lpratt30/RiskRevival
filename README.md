# RiskRevival

A graph-based Risk environment for training a Double DQN agent against neutral
and randomized heuristic bots. RiskRevival continues the original Georgia Tech
CS7643 project while separating game rules, turn handling, learning, and training.

The refactor preserves the original research intent and observation layout.
See [BUGS.md](BUGS.md) for fixes, intentional behavior changes, validation, and
remaining limitations.

## Setup

Use Python 3.9–3.11 with the pinned dependencies. Validation used Python 3.9 on
Windows; the inherited PyTorch version does not support Python 3.13.

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

On macOS/Linux, create the environment with a compatible Python and use
`.venv/bin/python` in place of the Windows executable.

Edit [training_config.yaml](training_config.yaml) to choose the experiment,
board, opponents, network, exploration schedule, and training budget. You can
also provide a separate configuration and output directory:

```text
python train.py --config my_config.yaml --output-dir experiment_results/my_run
```

Paths in the default run are resolved relative to the project, so invoking
`train.py` from another directory works. Existing checkpoints are protected:
choose a new experiment name, use `--load-model` to initialize from configured
weights, or explicitly set `overwrite: true`. Loading weights does not restore
optimizer/replay state.

Each run writes configuration, per-episode metrics, best/final model weights,
and plots. Set `save_plots: false` to omit plots. Set `eval_only: true` to
evaluate the configured `load_checkpoint`; `render_final_results` controls
evaluation after training. Evaluation is greedy and does not change epsilon.
`show_board: true` enables interactive board displays.

For optional local CPU/GPU monitoring:

```text
python -m pip install -r requirements-monitoring.txt
python train_hardware_usage.py --config my_config.yaml
```

The monitoring entry point uses the same trainer and writes
`hardware_metrics.json`. Add `--profile profile_stats.prof` for a local
profile. W&B is used only when `--wandb-project YOUR_PROJECT` is supplied; it
uses your configured account.

## Live experiment monitoring

After installing `requirements-monitoring.txt`, start training with live logs:

```text
python train.py --monitor
```

In a second terminal, open the local dashboard:

```text
python monitor.py --open
```

The dashboard listens at `http://127.0.0.1:6006` and scans `experiment_results`.
On first use, open the **gear icon → Reload data** and set **Reload Period** to
2 seconds. TensorBoard's browser refresh is separate from the server's file
reload interval and may start disabled. Expand `train`, `evaluation`, `live`, or
`hardware` to see their charts; use the run checkboxes to compare experiments.

| Chart group | Contents | Horizontal step |
| --- | --- | --- |
| `train` | Reward, loss, epsilon, illegal/skip ratios, map ownership, results, replay size, optimizer steps, episode duration | Completed episode |
| `evaluation` | Greedy evaluation rewards, results, action counts, illegal ratio | Evaluation episode |
| `live` | Current episode/action, reward, recent loss, throughput, elapsed time, completed episodes | Heartbeat sample |
| `hardware` | System CPU/RAM, process memory, available NVIDIA GPU utilization/memory/temperature | Heartbeat sample |
| `run` (Text tab) | Configuration, current stage, completion/interruption/error status | Heartbeat sample |

Use the dashboard's wall-time axis when comparing the heartbeat groups with
episode groups. Missing hardware sensors are omitted, rather than shown as
zero; CPU/RAM sampling requires psutil and NVIDIA readings require nvidia-smi.
The NVIDIA query has a one-second timeout and runs in the background.

Both trainers accept `--monitor`, `--no-monitor`, and
`--monitor-interval 2`. The equivalent YAML options are `live_monitoring` and
`monitor_interval_seconds`. Monitoring is opt-in and disabled runs do not load
TensorBoard or create monitoring files. For custom outputs, start the dashboard
with `python monitor.py --logdir PATH --port 6007`.

Each invocation, including weight loading and evaluation-only runs, creates a
separate `OUTPUT/monitoring/TIMESTAMP-ID` directory containing:

- TensorBoard event files, periodically flushed while training is active.
- `records.jsonl`, flushed after every training/evaluation episode.
- `status.json`, an atomically replaced heartbeat with stage, progress, hardware,
  timestamps, and `running`, `completed`, `failed`, or `interrupted` status.

Closing the dashboard does not stop training. Ctrl+C in the trainer records
`interrupted` and closes the logger; an abrupt process kill or power loss cannot
write a final status, so a stale `updated_at` means the run may no longer be
active. Completed logs remain viewable afterward. Turning monitoring on does
not change action selection, rewards, replay, or optimization; hardware and
disk activity can still affect elapsed-time measurements.

TensorBoard integration uses PyTorch's
[SummaryWriter API](https://docs.pytorch.org/docs/2.1/tensorboard.html).

## Game and agent

- Maps: `size: 0` (4 territories), `1` (9), `2` (13), or `classic` (42).
  Each board supports 2–6 players, limited by its territory count.
- Player zero is the learning agent. Supply one `bot_types` entry per opponent:
  `Neutral`, `Random`, or YAML `null` for a passive manually controlled player.
  Neutral bots only reinforce. Random bots reinforce randomly, attack weaker
  neighbors up to three times (restarting after mandatory elimination trades),
  then fortify. Border armies hold their position. An inland selected army moves
  all but one troop along a friendly path to the strongest reachable border;
  tied borders are selected by territory index. Declining an attack still allows
  this fortification step.
- Reinforcements deliberately use the project's aggressive rule:
  **3 + owned territories + continent bonuses**, rather than standard Risk.
- Actions select a territory index or the skip index equal to the map size.
  The five phases are placement, attack source, attack target, fortify source,
  and fortify target. Source and target selection are separate observations.
  Skip is legal only during source selection; placement is mandatory.
- The environment places all reinforcements, attacks with all but one troop,
  and fortifies all but one troop along a friendly path. Cards trade
  automatically when at least five are held, choosing the highest-value set.
- Observations contain five phase flags, signed troop counts normalized by the
  largest army, and the selected source (or map size when none is selected).
  Positive troops belong to the agent; all opponents are negative.
- Illegal actions return a penalty and preserve game state. The response
  includes `info["illegal_action"]` for reliable metrics. Sources without a
  legal target are rejected. Training still exposes the full action space.
- Double DQN uses online action selection, target evaluation, bounded replay,
  gradient clipping, and periodic soft target updates. Time-limited episodes
  are truncated and bootstrap; game wins/losses terminate.
- `seed` in the training configuration seeds training randomness.
  `env.reset(seed=...)` independently seeds board setup, bots, cards, and combat.

```python
from env import RiskEnvFlat

env = RiskEnvFlat({"size": 1, "bot_types": ["Random"]})
state, info = env.reset(seed=42)
action = next(t.key for t in env.territories if t.owner is env.agent)
state, reward, terminated, truncated, info = env.step(action)
env.close()
```

## Code organization

| File | Responsibility |
| --- | --- |
| `board.py` | Territory/continent graphs, validated setup, pathing, display |
| `atomic_actions.py` | Combat, troop accounting, card rules, legal fortification |
| `actors.py` | Players, hands, neutral/random bot policies |
| `env.py` | Gym API, five-phase state machine, original reward shaping |
| `dqn.py` | Network, replay buffer, Double DQN optimization, checkpoints |
| `train.py` | Configuration, training, evaluation, checkpoints, metrics |
| `reporting.py` | Shared training plots |
| `train_hardware_usage.py` | Optional monitoring/profiling around the trainer |
| `monitoring.py` | Live TensorBoard records, background hardware sampling, durable status |
| `monitor.py` | Local dashboard launcher for active and saved runs |
| `test/` | Current unit and regression tests |
| `debug_only/` | Historical experiments; see its README before use |

Existing `from env import DQN, DQNAgent` imports remain supported, as do the
original network parameter names and shapes. Old checkpoints can be loaded
when the map/network configuration matches, but corrected mechanics and
learning updates mean old and new training curves are not directly comparable.

## Verification

```text
python -m unittest discover -s test -v
python -m pip check
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
```

For a headless machine, set `MPLBACKEND=Agg`. The test suite runs a small
training/evaluation cycle in temporary directories; it does not launch a full
experiment or send telemetry. Archived experiments are excluded from linting.

## Research context

The project explores discrete reinforcement learning for Risk. Original topics
include illegal-action handling, richer state representations, hierarchical
actions, self-play, stronger opponents, credit assignment, reward design, and
multi-agent play. The current observations and reward shaping remain oriented
toward one-versus-one play, even though the board supports more players.

Related reading from the original project:
[Learn What Not to Learn: Action Elimination with Deep Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2018/file/645098b086d2f9e1e0e939c27f9f2d6f-Paper.pdf).

Original code contributors: Luke Pratt, Julia Shuieh, Robert Kiesler, and Ashish
Panchal. The project was transferred from Georgia Tech's private repository to
the public Risk repository on June 10, 2024. RiskRevival is an independent copy
of [lpratt30/Risk](https://github.com/lpratt30/Risk).

