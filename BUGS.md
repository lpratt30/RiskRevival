# Refactor findings

Baseline: `e1fa86fa215b88cb0ada0b5938bfd6aa87d8e051` (the copied Risk repository).
The baseline suite ran 23 tests with one failure in dice-bag sorting. Findings
below combine that reproduced failure, source inspection, and regression tests.

## Fixed

| Area | Original failure | Correction |
| --- | --- | --- |
| Combat dice | An incomplete attacker triple included the first defender die when sorted, moving dice between armies. | Split armies before sorting; the existing failing regression now passes. Its defender expectation also incorrectly left the last pair unsorted and was corrected. |
| Combat rounds | Cached triples/pairs could be reused after casualties changed the number of legal dice. | Roll a fresh round using at most three attacker and two defender dice, bounded by survivors. Defender still wins ties. |
| Partial attacks | Losses were calculated against every troop except one, even when fewer troops were committed; conquest discarded uncommitted troops. | Track the committed army separately and preserve reserves at the source. The environment still commits all available troops. |
| Invalid troop counts | Negative placement/attack/fortification amounts could corrupt counts. Fortification raised strings, causing TypeError, and allowed movement without a friendly path. | Reject invalid whole-troop counts before mutation; fortification validates connectivity and raises ValueError. |
| Initial armies | Ownership was shuffled but reinforcement budgets used the unshuffled player index. Uneven territory counts could give players the wrong starting total. | Charge each actual owner's budget; verify totals for every board and supported player count. |
| Board construction | The classic map hard-coded six players, the TFT bot referenced an undefined class, and small boards could start eliminated players. | Share validated player construction; classic supports 2–6 players; reject unsupported bots and too many players for the map. |
| Board display/demo | create_graph(display=True) referenced undefined colors and used name-based positions for object nodes; the direct board demo called missing/wrong APIs. | Reuse display_graph and provide a working two-player demo. Remove obsolete atomic-actions demo code with outdated signatures. |
| Card ownership | Taking an eliminated player's cards copied them without emptying the original hand. | Transfer and clear all card counters; repeated transfers cannot duplicate cards. |
| Card trades | Invalid trades still subtracted three cards; natural mixed sets could waste wilds; several valid wildcard trades were missed. | Use one highest-value recipe selector for checking and trading, preserve wilds when possible, and leave invalid hands unchanged. Repeat mandatory trades until fewer than five cards remain. |
| Reinforcements | reset() generated troops and the first placement generated them again. | Prepare reinforcements exactly once at the start of each turn. Elimination placement uses card troops only. |
| Illegal placement | Troops/cards could be generated or consumed before discovering the destination was illegal, then the reserve was discarded. | Validate first. Invalid actions preserve the board, pending selection, cards, and reserves. |
| Action handling | Negative indices selected real territories, large indices crashed, and invalid targets inconsistently cleared their source. Sources with no legal target could strand the agent. | Check the action space, retain state after invalid actions, and reject sources with no destination. |
| Reward accounting | Some illegal moves updated cumulative reward but returned a different penalty; others never updated totals. Terminal defeat could miss its reward with non-phasic credit. Further steps could repeat terminal rewards. | Centralize returned/accounted penalties, deliver terminal reward in both credit modes, and require reset after termination. |
| Randomness | reset(seed=...) seeded Python but not NumPy combat; shuffling mutated the caller's bot list. | Give each environment its own Python and NumPy generators and copy bot configuration. Interleaved seeded environments produce identical transitions. |
| Bot execution | Placement and bot turns ran inside assert expressions, so Python -O omitted the moves entirely. Nested bot helpers also captured an outer source variable. | Execute moves explicitly and check results; use methods with explicit source arguments. |
| Double DQN | The online and target networks started independently; targets used the target network's maximum (ordinary DQN) and built gradients through it. | Copy initial weights, freeze target gradients, select next actions online and evaluate them with the target network. |
| Replay buffer | Replacement depended on optimizer steps, repeatedly overwriting one slot between updates. | Advance a circular write index on every remembered transition; copy observations to prevent later mutation. |
| Exploration/evaluation | Geometric decay could go below the configured minimum; the oscillating schedule could exceed bounds; evaluation permanently set epsilon to zero. | Clamp schedules and use greedy evaluation without changing epsilon. |
| Training completion | Short runs never wrote the best checkpoint that final evaluation tried to load. eval_only could do nothing when final rendering was disabled. | Save the best checkpoint from episode one and always evaluate in eval_only mode. Also save final weights. |
| Time limits/metrics | The action cap broke the loop without marking replay transitions truncated; illegal moves were counted by reward equality. | Record time-limit truncation, bootstrap through truncation, and count illegal moves through info. |
| Training duplication | The hardware script duplicated training/plotting and hard-coded a W&B account. | Share the training loop via an episode callback. Hardware metrics are local; W&B requires an explicit CLI option. |
| Dependencies | NumPy 2.0 conflicted with matplotlib 3.7.4 and the older Gym/PyTorch stack; PyYAML appeared twice. | Pin NumPy 1.26.4, remove the duplicate, and separate optional monitoring/development dependencies. Installation and pip check pass in the validation environment. |

## Preserved choices and remaining limitations

- This is the original simplified research game, not a complete implementation
  of commercial Risk. It retains `3 + territories + continent bonuses`, fixed
  card-set values (4/6/8/10), independent card draws, and full troop commitment
  by the environment. The original map territory order and edges are retained.
- The observation and reward shaping remain designed for one agent versus one
  opponent. Additional players can simulate games, but opponents share the same
  negative troop representation and relative-territory reward compares only
  player 1. Multi-agent reward/observation redesign is outside this refactor.
- Normal valid-action reward coefficients and phase timing are retained. The
  illegal-action behavior now consistently follows the README's state-preserving
  penalty policy; learning curves will change after the documented fixes.
- Existing compatible model weights load, but checkpoints contain weights only,
  not replay, optimizer, RNG, or schedule state. Loading is not an exact resume.
- Gym 0.26.2 reports that it is unmaintained and emits a NumPy alias deprecation
  warning during its checker. PyTorch 2.1.2 emits a TypedStorage deprecation
  warning when reading old checkpoints. These dependencies were retained to
  avoid combining a framework migration with the refactor. Use Python 3.9–3.11
  with these pins; the available Python 3.13 runtime is not the supported target.
- `debug_only/` remains historical research material, not the supported training
  implementation. In particular, its two old environment copies still pass the
  obsolete `reduce_kurtosis` attack argument and raise strings. Do not use those
  files as current examples; use `env.py`, `dqn.py`, and `train.py`.
- Previously committed experiment data is retained. Generated Python bytecode
  and IDE settings are removed from version control and ignored for future runs.
- New training runs reject existing checkpoint directories unless explicitly
  configured with `overwrite: true` (or weight loading is requested). Choose a
  new experiment name to keep runs separate.

## Validation

- 55 unit/regression tests, including combat, cards, all board sizes/player
  budgets, seeded trajectories, bot elimination, replay, numerical Double DQN
  targets, time limits, checkpoint round trips, and short training/evaluation.
- Gym environment checker; dependency consistency; Ruff lint and formatting.
- Eight environment regressions also pass under Python `-O`; reward calculations
  match the original formula in 120 scenarios. A source comparison confirms
  unchanged territory order, continent bonuses, and map connections.
- A two-episode CPU smoke run with 24 actions per episode produced best/final
  checkpoints, metrics, evaluation, 11 plots, and local hardware metrics.
- The inherited `test_default6` best checkpoint loaded and selected an action.

Smoke runs establish that the pipeline executes; they do not establish strategy
quality or a performance improvement. GPU training and live W&B logging were
not validated.

## Live monitoring follow-up

- Previously, metrics were only written after training completed, so neither
  live inspection nor recovery of completed episode records was available.
  Opt-in monitoring now flushes event/episode records during the run and writes
  progress/hardware heartbeats independently of long training steps.
- TensorBoard's browser-level **Reload data** setting starts disabled in the
  validated installation. The dashboard launcher reloads event files every two
  seconds; the README explains enabling browser refresh as well.
- Errors and Ctrl+C preserve completed monitoring records and write a terminal
  status. Forced process termination cannot do so; check heartbeat freshness.
- Every invocation receives a unique monitoring directory, avoiding misleading
  overlapping step histories after evaluation, weight loading, or overwrites.
- NVIDIA hardware queries time out after one second; unavailable GPU sensors
  are omitted. Both timeout behavior and multi-GPU parsing are covered by tests.
- A Windows reader could briefly lock `status.json` during its atomic replacement,
  stopping the sampling thread. Snapshot replacement now retries permission
  errors up to five times with 10 ms between attempts; persistent errors still
  surface instead of being hidden.

## Random bot review follow-up

- A declined attack returned `None` and overwrote the selected source, preventing
  otherwise useful fortification. The source is now replaced only on conquest.
- Fortifying into the strongest friendly neighbor could withdraw the invading
  army from a newly conquered border. The bot now holds armies adjacent to an
  enemy and moves inland armies to the strongest reachable friendly border,
  using friendly paths and leaving one troop behind. No reachable border means
  no fortification. This is a deliberate opponent-policy change.
- The large-map Random-bot configuration now uses learning rate `0.0003`, down
  from `0.005`. This is a conservative starting candidate, not an empirically
  established optimum. Existing running processes must restart to use changes.
