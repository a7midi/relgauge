# Observer connection certification and multi-seed audit

This patch adds two audit-only modules:

```text
relgauge/observerconnectioncertifier.py
relgauge/observerconnectionsuccessaudit.py
```

They do not change the observer-frame selection objective, and they do not reward C2, nontriviality, curvature, flux, or matter. They only re-audit selected objects and estimate random-start success frequencies.

## 1. Exhaustively certify a saved winner

For the q=4, n=8 controlled observer diamond, the full state space is 4^8 = 65536, so the certifier can enumerate all global states and use exact local channel input/background caps.

```bash
python -m relgauge.observerconnectioncertifier ^
  --winners example_results/blind_observer_connection_v4_q4_winners.pkl ^
  --winner-index 0 ^
  --frame-coordinate-mode local_charts ^
  --require-nontrivial ^
  --require-c2 ^
  --out example_results/certified_observer_connection_v4_q4.csv
```

Key outputs:

```text
example_results/certified_observer_connection_v4_q4.csv
example_results/certified_observer_connection_v4_q4_checks.csv
example_results/certified_observer_connection_v4_q4_frames.csv
example_results/certified_observer_connection_v4_q4_edges.csv
example_results/certified_observer_connection_v4_q4_transports.csv
example_results/certified_observer_connection_v4_q4_cycles.csv
example_results/certified_observer_connection_v4_q4_summary.json
```

A strong pass has:

```text
CERTIFIED NON-FLAT C2 OBSERVER-FRAME CONNECTION
exhaustive_state_count = 65536
n_live_edge_quotients = n_observer_edges
n_true_live_frames = n_frames
n_single_port_label_frames = 0
n_true_live_frame_transports = n_frame_transports
max_two_branch_completion = 1
max_loop_automorphism_validity = 1
max_path_agreement = 0
global_generated_group = C2
global_nontrivial_holonomy = 1
```

To certify every stored winner payload item:

```bash
python -m relgauge.observerconnectioncertifier ^
  --winners example_results/blind_observer_connection_v4_q4_winners.pkl ^
  --all ^
  --frame-coordinate-mode local_charts ^
  --require-nontrivial ^
  --require-c2 ^
  --out example_results/certified_observer_connection_v4_q4_all.csv
```

## 2. Multi-seed success audit

Run the same blind selection repeatedly and summarize success rates.

```bash
python -m relgauge.observerconnectionsuccessaudit 4 ^
  --runs 30 ^
  --seed-start 0 ^
  --seed-stride 1 ^
  --graph-mode frame_random_diamond ^
  --seed-mode random ^
  --population 48 ^
  --generations 120 ^
  --elite 6 ^
  --mutation-rate 0.08 ^
  --random-injection 0.10 ^
  --max-state-samples 2048 ^
  --max-channel-inputs 256 ^
  --max-channel-backgrounds 256 ^
  --min-frame-ports 2 ^
  --frame-coordinate-mode local_charts ^
  --out example_results/blind_observer_connection_v4_multiseed.csv ^
  --winners example_results/blind_observer_connection_v4_multiseed_winners.pkl ^
  --plot example_results/fig_blind_observer_connection_v4_multiseed.png
```

Key outputs:

```text
example_results/blind_observer_connection_v4_multiseed.csv
example_results/blind_observer_connection_v4_multiseed_history.csv
example_results/blind_observer_connection_v4_multiseed_summary.json
example_results/blind_observer_connection_v4_multiseed_winners.pkl
example_results/fig_blind_observer_connection_v4_multiseed.png
```

Important summary fields:

```text
all_time_connection_fraction
all_time_nontrivial_fraction
all_time_flat_only_fraction
median_first_loop_auto
median_first_nontrivial
group_counts
```

The saved multi-seed winners pickle stores the all-time and final best candidate for each run, so any successful nontrivial winner can be fed back into `observerconnectioncertifier.py`.
