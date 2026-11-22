# Deep Learning-Assisted Reinforcement Learning for Budget-Constrained Satellite Telemetry Investigation — Corrected Project

Avula Dinesh Kumar (BL.SC.P2DSC25006) · Chamikar Peraboina (BL.SC.P2DSC25009) · 24DS742 Reinforcement Learning · Guide: Dr. Bharath K. P.

This folder is a complete, re-executed version of the project. Every notebook was run top-to-bottom without
errors, and every number, table and plot in it comes from that run. The original notebooks and outputs in the
parent folder were not modified.

## Folder layout

| Path | Contents |
|---|---|
| `notebooks/01_preprocessing.ipynb` | ESA-ADB Mission 1, 12 channels, 90 s grid, short-gap interpolation, event split, label-clean segments |
| `notebooks/02_lstm_autoencoder.ipynb` | LSTM-AE training, threshold selection on validation, full test metrics |
| `notebooks/03_dqn_investigation.ipynb` | Window scoring, investigation cases, environment, 5-seed DQN, baselines, statistics |
| `notebooks/05_extended_experiments.ipynb` | 5-fold cross-validation over all events (main result), ablation, budget sweep 3/5/7, anomaly-score variants |
| `notebooks/06_improved_pipeline.ipynb` | **Final pipeline:** overlapping-window detection (same LSTM and threshold) + DQN training augmentation, 5-fold CV, end-to-end metrics, budget sweep |
| `notebooks/04_final_results.ipynb` | Consolidated tables, final plots, auto-generated interpretation (run last) |
| `src/` | `rl_utils.py`, `dqn_core.py`, the notebook sources (`0x_*.py`) |
| `outputs/01_dataset … 03_splits` | Metadata, labels, telemetry matrix (parquet), masks, splits |
| `outputs/04_plots` | 23 figures (PNG, 200 dpi); `23_final_improved_summary.png` is the headline figure |
| `outputs/05_lstm_autoencoder` | Model (`.keras`), normalisation statistics, training history, detection results |
| `outputs/06_rl_dqn` | 5 DQN models, training histories, validation checkpoints, window scores, investigation cases |
| `outputs/07_results` | Per-case results, comparison tables, statistical tests, configuration, interpretation |
| `outputs/09_extended_experiments` | Cross-validation results, ablation, budget sweep, statistical tests, score comparison |
| `outputs/10_improved_pipeline` | Final pipeline results: event coverage, CV results, end-to-end metrics over 168 events, budget sweep, tests |

**To re-run:** open the notebooks from inside `notebooks/` in order 01 → 02 → 03 → 05 → 06 → 04 (they expect the raw data at
`../ESA_Mission1_Selected/`). Requirements: Python 3.10+, numpy, pandas, pyarrow, scikit-learn, scipy,
matplotlib, tensorflow ≥ 2.16. Notebook 02 takes about 70 min, and notebooks 05 and 06 about 45 min each, on a 2-core CPU (a GPU makes 02 a few
minutes); the others take a few minutes. DQN seeds and folds are trained in parallel processes.

## What was wrong and what was changed

| # | Problem in the original | Effect | Fix |
|---|---|---|---|
| 1 | **DQN evaluation loaded the wrong event.** `env.reset(options={"case": case})` was ignored, so a random event was loaded while scoring used another | DQN test reward 3.00 was inconsistent with its own hits (should be 1.61); 13 / 23 test rows and 10 / 18 validation rows were invalid; the "DQN has the highest reward" claim was an artefact | Cases loaded by index; an assertion checks reward = 2·hits − 5 for every episode, and an independent verification recomputes hits from the saved channel selections |
| 2 | DQN state used **raw** reconstruction errors (≈ 0.001 to 8·10⁷) | Unstable Q-learning, state dominated by one channel | Errors bounded by ±10σ input clipping, then `log1p` (state range 0 – 4.8); status scaled to 0 / 0.5 / 1, budget to 0 – 1 |
| 3 | channel_48 has σ ≈ 3·10⁻⁵, so normalised anomaly inputs reached ≈ 31 000σ; "clean" sensor drop-outs reached −82σ | Window errors up to 10⁸ dominated thresholding and case selection | Training-only z-score (float64), clipped to ±10σ |
| 4 | LSTM test set contained only anomaly windows | Only recall ("detection rate") could be reported | Label-clean segments split 70 / 15 / 15; test precision, recall, F1, PR-AUC, ROC-AUC and false-positive rate are now reported |
| 5 | `interpolate(limit=3)` also filled the first 3 samples of **long** gaps; summary said "5 minutes" | Contradicted "longer gaps are retained as missing" | Only gaps of ≤ 3 samples (4.5 min) are interpolated; long gaps stay fully missing |
| 6 | TD target used an unmasked max over already-investigated channels | Biased targets | Masked max in the TD target, masked actions when acting |
| 7 | One DQN seed, one random draw, no model selection | Results not robust | 5 seeds (trained in parallel), best validation checkpoint per seed, Random averaged over 1 000 draws per case, paired Wilcoxon tests |
| 8 | 27 events involve all 12 channels | Every policy scores 5/5 on them, which inflates all scores | Results also reported excluding 12-channel events |
| 9 | Diagram: "Detected window" had no outgoing arrow (a `data → case` arrow instead); no agent box, no reward feedback | Methodology figure did not match the method | Redrawn: detection → case → state, DQN agent box, reward feedback loop, event split, evaluation path |
| 10 | Markdown said 118 / 25 / 25; Course 3 claimed PPO as a comparison model | Text did not match the code | Text matches the code (117 / 25 / 26); no PPO claim |
| 11 | Two extra baselines added for context | — | **LSTM-greedy** (top-5 reconstruction errors) and **Oracle** (upper bound) |
| 12 | Only 20 test events → DQN vs heuristics not statistically decidable | Weak conclusion | **5-fold cross-validation over all 130 cases** (each tested once, 5 seeds per fold, inner validation for checkpoints) |
| 13 | No sensitivity analysis | Budget 5 not justified empirically | Budget sweep 3 / 5 / 7 |
| 14 | Detection used fixed, non-overlapping 3.2 h blocks | Anomalies split across block edges were missed (38 events undetected) | Same model and frozen threshold applied to overlapping windows every 48 min: 138 events detected (+8, none lost) |
| 15 | Each training event gave the DQN only one example | Few examples of rare event types | DQN also trains on all detected windows of each training event (top 20); adopted because it won on inner validation; testing still uses one case per event |

## Final result: improved pipeline (5-fold cross-validation, budget 5)

The methodology is the same as the synopsis: the same LSTM-AE, the same frozen threshold, and the same DQN state, reward
and budget. Detection now scores overlapping windows, and the DQN trains on every detected window of each training
event. 138 of 168 events are detected and each is tested exactly once.

| Method | Avg. reward | Hits / 5 | Precision@5 | Recall@5 | Unnecessary |
|---|---:|---:|---:|---:|---:|
| **DQN (5 seeds)** | **3.25 ± 0.05** | **4.12** | **0.825 ± 0.005** | **0.775** | **0.88** |
| LSTM-Greedy | 2.74 | 3.87 | 0.774 | 0.702 | 1.13 |
| Fixed Priority | 1.61 | 3.30 | 0.661 | 0.575 | 1.70 |
| Random (exact expectation) | 0.25 | 2.62 | 0.525 | 0.417 | 2.38 |
| Oracle (upper bound) | 3.61 | 4.30 | 0.861 | 0.817 | 0.70 |

Paired Wilcoxon tests on per-event hits (n = 138), all significant:
- vs LSTM-Greedy: p = 0.0006, +0.25 hits (95 % CI +0.14 to +0.36)
- vs Fixed Priority: p ≈ 10⁻¹⁴
- vs Random: p ≈ 10⁻²¹

**End-to-end over all 168 events** (an undetected event counts as 0 channels found): DQN Recall@5 rises from 0.588
with non-overlapping windows to **0.637**. The DQN is best before and after the change.

**Budget sweep (Precision at budget, DQN / LSTM-Greedy / Fixed / Random):**

| Budget | DQN | LSTM-Greedy | Fixed | Random |
|---:|---:|---:|---:|---:|
| 3 | **0.932** | 0.855 | 0.701 | 0.525 |
| 5 | **0.825** | 0.774 | 0.661 | 0.525 |
| 7 | **0.685** | 0.674 | 0.633 | 0.525 |

The DQN is significantly better than Fixed Priority and Random at every budget, and significantly better than
LSTM-Greedy at budget 5. At budgets 3 and 7 it is ahead of LSTM-Greedy but not significantly.

## Earlier result: cross-validation with non-overlapping windows (notebook 05)

This follows the synopsis method exactly: an LSTM-AE scores channel errors, and a DQN with state = channel errors +
investigation status + remaining budget and reward ±1 investigates 5 of 12 channels. Every case is tested exactly once.
The DQN state configuration was chosen on **inner-validation reward only**, and the synopsis configuration (raw
channel errors) won.

| Method | Avg. reward | Hits / 5 | Precision@5 | Recall@5 |
|---|---:|---:|---:|---:|
| **DQN (5 seeds)** | **3.27 ± 0.05** | **4.14** | **0.827 ± 0.005** | **0.760** |
| LSTM-Greedy | 2.86 | 3.93 | 0.786 | 0.698 |
| Fixed Priority | 1.62 | 3.31 | 0.662 | 0.557 |
| Random (exact expectation) | 0.36 | 2.68 | 0.536 | 0.417 |
| Oracle (upper bound) | 3.74 | 4.37 | 0.874 | 0.810 |

Paired Wilcoxon tests on per-case hits (n = 130) show **DQN is significantly better than every baseline**:
- vs LSTM-Greedy: p = 0.005, mean +0.21 hits (95 % CI +0.08 to +0.34)
- vs Fixed Priority: p ≈ 10⁻¹³
- vs Random: p ≈ 10⁻²⁰

DQN closes about 86 % of the gap between Random and the Oracle upper bound in Precision@5.

**Budget sweep (5-fold CV), Precision at budget:**

| Budget | DQN | LSTM-Greedy | Fixed | Random | Oracle |
|---:|---:|---:|---:|---:|---:|
| 3 | **0.933** | 0.833 | 0.728 | 0.536 | 0.995 |
| 5 | **0.827** | 0.786 | 0.662 | 0.536 | 0.874 |
| 7 | **0.693** | 0.684 | 0.642 | 0.536 | 0.729 |

**Ablation (budget 5, DQN Precision@5):**

| Configuration | Precision@5 |
|---|---:|
| Synopsis detector + raw-error state (selected) | 0.827 |
| Synopsis detector + channel-normalised state | 0.805 |
| Channel-normalised detector + normalised state | 0.794 (116 cases) |

**Anomaly-score finding:** normalising each channel error by its clean-data level raises window-level test PR-AUC
from 0.517 to 0.628 and precision from 0.75 to 0.92. However, it detects fewer events (116 vs 130) and does not help
the investigation stage, so the RL pipeline keeps the synopsis detector. Report it as an additional finding.

## Results on the fixed split (test set, investigation budget = 5)

LSTM autoencoder (threshold 0.0186 fixed on validation):

| Split | Clean windows | Anomaly windows | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 5 712 | 740 | 0.799 | 0.495 | 0.611 | 0.636 | 0.810 | 0.016 |
| Test | 5 282 | 491 | 0.748 | 0.399 | 0.521 | 0.517 | 0.745 | 0.013 |

130 of 168 events have at least one detected window and become investigation cases (train 90, validation 20, test 20).

Channel investigation on 20 unseen test events (DQN: mean ± std over 5 seeds):

| Method | Avg. reward | Hits / 5 | Precision@5 | Recall@5 | Unnecessary |
|---|---:|---:|---:|---:|---:|
| **DQN** | **3.18 ± 0.27** | **4.09 ± 0.14** | **0.818 ± 0.027** | **0.719 ± 0.028** | **0.91** |
| LSTM-Greedy | 2.90 | 3.95 | 0.790 | 0.698 | 1.05 |
| Fixed Priority | 2.30 | 3.65 | 0.730 | 0.586 | 1.35 |
| Random (1 000 draws) | 0.89 | 2.94 | 0.589 | 0.418 | 2.06 |
| Oracle (upper bound) | 3.60 | 4.30 | 0.860 | 0.754 | 0.70 |

Paired Wilcoxon tests on per-event hits: DQN vs Random p = 0.002; DQN vs Fixed Priority p = 0.080; DQN vs LSTM-Greedy p = 0.49.
Excluding the five 12-channel test events, DQN Precision@5 is 0.757 vs 0.640 (Fixed) and 0.452 (Random).

### Comparison with the original (invalid) numbers

| | Original | Corrected |
|---|---|---|
| DQN test Precision@5 | 0.661 (evaluated on the wrong events) | 0.818 ± 0.027 |
| DQN test reward | 3.00 (inconsistent with its own hits) | 3.18 ± 0.27 (verified = 2·hits − 5) |
| DQN vs Fixed Priority | DQN worse on Precision/Recall | DQN higher on every metric, but not significant at α = 0.05 (p = 0.08) |
| LSTM test metrics | detection rate 0.409 only | Precision 0.748, Recall 0.399, F1 0.521, PR-AUC 0.517 |

## End-to-end audit (data and method checks, all passed)

| Check | Result |
|---|---|
| 90 s grid regular from 2000-01-01 to 2013-12-31, no duplicates | ✅ |
| Only gaps ≤ 4.5 min interpolated; no short gaps remain | ✅ |
| Resampled values equal the mean of the raw samples (spot-checked against the raw file) | ✅ |
| Labels valid (no negative durations, all within the data period); consistent UTC handling | ✅ |
| Label-clean training data contains no annotated sample | ✅ |
| Event split 117 / 25 / 26 with no shared events | ✅ |
| Normalisation statistics computed only from training clean windows | ✅ |
| Every investigation case overlaps its own event and exceeds the frozen threshold | ✅ |
| Test/fold evaluation uses only held-out events; reward = 2·hits − budget for every episode | ✅ |
| Every model or configuration choice made on validation / inner validation only | ✅ |

## Limitations to state in the report

* **30 of 168 events remain undetected.** Most are very short: the median longest annotation is about 0.4 h,
  compared with the 3.2 h LSTM window. A shorter-window model would be needed for them; the RL stage cannot
  investigate an event that was never flagged.
* **Splits are random, not chronological.** Events and clean segments are split at random, not by time, so
  neighbouring periods can fall into different splits. This is standard for event-level evaluation, but a
  chronological split would be stricter.
* **Annotations are used to exclude anomalous periods** from autoencoder training and to associate a detection with its
  event. They are never part of the DQN's initial state.
* **The DQN's advantage over LSTM-Greedy is significant but modest** (+0.25 relevant channels per event). Most of
  the signal comes from the reconstruction errors; the DQN learns how to use them under the budget.
* **The 20-event fixed split is too small for significance.** The cross-validation results are the statistically
  supported ones.
* The LSTM-AE had converged: best epoch 89, no improvement in the remaining 11 epochs, and only about a 1 % change in
  validation loss over the last 20 epochs.
* Annotations indicate event association, not proven causal root causes.
