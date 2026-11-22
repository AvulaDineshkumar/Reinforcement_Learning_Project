
* The LSTM autoencoder separates label-clean and anomalous windows on unseen test data with PR-AUC
  0.517 and ROC-AUC 0.745
  (threshold 0.0186 chosen on validation). 130 of
  168 events produced at least one detected window and became investigation cases.
* On the 20 test cases the DQN reaches Precision@5 0.818 ± 0.027,
  Recall@5 0.719 and average reward 3.18.
* DQN vs Random: Precision@5 0.818 vs 0.589 (Wilcoxon p = 0.00203).
* DQN vs Fixed priority: Precision@5 is higher than the fixed ranking
  (0.730); the paired test gives p = 0.08, so
  the difference is not statistically significant at α = 0.05.
* LSTM-greedy (top-5 reconstruction errors) reaches Precision@5 0.790; the Oracle upper bound is
  0.860 (limited by events with fewer than 5 annotated channels).
* **Cross-validated main result (all 130 cases, budget 5):** DQN Precision@5 0.827 ± 0.005,
  Recall@5 0.760, reward 3.27; LSTM-greedy 0.786,
  Fixed priority 0.662, Random 0.536, Oracle 0.874.
  Paired Wilcoxon p-values: vs LSTM-greedy 0.00513, vs Fixed priority 1.51e-13, vs Random 5.12e-20.
* The budget sweep shows DQN at or above every baseline for budgets 3, 5 and 7 (Precision 0.933, 0.827, 0.693).
* Normalising channel errors improves window-level anomaly detection (test PR-AUC 0.517 →
  0.628) but does not improve the investigation stage, so the synopsis
  configuration (raw channel errors in the DQN state) is retained, as selected on inner validation.
* **Final improved pipeline (overlapping windows + training augmentation, 138 events, budget 5):** DQN Precision@5
  0.825 ± 0.005, Recall@5 0.775, reward 3.25;
  LSTM-greedy 0.774, Fixed priority 0.661, Random 0.525,
  Oracle 0.861. Wilcoxon p: vs LSTM-greedy 0.000624, vs Fixed 5.85e-15, vs Random 7.99e-22.
* End-to-end over all 168 events, DQN Recall@5 rises from 0.588 to 0.637.
  The remaining undetected events are mostly very short (median well under one hour), shorter than one LSTM window.
* Annotations indicate event association, not proven causal root causes; reward is computed from them offline only.
