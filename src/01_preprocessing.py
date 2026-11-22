# %% [markdown]
# # Course 1 — Dataset Preparation and Preprocessing (Corrected)
# **Deep Learning-Assisted Reinforcement Learning for Budget-Constrained Satellite Telemetry Investigation**
#
# Dataset: ESA-ADB Mission 1 (Kotowski et al., 2024), 12 selected target channels.
#
# Corrections versus the first version:
# * Only **short gaps (≤ 3 samples = 4.5 min)** are interpolated. Long gaps stay completely missing
#   (the original `interpolate(limit=3)` also partially filled the first 3 samples of long gaps).
# * The interpolation threshold is reported correctly as 4.5 minutes.
# * The 12 channels are processed **in parallel** (multiprocessing).
# * The split is reported exactly as produced: 117 / 25 / 26 anomaly events.

# %%
import os, sys, random, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

PROJECT = Path.cwd().parent
sys.path.insert(0, str(PROJECT / "src"))
import rl_utils as U

RAW = PROJECT.parent / "ESA_Mission1_Selected"
OUT = PROJECT / "outputs"
for d in ["01_dataset", "02_preprocessing", "03_splits", "04_plots"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)
PLOTS = OUT / "04_plots"

SEED = 42
random.seed(SEED); np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})

def save_show(name):
    plt.tight_layout(); plt.savefig(PLOTS / name, bbox_inches="tight"); plt.close()
    display(Image(filename=str(PLOTS / name)))

SELECTED_CHANNELS = ["channel_41", "channel_45", "channel_43", "channel_14", "channel_21", "channel_29",
                     "channel_48", "channel_52", "channel_31", "channel_22", "channel_39", "channel_15"]
print("Raw data folder:", RAW, "| exists:", RAW.exists())
print("Selected channels:", len(SELECTED_CHANNELS))

# %% [markdown]
# ## 1. Channel metadata and anomaly annotations

# %%
channels_meta = pd.read_csv(RAW / "channels.csv")
labels = pd.read_csv(RAW / "labels.csv")
labels["StartTime"] = pd.to_datetime(labels["StartTime"], utc=True, format="mixed").dt.tz_localize(None)
labels["EndTime"] = pd.to_datetime(labels["EndTime"], utc=True, format="mixed").dt.tz_localize(None)

sel_meta = channels_meta[channels_meta["Channel"].isin(SELECTED_CHANNELS)].reset_index(drop=True)
sel_labels = labels[labels["Channel"].isin(SELECTED_CHANNELS)].copy().reset_index(drop=True)
sel_labels["Duration_min"] = (sel_labels["EndTime"] - sel_labels["StartTime"]).dt.total_seconds() / 60

print(sel_meta.to_string(index=False))
print("\nAnnotation records:", len(sel_labels))
print("Anomaly events (IDs):", sel_labels["ID"].nunique())
print("Annotation period:", sel_labels["StartTime"].min(), "->", sel_labels["EndTime"].max())

# %% [markdown]
# ## 2. Parallel channel alignment to the 90-second grid
# Each channel is resampled (mean) to 30 s and 90 s grids to compare missingness; the 90 s series is kept.

# %%
jobs = [(ch, str(RAW / "channels" / f"{ch}.zip")) for ch in SELECTED_CHANNELS]
with Pool(processes=3) as pool:
    results = pool.map(U.process_channel, jobs)

grid = U.common_grid("90s")
telemetry_df = pd.DataFrame({ch: arr for ch, arr, _, _ in results}, index=grid)[SELECTED_CHANNELS]
telemetry_df.index.name = "datetime"
channel_summary = pd.DataFrame([info for _, _, info, _ in results]).set_index("Channel").loc[SELECTED_CHANNELS].reset_index()
gap_minutes = np.concatenate([g for _, _, _, g in results])
del results

pd.set_option("display.width", 200)
print(channel_summary.round(3).to_string(index=False))

# %% [markdown]
# ## 3. Choice of the common time resolution (30 s vs 90 s)

# %%
x = np.arange(len(channel_summary)); w = 0.38
plt.figure(figsize=(11, 4.5))
plt.bar(x - w/2, channel_summary["Missing_30s_%"], w, label="30 s grid")
plt.bar(x + w/2, channel_summary["Missing_90s_%"], w, label="90 s grid")
plt.xticks(x, channel_summary["Channel"], rotation=45); plt.ylabel("Missing data (%)")
plt.title("Missing data after alignment: 30 s vs 90 s grid"); plt.legend()
save_show("01_missing_30s_vs_90s.png")

coverage = []
for res in [30, 90]:
    valid = sel_labels[sel_labels["Duration_min"] * 60 >= res]
    coverage.append({"Resolution_s": res,
                     "Annotations_>=_resolution": len(valid), "Annotation_coverage_%": round(len(valid) / len(sel_labels) * 100, 2),
                     "Event_IDs_represented": valid["ID"].nunique(),
                     "Event_coverage_%": round(valid["ID"].nunique() / sel_labels["ID"].nunique() * 100, 2)})
coverage_df = pd.DataFrame(coverage)
print(coverage_df.to_string(index=False))
print("\nDecision: 90 s grid — 8 of 12 channels are natively ~90 s, so a 30 s grid leaves ~67 % of their samples empty,"
      "\nwhile annotation coverage only drops from 88.4 % to 85.8 %.")

# %% [markdown]
# ## 4. Missing-gap analysis and short-gap interpolation

# %%
steps = np.rint(gap_minutes / 1.5).astype(int)
cnt = pd.Series(steps).value_counts().sort_index()
cnt = cnt[cnt.index <= 20]
plt.figure(figsize=(10, 4.5))
plt.bar(cnt.index * 1.5, cnt.values, width=1.2, color=["#2a7" if k <= U.MAX_INTERP_STEPS else "#c55" for k in cnt.index])
plt.yscale("log"); plt.axvline(U.MAX_INTERP_STEPS * 1.5 + 0.75, color="black", ls="--", label="Interpolation limit (≤ 4.5 min)")
plt.xticks(cnt.index * 1.5, [f"{v:g}" for v in cnt.index * 1.5], rotation=90)
plt.xlabel("Missing-gap duration (minutes; 1 sample = 1.5 min)"); plt.ylabel("Number of gaps (log scale)")
plt.title("Missing-gap durations on the 90 s grid, all 12 channels (green = interpolated, red = left missing; gaps ≤ 30 min shown)")
plt.legend()
save_show("02_missing_gap_distribution_90s.png")

print("Total gaps (all channels):", len(gap_minutes))
print("Gaps <= 4.5 min: %.2f %%" % ((gap_minutes <= 4.5).mean() * 100))
print("Longest gap: %.1f min" % gap_minutes.max())
print("\nRemaining missing after short-gap interpolation:")
print(channel_summary[["Channel", "Missing_90s_%", "Interpolated_Values", "Remaining_Missing", "Remaining_Missing_%"]].round(4).to_string(index=False))

# %% [markdown]
# ## 5. Common 12-channel telemetry matrix

# %%
print("Shape:", telemetry_df.shape)
print("Range:", telemetry_df.index.min(), "->", telemetry_df.index.max())
print("Duplicate timestamps:", int(telemetry_df.index.duplicated().sum()))
print("Dtypes:", telemetry_df.dtypes.unique())
print(telemetry_df.describe().T.round(4))

# %% [markdown]
# ## 6. Anomaly-event statistics

# %%
event_channels = sel_labels.groupby("ID")["Channel"].nunique()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(np.log10(sel_labels["Duration_min"].clip(lower=0.1)), bins=40, edgecolor="white")
ax[0].set_xlabel("log10(annotation duration, minutes); point annotations (< 0.1 min) shown at −1"); ax[0].set_ylabel("Annotations")
ax[0].set_title("Annotation durations")
vc = event_channels.value_counts().sort_index()
ax[1].bar(vc.index, vc.values); ax[1].set_xticks(range(1, 13))
ax[1].set_xlabel("Selected channels annotated in the event"); ax[1].set_ylabel("Events")
ax[1].set_title("Channels per anomaly event")
save_show("03_anomaly_event_statistics.png")
print(vc.to_string())

# %% [markdown]
# ## 7. Event-level train / validation / test split (no leakage)

# %%
anomaly_ids = sorted(sel_labels["ID"].unique())
rng = random.Random(SEED); rng.shuffle(anomaly_ids)
n = len(anomaly_ids); n_train = int(n * 0.70); n_val = int(n * 0.15)
split_ids = pd.DataFrame({"ID": anomaly_ids,
                          "Split": ["Train"] * n_train + ["Validation"] * n_val + ["Test"] * (n - n_train - n_val)})
S = {s: set(split_ids.loc[split_ids.Split == s, "ID"]) for s in ["Train", "Validation", "Test"]}
assert not (S["Train"] & S["Validation"]) and not (S["Train"] & S["Test"]) and not (S["Validation"] & S["Test"])
split_summary = (sel_labels.merge(split_ids, on="ID").groupby("Split")
                 .agg(Anomaly_IDs=("ID", "nunique"), Annotation_Records=("ID", "size"))
                 .reindex(["Train", "Validation", "Test"]).reset_index())
split_summary["Record_%"] = (split_summary["Annotation_Records"] / len(sel_labels) * 100).round(2)
print(split_summary.to_string(index=False))
print("Overlap between splits: none (verified)")

# %% [markdown]
# ## 8. Known-anomaly mask and label-clean segments
# Timestamps covered by **any** annotation of the 168 events are excluded from autoencoder training.
# Unannotated data is called *label-clean*, not "proven normal".

# %%
known_mask = U.interval_mask(telemetry_df.index, sel_labels["StartTime"].values, sel_labels["EndTime"].values)
clean = ~known_mask
d = np.diff(np.concatenate([[0], clean.astype(np.int8), [0]]))
starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
clean_segments = pd.DataFrame({"Start": telemetry_df.index[starts], "End": telemetry_df.index[ends - 1],
                               "Length": ends - starts})
clean_segments["Duration_hours"] = (clean_segments["Length"] * 1.5 / 60).round(2)
print("Known-anomaly timestamps: %d (%.3f %%)" % (known_mask.sum(), known_mask.mean() * 100))
print("Label-clean timestamps:", int(clean.sum()))
print("Label-clean segments:", len(clean_segments))
print(clean_segments["Duration_hours"].describe().round(2))

plt.figure(figsize=(9, 4.2))
plt.hist(clean_segments["Duration_hours"], bins=50, edgecolor="white")
plt.xlabel("Label-clean segment duration (hours)"); plt.ylabel("Segments")
plt.title("Distribution of label-clean telemetry segments")
save_show("04_label_clean_segment_durations.png")

# %% [markdown]
# ## 9. LSTM window length

# %%
values = telemetry_df.to_numpy()
win_rows = []
for wlen in [64, 128, 256]:
    usable = clean_segments[clean_segments["Length"] >= wlen]
    total = valid = 0
    for s, L in zip(starts, ends - starts):
        for k in range(L // wlen):
            total += 1
            valid += not np.isnan(values[s + k * wlen: s + (k + 1) * wlen]).any()
    win_rows.append({"Window": wlen, "Hours": wlen * 1.5 / 60, "Usable_segments": len(usable),
                     "Candidate_windows": total, "NaN-free_windows": valid,
                     "NaN-free_%": round(valid / total * 100, 2)})
window_df = pd.DataFrame(win_rows)
print(window_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(window_df["Window"].astype(str), window_df["NaN-free_windows"], color=["#9bb", "#2a7", "#9bb"])
for i, r in window_df.iterrows():
    ax.text(i, r["NaN-free_windows"], f'{r["Hours"]:.1f} h', ha="center", va="bottom")
ax.set_xlabel("Window length (samples)"); ax.set_ylabel("NaN-free label-clean windows")
ax.set_title("Candidate LSTM window lengths (selected: 128 = 3.2 h)")
save_show("05_candidate_lstm_window_lengths.png")

# %% [markdown]
# ## 10. Save all Course 1 outputs

# %%
telemetry_df.to_parquet(OUT / "02_preprocessing" / "telemetry_90s_12channels.parquet")
pd.DataFrame({"known_anomaly": known_mask}, index=telemetry_df.index).to_parquet(OUT / "02_preprocessing" / "known_anomaly_mask.parquet")
clean_segments.to_csv(OUT / "02_preprocessing" / "label_clean_segments.csv", index=False)
channel_summary.to_csv(OUT / "02_preprocessing" / "channel_alignment_and_missing_summary.csv", index=False)
coverage_df.to_csv(OUT / "02_preprocessing" / "resolution_coverage_comparison.csv", index=False)
window_df.to_csv(OUT / "02_preprocessing" / "lstm_window_comparison.csv", index=False)
sel_meta.to_csv(OUT / "01_dataset" / "selected_channel_metadata.csv", index=False)
sel_labels.to_csv(OUT / "01_dataset" / "selected_anomaly_labels.csv", index=False)
split_ids.to_csv(OUT / "03_splits" / "anomaly_event_split.csv", index=False)
split_summary.to_csv(OUT / "03_splits" / "split_summary.csv", index=False)

summary = pd.DataFrame({"Parameter": [
    "Dataset", "Selected channels", "Time resolution", "Interpolation rule", "Telemetry rows",
    "Telemetry period", "Anomaly events", "Annotation records", "Event split (train/val/test)",
    "Known-anomaly coverage (%)", "Label-clean segments", "LSTM window", "NaN-free 128-sample clean windows"],
  "Value": [
    "ESA-ADB Mission 1", len(SELECTED_CHANNELS), "90 seconds", "gaps <= 3 samples (4.5 min) only; longer gaps left missing",
    len(telemetry_df), f"{telemetry_df.index.min()} -> {telemetry_df.index.max()}", sel_labels["ID"].nunique(), len(sel_labels),
    "/".join(str(v) for v in split_summary["Anomaly_IDs"]), round(known_mask.mean() * 100, 3), len(clean_segments),
    "128 samples (3.2 h)", int(window_df.loc[window_df.Window == 128, "NaN-free_windows"].iloc[0])]})
summary.to_csv(OUT / "02_preprocessing" / "preprocessing_summary.csv", index=False)
print(summary.to_string(index=False))
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.parts[-2] in {"01_dataset", "02_preprocessing", "03_splits", "04_plots"}:
        print(f"{p.relative_to(OUT)}  ({p.stat().st_size/1e6:.2f} MB)")
