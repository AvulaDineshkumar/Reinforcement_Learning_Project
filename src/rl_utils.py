"""Shared helpers for the corrected Satellite-Telemetry RL project."""
import zipfile
import numpy as np
import pandas as pd

GRID_START = pd.Timestamp("2000-01-01 00:00:00")
GRID_END = pd.Timestamp("2013-12-31 23:58:30")
MAX_INTERP_STEPS = 3          # 3 x 90 s = 4.5 min -> only gaps of <= 4.5 min are interpolated


def common_grid(freq="90s"):
    return pd.date_range(GRID_START, GRID_END, freq=freq)


def run_lengths(mask):
    """Lengths of consecutive True runs in a boolean array, and a per-element run length."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.array([], dtype=np.int64), np.zeros(len(mask), dtype=np.int64)
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    lengths = ends - starts
    per_elem = np.zeros(len(mask), dtype=np.int64)
    for s, l in zip(starts, lengths):
        per_elem[s:s + l] = l
    return lengths, per_elem


def process_channel(args):
    """Load one ESA-ADB channel, align it to the 90 s grid and interpolate ONLY short gaps.

    Long gaps (> MAX_INTERP_STEPS samples) are left entirely missing (no partial filling).
    """
    channel, zip_path = args
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            raw = pd.read_pickle(f)[channel].astype("float32")
    raw.index = pd.to_datetime(raw.index)

    diffs = np.diff(raw.index.asi8) / 1e9
    native = {
        "Channel": channel,
        "Raw_Rows": int(len(raw)),
        "Median_Interval_s": float(np.median(diffs)),
        "Missing_30s_%": float(raw.resample("30s", origin=GRID_START).mean()
                               .reindex(common_grid("30s")).isna().mean() * 100),
    }
    s90 = raw.resample("90s", origin=GRID_START).mean().reindex(common_grid("90s"))
    del raw

    miss = s90.isna().to_numpy()
    gaps, per_elem = run_lengths(miss)
    interp = s90.interpolate(method="time", limit_area="inside")
    fill = miss & (per_elem <= MAX_INTERP_STEPS)
    out = s90.to_numpy().copy()
    out[fill] = interp.to_numpy()[fill]
    out = out.astype("float32")

    native.update({
        "Missing_90s_%": float(miss.mean() * 100),
        "Missing_Gaps_90s": int(len(gaps)),
        "Gaps_<=4.5min": int((gaps <= MAX_INTERP_STEPS).sum()),
        "Gaps_>4.5min": int((gaps > MAX_INTERP_STEPS).sum()),
        "Longest_Gap_min": float(gaps.max() * 1.5) if len(gaps) else 0.0,
        "Interpolated_Values": int(fill.sum()),
        "Remaining_Missing": int(np.isnan(out).sum()),
        "Remaining_Missing_%": float(np.isnan(out).mean() * 100),
    })
    return channel, out, native, (gaps * 1.5).astype("float32")


def interval_mask(index, starts, ends):
    """Boolean mask over a sorted DatetimeIndex covering [start, end] intervals."""
    mask = np.zeros(len(index), dtype=bool)
    s = index.searchsorted(starts, side="left")
    e = index.searchsorted(ends, side="right")
    diff = np.zeros(len(index) + 1, dtype=np.int64)
    np.add.at(diff, s, 1)
    np.add.at(diff, e, -1)
    return np.cumsum(diff)[:-1] > 0
