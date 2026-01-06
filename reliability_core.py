# reliability_core.py
"""
Resting-state EO/EC reliability analysis.

This is a clean, importable module with no S3, no CLI, no ERP/flanker.

Expected layout for a *single client* (set externally):
- DATA_DIR points to a directory that contains:
    <MID>.parquet   : EEG dataframe
    <MID>e.parquet  : events dataframe
- JSON_PATH points to the headset geometry JSON.

The orchestration layer (dc_reliability.py) is responsible for:
- staging those parquet files and the geom JSON per client
- calling:
    build_rec_df_for_client(...)
    compute_reliability_for_client(...)
    summarize_client_conclusions(...)
"""

from __future__ import annotations

import os
import json
import gc
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import mne
from mne.io.constants import FIFF
from mne.filter import filter_data
from scipy.signal import hilbert

# ------------------------------------------------------------
# Global configuration (resting-state only)
# ------------------------------------------------------------

# These will be overridden by the driver at runtime
DATA_DIR: str = "."
JSON_PATH: str = "your_headset_geometry.json"

SUBJECT = "fsaverage"
BEM_SOL_FILE = "fsaverage-5120-5120-5120-bem-sol.fif"
FID_FILE = "fsaverage-fiducials.fif"

SFREQ = 250.0  # EEG sampling frequency (Hz) – adjust if needed

# Montage fiducial proxies (must exist in the geometry JSON)
NASION_CH = "fpz"
LPA_CH = "t7"
RPA_CH = "t8"

# Envelope/connectivity chunks
CHUNK_SEC = 10.0
MINDIST_MM = 5.0

# Frequency bands used for reliability
BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Resting periods only
PERIODS: List[str] = ["EO", "EC", "EO_EC"]

# Globals initialized once by init_forward_model()
montage: mne.channels.DigMontage | None = None
ch_names_ordered: List[str] | None = None
bem: Any = None
src: Any = None
trans: Any = None


# ------------------------------------------------------------
# Index / event helpers
# ------------------------------------------------------------

def _ensure_datetime_utc_index(df_full: pd.DataFrame) -> pd.DataFrame:
    """Ensure df_full.index is tz-aware UTC datetime (or best effort)."""
    if pd.api.types.is_datetime64_any_dtype(df_full.index):
        df2 = df_full.copy()
        try:
            if df2.index.tz is None:
                df2.index = df2.index.tz_localize("UTC")
            else:
                df2.index = df2.index.tz_convert("UTC")
        except Exception:
            pass
        return df2

    # If index is not datetime, attempt epoch seconds
    try:
        idx = pd.to_datetime(df_full.index, unit="s", utc=True)
        df2 = df_full.copy()
        df2.index = idx
        return df2
    except Exception:
        return df_full


def derive_label_windows_abs(ev: pd.DataFrame, label: str) -> List[Tuple[float, float]]:
    """
    From events dataframe with columns:
      - event_type
      - event_label
      - start_timestamp
      - end_timestamp
    compute absolute (epoch seconds) windows for a given label.
    """
    e = ev.copy()
    e["event_type_norm"] = e["event_type"].astype(str).str.strip().str.lower()
    e["event_label_norm"] = e["event_label"].astype(str).str.strip().str.lower()

    mask = (
        (e["event_type_norm"] == "epoch")
        & (e["event_label_norm"] == label.lower())
    )
    sel = e.loc[mask, ["start_timestamp", "end_timestamp"]].dropna()
    if sel.empty:
        return []

    windows_abs = [
        (float(row["start_timestamp"]), float(row["end_timestamp"]))
        for _, row in sel.iterrows()
    ]
    return windows_abs


def derive_eo_ec_windows_abs(ev: pd.DataFrame) -> List[Tuple[float, float]]:
    """Return sorted list of EO+EC windows concatenated."""
    eo = derive_label_windows_abs(ev, "eo")
    ec = derive_label_windows_abs(ev, "ec")
    all_w = eo + ec
    all_w.sort(key=lambda w: w[0])
    return all_w


def windows_to_df(df_full: pd.DataFrame,
                  windows_abs: List[Tuple[float, float]],
                  label: str = "segment") -> pd.DataFrame:
    """
    Slice df_full by epoch-second windows (UTC).
    If all windows are out of range/empty, raises.
    """
    if not windows_abs:
        raise RuntimeError(f"[{label}] No windows provided.")

    df_full = _ensure_datetime_utc_index(df_full)

    segments = []
    for (t0, t1) in windows_abs:
        try:
            seg = df_full[
                (df_full.index >= pd.to_datetime(t0, unit="s", utc=True))
                & (df_full.index <= pd.to_datetime(t1, unit="s", utc=True))
            ]
        except Exception:
            continue
        if seg.empty:
            continue
        segments.append(seg)

    if not segments:
        raise RuntimeError(f"[{label}] All windows were empty/out of range.")

    return pd.concat(segments, axis=0)


# ------------------------------------------------------------
# Geometry / montage / forward model init
# ------------------------------------------------------------

def init_forward_model(json_path: str) -> None:
    """
    Initialize montage, BEM, src, trans from headset geometry JSON + fsaverage.
    Call this once before processing a client's sessions.
    """
    global JSON_PATH, montage, ch_names_ordered, bem, src, trans

    JSON_PATH = json_path

    # --- 1) Load geometry JSON -> montage + channel order ---
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        geom = json.load(f)

    ch_names = [str(v["name"]).lower().strip() for v in geom.values()]
    seen = set()
    ch_names = [c for c in ch_names if not (c in seen or seen.add(c))]
    ch_pos_m = {}

    for v in geom.values():
        nm = str(v["name"]).lower().strip()
        xyz_mm = np.array(v["coords"], dtype=float)
        ch_pos_m[nm] = xyz_mm / 1000.0  # mm -> m

    for key in (NASION_CH, LPA_CH, RPA_CH):
        if key not in ch_pos_m:
            raise RuntimeError(f"Missing fiducial proxy channel in JSON: '{key}'")

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos_m,
        nasion=ch_pos_m[NASION_CH],
        lpa=ch_pos_m[LPA_CH],
        rpa=ch_pos_m[RPA_CH],
        coord_frame="head",
    )
    ch_names_ordered = ch_names

    # --- 2) fsaverage + BEM / fiducials ---
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=True)
    if os.path.basename(os.path.normpath(subjects_dir)).lower() == "fsaverage":
        subjects_dir = os.path.dirname(os.path.normpath(subjects_dir))

    bem_sol_fname = os.path.join(subjects_dir, SUBJECT, "bem", BEM_SOL_FILE)
    fid_fname = os.path.join(subjects_dir, SUBJECT, "bem", FID_FILE)

    if not os.path.exists(bem_sol_fname):
        raise RuntimeError(f"Missing BEM solution: {bem_sol_fname}")
    if not os.path.exists(fid_fname):
        raise RuntimeError(f"Missing fiducials file: {fid_fname}")

    bem_ = mne.read_bem_solution(bem_sol_fname)
    mri_fids, _ = mne.io.read_fiducials(fid_fname)

    def _dig_r(d):
        return d["r"] if isinstance(d, dict) else d.r

    def pick_fid(fid_list, ident_code, label):
        for d in fid_list:
            kind = d.get("kind", None) if isinstance(d, dict) else getattr(d, "kind", None)
            ident = d.get("ident", None) if isinstance(d, dict) else getattr(d, "ident", None)
            if kind == FIFF.FIFFV_POINT_CARDINAL and ident == ident_code:
                return _dig_r(d)
            if kind == ident_code:
                return _dig_r(d)
        raise RuntimeError(f"Could not find fiducial: {label}")

    nasion = pick_fid(mri_fids, FIFF.FIFFV_POINT_NASION, "nasion")
    lpa = pick_fid(mri_fids, FIFF.FIFFV_POINT_LPA, "LPA")
    rpa = pick_fid(mri_fids, FIFF.FIFFV_POINT_RPA, "RPA")

    # Create src space
    src_ = mne.setup_source_space(
        SUBJECT,
        spacing="ico5",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )

    # A simple native head transform using the montage
    info_tmp = mne.create_info(ch_names=list(ch_pos_m.keys()),
                               sfreq=SFREQ,
                               ch_types="eeg")
    raw_tmp = mne.io.RawArray(np.zeros((len(ch_pos_m), 100)), info_tmp, verbose=False)
    raw_tmp.set_montage(montage, on_missing="ignore")
    trans_ = mne.channels.compute_native_head_t(raw_tmp.info)

    montage = montage
    ch_names_ordered = ch_names
    bem = bem_
    src = src_
    trans = trans_
    print("Forward model initialized for resting-state reliability.")


# ------------------------------------------------------------
# Connectivity / metric helpers
# ------------------------------------------------------------

def yeo7_from_schaefer_name(name: str) -> str:
    """
    Map Schaefer parcel name to Yeo-7 network label.

    Assumes names containing substrings like '_VIS_', '_SM_', '_DMN_', etc.
    """
    nm = str(name).upper()
    for net in ["VIS", "SM", "DAN", "VAN", "LIM", "FP", "DMN"]:
        if f"_{net}_" in nm:
            return net
    return "UNK"


def within_network_mean(C: np.ndarray, idx: List[int]) -> float:
    """Mean connectivity within indices idx of C."""
    if len(idx) < 2:
        return np.nan
    sub = C[np.ix_(idx, idx)]
    iu = np.triu_indices_from(sub, k=1)
    vals = sub[iu]
    return float(np.nanmean(vals)) if vals.size > 0 else np.nan


def envelope_corr(ts: np.ndarray,
                  sfreq: float,
                  fmin: float,
                  fmax: float) -> np.ndarray:
    """
    Bandpass filter, Hilbert, and compute envelope correlation.
    ts: (n_nodes, n_times)
    """
    data_f = filter_data(
        ts,
        sfreq=sfreq,
        l_freq=fmin,
        h_freq=fmax,
        method="iir",
        verbose=False,
    )
    analytic = hilbert(data_f, axis=-1)
    env = np.abs(analytic)
    C = np.corrcoef(env)
    return C


def hub_strength(C: np.ndarray) -> np.ndarray:
    """Hub strength: absolute sum of correlations per node."""
    C = np.asarray(C, dtype=float)
    np.fill_diagonal(C, 0.0)
    return np.sum(np.abs(C), axis=1)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def rankdata(x):
        temp = np.argsort(x)
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(x))
        return ranks

    ra = rankdata(a)
    rb = rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_icc_2_1(X: np.ndarray) -> float:
    """
    ICC(2,1) as in Shrout & Fleiss.
    X: features x sessions
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape

    mean_per_feature = X.mean(axis=1, keepdims=True)
    mean_per_session = X.mean(axis=0, keepdims=True)
    grand_mean = X.mean()

    ssw = ((X - mean_per_feature) ** 2).sum()
    ssb = (k * ((mean_per_feature - grand_mean) ** 2)).sum()
    sss = (n * ((mean_per_session - grand_mean) ** 2)).sum()

    msb = ssb / (n - 1)
    msw = ssw / (n * (k - 1))
    mss = sss / (k - 1)

    icc = (msb - msw) / (msb + (k - 1) * msw + (k * (mss - msw)) / n)
    return float(icc)


def zscore_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if not np.isfinite(s) or s == 0.0:
        return np.zeros_like(v)
    return (v - m) / s


# ------------------------------------------------------------
# Period extraction + source pipeline
# ------------------------------------------------------------

def build_raw_from_df_period(df_period: pd.DataFrame,
                             ch_order: List[str],
                             mont: mne.channels.DigMontage,
                             label: str = "segment") -> mne.io.BaseRaw:
    """
    Convert a dataframe (channels as columns, time as index) to an MNE Raw.
    """
    df_period = df_period.copy()
    for c in df_period.columns:
        if df_period[c].dtype == "O":
            df_period[c] = pd.to_numeric(df_period[c], errors="coerce")

    df_period = df_period[ch_order].astype(float)
    data = df_period.T.values

    info = mne.create_info(ch_names=list(df_period.columns),
                           sfreq=SFREQ,
                           ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    raw.set_montage(mont, on_missing="ignore")
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()

    return raw


def run_profile_and_hubs_multiband(raw_p: mne.io.BaseRaw) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Core pipeline (multi-band):
      - forward model (fsaverage, BEM, trans, src)
      - noise covariance (shrunk)
      - min-norm inverse (dSPM)
      - Schaefer 200 ROI time series
      - envelope connectivity per band
      - node hub strengths & network profiles per band

    Returns:
        dict:
            band_name -> {
                "prof": network profile (vector),
                "hubs": node hub strengths,
                "hubs_abs": abs(hubs),
            }
    """
    sfreq = float(raw_p.info["sfreq"])

    fwd = mne.make_forward_solution(
        raw_p.info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=MINDIST_MM / 1000.0,
        n_jobs=1,
        verbose=False,
    )

    noise_cov = mne.compute_raw_covariance(
        raw_p,
        rank="info",
        method="shrunk",
        verbose=False,
    )

    inv = mne.minimum_norm.make_inverse_operator(
        raw_p.info,
        fwd,
        noise_cov,
        loose=0.2,
        depth=0.8,
        verbose=False,
    )

    labels = mne.read_labels_from_annot(
        SUBJECT,
        parc="Schaefer2018_200Parcels_7Networks_order",
        subjects_dir=mne.datasets.fetch_fsaverage(verbose=False),
        verbose=False,
    )

    stcs = mne.minimum_norm.apply_inverse_raw(
        raw_p,
        inv,
        lambda2=1.0 / 9.0,
        method="dSPM",
        pick_ori=None,
        verbose=False,
    )

    label_ts = mne.extract_label_time_course(
        stcs,
        labels,
        inv["src"],
        mode="mean_flip",
        return_generator=False,
        verbose=False,
    )

    nets = [yeo7_from_schaefer_name(lb.name) for lb in labels]
    uniq_nets = sorted(set(nets))

    band_dict: Dict[str, Dict[str, np.ndarray]] = {}

    for band_name, (fmin, fmax) in BANDS.items():
        C = envelope_corr(label_ts, sfreq, fmin, fmax)
        hubs = hub_strength(C)
        hubs_abs = np.abs(hubs)

        prof_vals = []
        for net in uniq_nets:
            idx = [i for i, n in enumerate(nets) if n == net]
            prof_vals.append(within_network_mean(C, idx))
        prof_vals = np.array(prof_vals, dtype=float)

        band_dict[band_name] = {
            "prof": prof_vals,
            "hubs": hubs,
            "hubs_abs": hubs_abs,
        }

    return band_dict


def load_df_and_events(session_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load EEG + events parquet for a single MID from DATA_DIR."""
    eeg_path = os.path.join(DATA_DIR, f"{session_id}.parquet")
    ev_path = os.path.join(DATA_DIR, f"{session_id}e.parquet")

    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"EEG parquet not found for session '{session_id}': {eeg_path}")
    if not os.path.exists(ev_path):
        raise FileNotFoundError(f"Events parquet not found for session '{session_id}': {ev_path}")

    df = pd.read_parquet(eeg_path)
    ev = pd.read_parquet(ev_path)
    df = _ensure_datetime_utc_index(df)
    return df, ev


def extract_period_raw(df: pd.DataFrame,
                       ev: pd.DataFrame,
                       period_name: str) -> mne.io.BaseRaw:
    """
    Extract a period (EO, EC, EO_EC) from df using events and build an MNE Raw.
    """
    if period_name.upper() == "EC":
        windows_abs = derive_label_windows_abs(ev, "ec")
    elif period_name.upper() == "EO":
        windows_abs = derive_label_windows_abs(ev, "eo")
    elif period_name.upper() == "EO_EC":
        windows_abs = derive_eo_ec_windows_abs(ev)
    else:
        raise ValueError(f"Unknown period: {period_name}")

    df_per = windows_to_df(df, windows_abs, label=period_name)
    raw_p = build_raw_from_df_period(df_per, ch_names_ordered, montage, label=period_name)
    return raw_p


# ------------------------------------------------------------
# Public API used by dc_reliability.py
# ------------------------------------------------------------

def build_rec_df_for_client(
    client_id: str,
    mids: List[str],
    data_dir: str,
    json_path: str,
) -> pd.DataFrame:
    """
    Build per-period, per-band metrics table (rec_df) for a client.

    Each row represents one (client, session, period, band) combination with:
      - prof: 7-network profile vector
      - hubs: node hub strengths vector
      - hubs_abs: |hubs|
      - and their z-scored versions.
    """
    global DATA_DIR
    DATA_DIR = data_dir

    init_forward_model(json_path)

    records: List[Dict[str, Any]] = []

    for sess in mids:
        df, ev = load_df_and_events(sess)

        for period in PERIODS:
            try:
                raw_p = extract_period_raw(df, ev, period)
            except Exception as e:
                print(f"[{client_id} | {sess} | {period}] skipped: {e}")
                continue

            band_dict = run_profile_and_hubs_multiband(raw_p)

            for band_name, vals in band_dict.items():
                records.append({
                    "client_id": client_id,
                    "session": sess,
                    "period": period,
                    "band": band_name,
                    "prof": np.asarray(vals["prof"], dtype=float),
                    "hubs": np.asarray(vals["hubs"], dtype=float),
                    "hubs_abs": np.asarray(vals["hubs_abs"], dtype=float),
                })

        gc.collect()

    rec_df = pd.DataFrame.from_records(records)

    # Z-normalized versions
    if not rec_df.empty:
        rec_df["prof_z"] = rec_df["prof"].apply(zscore_vector)
        rec_df["hubs_z"] = rec_df["hubs"].apply(zscore_vector)
        rec_df["hubs_abs_z"] = rec_df["hubs_abs"].apply(zscore_vector)

    return rec_df


def reliability_by_period_band(
    rec_df: pd.DataFrame,
    period: str,
    band: str,
    base_measure: str = "prof",
    norm_type: str = "raw",
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Compute pairwise correlations and ICC(2,1) for a given
    (period, band, base_measure, norm_type) combination.

    base_measure ∈ {"prof", "hubs", "hubs_abs"}
    norm_type ∈ {"raw", "z"}
    """
    if norm_type == "raw":
        col_name = base_measure
    elif norm_type == "z":
        col_name = base_measure + "_z"
    else:
        raise ValueError("norm_type must be 'raw' or 'z'.")

    sub = rec_df[(rec_df["period"] == period) & (rec_df["band"] == band)].copy()
    sessions = sorted(sub["session"].unique())

    if len(sessions) < 2:
        return None, None

    pairs = []
    corrs = []

    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            s1, s2 = sessions[i], sessions[j]
            row1 = sub[sub["session"] == s1].iloc[0]
            row2 = sub[sub["session"] == s2].iloc[0]

            v1 = row1[col_name]
            v2 = row2[col_name]

            if base_measure == "prof":
                r = float(np.corrcoef(v1, v2)[0, 1])
            elif base_measure in ("hubs", "hubs_abs"):
                r = spearman_corr(v1, v2)
            else:
                raise ValueError(f"Unsupported base_measure: {base_measure}")

            pairs.append((s1, s2))
            corrs.append(r)

    detail = pd.DataFrame({
        "period": period,
        "band": band,
        "measure": base_measure,
        "norm_type": norm_type,
        "session1": [p[0] for p in pairs],
        "session2": [p[1] for p in pairs],
        "r": corrs,
    })

    # Build matrix for ICC
    vecs = []
    for s in sessions:
        row = sub[sub["session"] == s].iloc[0]
        v = np.asarray(row[col_name], dtype=float)
        vecs.append(v)
    X = np.column_stack(vecs)
    icc = compute_icc_2_1(X)

    summary = pd.DataFrame([{
        "period": period,
        "band": band,
        "measure": base_measure,
        "norm_type": norm_type,
        "n_sessions": len(sessions),
        "n_pairs": len(corrs),
        "mean_r": float(np.mean(corrs)),
        "min_r": float(np.min(corrs)),
        "max_r": float(np.max(corrs)),
        "icc_2_1": icc,
    }])

    return summary, detail


def compute_reliability_for_client(
    client_id: str,
    rec_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run reliability_by_period_band across all periods/bands/measures/norms
    and concatenate into:
      - rel_summary (one row per combination)
      - rel_detail (one row per session pair per combination)
    """
    all_summaries = []
    all_details = []

    measures = ["prof", "hubs", "hubs_abs"]
    norms = ["raw", "z"]

    for period in PERIODS:
        for band in BANDS.keys():
            for base_measure in measures:
                for norm_type in norms:
                    summ, det = reliability_by_period_band(
                        rec_df,
                        period,
                        band,
                        base_measure=base_measure,
                        norm_type=norm_type,
                    )
                    if summ is not None:
                        all_summaries.append(summ)
                        all_details.append(det)

    if not all_summaries:
        return pd.DataFrame(), pd.DataFrame()

    rel_summary = pd.concat(all_summaries, ignore_index=True)
    rel_detail = pd.concat(all_details, ignore_index=True)

    rel_summary.insert(0, "client_id", client_id)
    rel_detail.insert(0, "client_id", client_id)

    return rel_summary, rel_detail


def summarize_client_conclusions(
    client_id: str,
    client_summary: pd.DataFrame,
    icc_threshold: float = 0.75,
) -> Tuple[str, pd.DataFrame]:
    """
    Create a short text summary from the reliability summary table.
    """
    if client_summary.empty:
        text = f"Client {client_id}: no valid data available for resting-state reliability."
        return text, client_summary

    if "icc_2_1" in client_summary.columns:
        n_good = (client_summary["icc_2_1"] >= icc_threshold).sum()
        n_total = len(client_summary)
        text = f"Client {client_id}: {n_good}/{n_total} metric combinations show ICC(2,1) ≥ {icc_threshold:.2f}."
    else:
        text = f"Client {client_id}: reliability summary available (no ICC column)."

    return text, client_summary
