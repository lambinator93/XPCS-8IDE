from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import hdf5plugin
import numpy as np

import matplotlib as mpl
mpl.use("macosx")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass
class RunData:
    # paths
    raw_path: Path
    meta_path: Path
    results_path: Path

    # open handles (raw + metadata stay open so you can read frames lazily)
    f_raw: h5py.File
    dset_raw: h5py.Dataset
    f_meta: h5py.File

    # processed arrays (loaded into memory)
    dynamic_roi_map: np.ndarray
    scattering_2d: np.ndarray
    ttc: np.ndarray
    g2: np.ndarray

    def close(self) -> None:
        """Close open HDF5 handles."""
        try:
            if self.f_raw:
                self.f_raw.close()
        finally:
            self.f_raw = None  # type: ignore
        try:
            if self.f_meta:
                self.f_meta.close()
        finally:
            self.f_meta = None  # type: ignore


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_processed_results(base_dir: Path, sample_id: str) -> Path:
    """
    Example target:
      <BASE_DIR>/Twotime_PostExpt_01/A073_*_results.hdf
    """
    proc_dir = base_dir / "Twotime_PostExpt_01"
    matches = sorted(proc_dir.glob(f"{sample_id}_*_results.hdf"))
    if not matches:
        raise FileNotFoundError(f"No processed results found in {proc_dir} matching {sample_id}_*_results.hdf")
    return matches[0]


def find_raw_run_dir(base_dir: Path, sample_id: str) -> Path:
    """
    Finds the raw run folder:
      <BASE_DIR>/data/<RUN_NAME>/
    where RUN_NAME starts with SAMPLE_ID, e.g. A073_IPA_NBH_...
    """
    data_dir = base_dir / "data"
    matches = sorted([p for p in data_dir.glob(f"{sample_id}_*") if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No raw run directory found in {data_dir} starting with {sample_id}_")
    return matches[0]


def find_raw_data_files(run_dir: Path) -> tuple[Path, Path]:
    """
    Inside run_dir, find:
      - raw images file: <RUN_NAME>.h5  (or .hdf/.hdf5)
      - metadata file:   <RUN_NAME>_metadata.hdf (or .h5/.hdf5)

    Returns (raw_path, meta_path).
    """
    run_name = run_dir.name

    raw_candidates = [
        run_dir / f"{run_name}.h5",
        run_dir / f"{run_name}.hdf",
        run_dir / f"{run_name}.hdf5",
        run_dir / f"{run_name}.h5py",
    ]
    raw_path = _first_existing(raw_candidates)

    meta_candidates = [
        run_dir / f"{run_name}_metadata.hdf",
        run_dir / f"{run_name}_metadata.h5",
        run_dir / f"{run_name}_metadata.hdf5",
    ]
    meta_path = _first_existing(meta_candidates)

    if raw_path is None:
        raise FileNotFoundError(f"Could not find raw data file for run {run_name} in {run_dir}")
    if meta_path is None:
        raise FileNotFoundError(f"Could not find metadata file for run {run_name} in {run_dir}")

    return raw_path, meta_path


def load_run_data(
    base_dir: Path,
    sample_id: str,
    *,
    mask_n: int,
    scattering_first_frame_only: bool = True,
) -> RunData:
    """
    Loads:
      - raw image dataset handle: entry/data/data  (lazy; not read into RAM)
      - raw metadata file handle (lazy)
      - processed arrays from results.hdf into memory:
          dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
          scattering_2d   = f["xpcs/temporal_mean/scattering_2d"][...]
          ttc             = f["xpcs/twotime/correlation_map/c2_00{mask_n:03d}"][...]
          g2              = f["xpcs/twotime/normalized_g2"][...]
    """
    # Needed for compressed detector data (registers HDF5 filters)
    import hdf5plugin  # noqa: F401

    # --- locate files ---
    results_path = find_processed_results(base_dir, sample_id)

    run_dir = find_raw_run_dir(base_dir, sample_id)
    raw_path, meta_path = find_raw_data_files(run_dir)

    # --- open raw + metadata (keep open) ---
    f_raw = h5py.File(raw_path, "r")
    if "entry/data/data" not in f_raw:
        raise KeyError(f"Raw file missing dataset 'entry/data/data': {raw_path}")
    dset_raw = f_raw["entry/data/data"]

    f_meta = h5py.File(meta_path, "r")

    # --- load processed arrays (into memory) ---
    ttc_path = f"xpcs/twotime/correlation_map/c2_00{int(mask_n):03d}"
    with h5py.File(results_path, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        if scattering_first_frame_only and scattering_2d.ndim == 3:
            scattering_2d = scattering_2d[0, :, :]
        ttc = f[ttc_path][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]

    return RunData(
        raw_path=raw_path,
        meta_path=meta_path,
        results_path=results_path,
        f_raw=f_raw,
        dset_raw=dset_raw,
        f_meta=f_meta,
        dynamic_roi_map=dynamic_roi_map,
        scattering_2d=scattering_2d,
        ttc=ttc,
        g2=g2,
    )

def _make_roi_boolean_mask(dynamic_roi_map, mask_n: int):
    """
    Tries to build a boolean mask from dynamic_roi_map for the given mask_n.

    Assumptions (common in XPCS pipelines):
      - dynamic_roi_map is an integer label image
      - pixels belonging to ROI k have value k (or sometimes k-1)

    This function tries:
      1) map == mask_n
      2) map == (mask_n - 1)
    """
    m = np.asarray(dynamic_roi_map)

    if m.ndim != 2:
        raise ValueError(f"dynamic_roi_map should be 2D, got {m.shape}")

    mask = (m == int(mask_n))
    if np.any(mask):
        return mask, int(mask_n)

    mask2 = (m == int(mask_n) - 1)
    if np.any(mask2):
        return mask2, int(mask_n) - 1

    # helpful debug info
    vals = np.unique(m[:: max(1, m.shape[0] // 64), :: max(1, m.shape[1] // 64)])
    raise ValueError(
        f"No pixels matched mask_n={mask_n} (or mask_n-1). "
        f"Sample of unique dynamic_roi_map values: {vals[:30]}{'...' if vals.size > 30 else ''}"
    )


def _apply_mask_and_clip(frame2d, roi_mask, clip_percentile: float):
    """
    Returns (masked_frame_float, vmin, vmax).

    - Outside ROI => NaN (so it won't drive percentiles)
    - Clip inside ROI at [0, clip_percentile] percentiles to suppress hot pixels
    """
    img = np.asarray(frame2d).astype(np.float32, copy=False)

    out = img.copy()
    out[~roi_mask] = np.nan

    # percentile computed on ROI pixels only
    roi_vals = out[roi_mask]
    roi_vals = roi_vals[np.isfinite(roi_vals)]

    if roi_vals.size == 0:
        return out, 0.0, 1.0

    p = float(clip_percentile)
    p = np.clip(p, 0.0, 100.0)

    lo = np.percentile(roi_vals, 0.0)
    hi = np.percentile(roi_vals, p)

    if not np.isfinite(lo):
        lo = float(np.nanmin(out))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    # clip only ROI pixels; keep outside as NaN
    out_roi = np.clip(out[roi_mask], lo, hi)
    out[roi_mask] = out_roi

    return out, float(lo), float(hi)


def launch_masked_raw_viewer(
    run,
    *,
    mask_n: int,
    start_frame: int = 0,
    clip_percentile_init: float = 99.9,
    cmap: str = "magma",
):
    """
    Interactive viewer:
      - Left/right arrow keys: previous/next frame
      - Slider: clip percentile (suppresses hot pixels within ROI)
      - Shows ONLY the ROI pixels (outside ROI is NaN)

    Requires:
      run.dset_raw           (HDF5 dataset: frames, ny, nx)
      run.dynamic_roi_map    (2D label map)
    """
    dset = run.dset_raw
    n_frames = int(dset.shape[0])

    roi_mask, used_label = _make_roi_boolean_mask(run.dynamic_roi_map, mask_n)

    # --- find ROI centre once ---
    cy, cx = roi_center_from_label_map(run.dynamic_roi_map, mask_n)

    crop_h = 100
    crop_w = 50

    i0 = int(np.clip(start_frame, 0, n_frames - 1))

    fig = plt.figure(figsize=(5.0, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.12], hspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax_slider = fig.add_subplot(gs[1, 0])
    ax_slider.axis("off")

    state = {
        "i": i0,
        "clip_p": float(clip_percentile_init),
    }

    # initial frame
    frame_full = dset[state["i"], :, :]
    frame, _ = crop_around_center(
        frame_full, cy, cx,
        crop_h=crop_h, crop_w=crop_w
    )
    roi_crop, _ = crop_around_center(
        roi_mask.astype(float), cy, cx,
        crop_h=crop_h, crop_w=crop_w
    )
    roi_crop = roi_crop > 0

    masked, vmin, vmax = _apply_mask_and_clip(frame, roi_crop, state["clip_p"])

    im = ax.imshow(masked, origin="upper", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"Raw frame {state['i']}/{n_frames-1}  |  ROI label={used_label}  |  clip p={state['clip_p']:.2f}")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("ADU (clipped)")

    # slider (placed manually within the slider row)
    bbox = ax_slider.get_position()
    x0, y0, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height

    axp = fig.add_axes([x0 + 0.10 * w, y0 + 0.35 * h, 0.82 * w, 0.40 * h])
    s_clip = Slider(axp, "clip percentile", 90.0, 100.0, valinit=state["clip_p"])

    def _redraw():
        frame_full = dset[state["i"], :, :]
        frame, _ = crop_around_center(frame_full, cy, cx, crop_h=100, crop_w=50)

        roi_crop, _ = crop_around_center(
            roi_mask.astype(float), cy, cx,
            crop_h=100, crop_w=50
        )
        roi_crop = roi_crop > 0

        masked, vmin, vmax = _apply_mask_and_clip(frame, roi_crop, state["clip_p"])
        im.set_data(masked)
        im.set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(
            f"Raw frame {state['i']}/{n_frames-1}  |  ROI label={used_label}  |  clip p={state['clip_p']:.2f}"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["i"] = min(n_frames - 1, state["i"] + 1)
            _redraw()
        elif event.key in ("left", "a"):
            state["i"] = max(0, state["i"] - 1)
            _redraw()
        elif event.key in ("home",):
            state["i"] = 0
            _redraw()
        elif event.key in ("end",):
            state["i"] = n_frames - 1
            _redraw()

    def on_clip(_val):
        state["clip_p"] = float(s_clip.val)
        _redraw()

    s_clip.on_changed(on_clip)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    return {"roi_label_used": used_label, "roi_pixel_count": int(np.sum(roi_mask))}

def roi_center_from_label_map(dynamic_roi_map, mask_n: int):
    """
    dynamic_roi_map: 2D int array, same shape as detector image (rows, cols)
    mask_n: ROI label (e.g. 145)

    Returns (cy, cx) in pixel indices (row, col).
    """
    ys, xs = np.where(dynamic_roi_map == int(mask_n))
    if ys.size == 0:
        raise ValueError(f"mask_n={mask_n} not found in dynamic_roi_map")
    cy = int(np.round(np.mean(ys)))
    cx = int(np.round(np.mean(xs)))
    return cy, cx


def crop_around_center(img2d, cy: int, cx: int, *, crop_h: int = 100, crop_w: int = 50):
    """
    Returns cropped view of img2d centered at (cy, cx), clipped to image bounds.
    crop_h, crop_w are total output sizes.
    """
    H, W = img2d.shape
    hh = crop_h // 2
    hw = crop_w // 2

    y0 = max(0, cy - hh)
    y1 = min(H, cy + hh + (crop_h % 2))
    x0 = max(0, cx - hw)
    x1 = min(W, cx + hw + (crop_w % 2))

    return img2d[y0:y1, x0:x1], (y0, y1, x0, x1)

def inspect_raw_mask_oscillations(
    run: RunData,
    *,
    mask_signal: int,
    mask_control: int,
    dt_s: float = 1.0,
    fmin: float = 1 / 1000,
    fmax: float = 1 / 10,
    detrend: bool = True,
    window: bool = True,
    figsize=(12.5, 7.0),
):
    """
    Compare raw-intensity oscillations between a signal ROI and a control ROI.

    Uses raw detector frames (run.dset_raw) and dynamic_roi_map.
    """

    dset = run.dset_raw
    n_frames = int(dset.shape[0])
    t = np.arange(n_frames) * float(dt_s)

    # --- build masks ---
    mask_sig, used_sig = _make_roi_boolean_mask(run.dynamic_roi_map, mask_signal)
    mask_ctl, used_ctl = _make_roi_boolean_mask(run.dynamic_roi_map, mask_control)

    # --- extract summed intensity traces ---
    y_sig = np.zeros(n_frames, dtype=np.float64)
    y_ctl = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        frame = dset[i, :, :]
        y_sig[i] = np.sum(frame[mask_sig])
        y_ctl[i] = np.sum(frame[mask_ctl])

    # --- preprocessing helper ---
    def preprocess_and_fft(y):
        y = y.astype(np.float64)
        y = y - np.mean(y)

        if detrend:
            A = np.column_stack([t, np.ones_like(t)])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            y = y - (A @ beta)

        if window:
            y = y * np.hanning(len(y))

        F = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), d=dt_s)
        power = np.abs(F) ** 2

        m = (freqs >= fmin) & (freqs <= fmax)
        return y, freqs[m], power[m]

    y_sig_p, f_sig, P_sig = preprocess_and_fft(y_sig)
    y_ctl_p, f_ctl, P_ctl = preprocess_and_fft(y_ctl)

    # --- plotting ---
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    axs[0, 0].plot(t, y_sig_p, lw=1.4, color="C3")
    axs[0, 0].set_title(f"Signal mask {used_sig} (raw intensity)")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Intensity (a.u.)")

    axs[0, 1].plot(t, y_ctl_p, lw=1.4, color="0.3")
    axs[0, 1].set_title(f"Control mask {used_ctl} (raw intensity)")
    axs[0, 1].set_xlabel("Time [s]")

    axs[1, 0].plot(f_sig, P_sig, lw=1.8, color="C3")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 0].set_ylabel("Power")

    axs[1, 1].plot(f_ctl, P_ctl, lw=1.8, color="0.3")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("Frequency [Hz]")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "signal": {"t": t, "y": y_sig, "freqs": f_sig, "power": P_sig},
        "control": {"t": t, "y": y_ctl, "freqs": f_ctl, "power": P_ctl},
    }


def extract_roi_intensity_matrix(
    dset_raw,
    *,
    dynamic_roi_map=None,
    mask_n: int | None = None,
    roi_mask=None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    dtype=np.float32,
):
    """
    Build the per-pixel intensity matrix for one ROI from the raw frames.

    Returns
    -------
    I : (T, P) array
        I[t, p] is the intensity of ROI pixel p at time/frame t.
        T is number of selected frames, P is number of ROI pixels.
    frame_idxs : (T,) array
        The raw frame indices used (accounts for start/stop/stride).

    Notes
    -----
    - You may pass either:
        (A) roi_mask= (bool 2D array), OR
        (B) dynamic_roi_map= (int 2D array) AND mask_n= (ROI label)
      If roi_mask is provided, it is used directly.
    - Reads frames lazily from the HDF5 dataset (no full load).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    n_frames = int(dset_raw.shape[0])
    if stop is None:
        stop = n_frames
    start = int(np.clip(start, 0, n_frames))
    stop = int(np.clip(stop, 0, n_frames))
    if stop <= start:
        raise ValueError(f"Invalid frame range: start={start}, stop={stop}")

    # --- resolve ROI mask ---
    if roi_mask is not None:
        m = np.asarray(roi_mask).astype(bool, copy=False)
        if m.ndim != 2:
            raise ValueError(f"roi_mask must be 2D, got shape {m.shape}")
    else:
        if dynamic_roi_map is None or mask_n is None:
            raise ValueError("Provide either roi_mask=..., or (dynamic_roi_map=... and mask_n=...)")
        m, _ = _make_roi_boolean_mask(dynamic_roi_map, int(mask_n))  # uses your existing helper

    P = int(np.sum(m))
    if P <= 0:
        raise ValueError("ROI mask has zero pixels")

    frame_idxs = np.arange(start, stop, stride, dtype=int)
    T = int(frame_idxs.size)

    # Preallocate output: (T, P)
    I = np.empty((T, P), dtype=dtype)

    # Read each frame lazily and vectorize ROI pixels
    for j, fi in enumerate(frame_idxs):
        frame = dset_raw[int(fi), :, :]
        I[j, :] = np.asarray(frame)[m].astype(dtype, copy=False)

    return I, frame_idxs


def ttc_corr_and_g_from_I(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Corr-TTC and G-TTC from intensity matrix I (T, P),
    where averaging is over pixels.

    Corr(t1,t2) = <I(t1)I(t2)> / (<I(t1)><I(t2)>)
    G(t1,t2)    = (<I(t1)I(t2)> - <I(t1)><I(t2)>) / (sigma(t1)sigma(t2))

    Returns (Corr, G), each shape (T, T).
    """
    I = np.asarray(I, dtype=np.float64)
    T, P = I.shape
    if T < 2 or P < 2:
        raise ValueError(f"Need at least 2 frames and 2 pixels, got I={I.shape}")

    # <I(t)> over pixels
    mu = I.mean(axis=1)  # (T,)

    # <I(t1)I(t2)> over pixels:
    # mean over pixels of product = (I @ I.T) / P
    cross = (I @ I.T) / float(P)  # (T,T)

    # Corr TTC
    denom_corr = mu[:, None] * mu[None, :]
    Corr = cross / np.where(denom_corr == 0, np.nan, denom_corr)

    # G TTC
    var = (I * I).mean(axis=1) - mu * mu
    var = np.clip(var, 0.0, np.inf)
    sigma = np.sqrt(var)  # (T,)
    denom_g = sigma[:, None] * sigma[None, :]
    G = (cross - denom_corr) / np.where(denom_g == 0, np.nan, denom_g)

    return Corr, G


def symmetrize_ttc(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    return C + C.T - np.diag(np.diag(C))


def clip_ttc(C: np.ndarray, p_hi: float = 99.9) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    lo, hi = np.nanpercentile(C, [0.0, float(p_hi)])
    return np.clip(C, lo, hi)


def plot_corr_vs_g_ttc(
    Corr: np.ndarray,
    G: np.ndarray,
    *,
    clip_hi_percentile: float = 99.9,
    cmap: str = "plasma",
    figsize=(13.0, 4.6),
):
    """
    Side-by-side:
      Left: Corr-TTC (symmetrized, clipped)
      Mid : G-TTC    (symmetrized, clipped)
      Right: (G - (Corr - 1)) as a diagnostic map

    The diagnostic uses the fully coherent / stable mean relation:
      G ≈ Corr - 1   (see their discussion around eqs. 3–4)  [oai_citation:2‡Ragulskaya et al. - 2024 - On the analysis of two-time correlation functions equilibrium versus non-equilibrium systems.pdf](sediment://file_00000000923c71f88df245b998ba4918)
    """
    Cc = symmetrize_ttc(Corr)
    Cg = symmetrize_ttc(G)

    Cc_plot = clip_ttc(Cc, p_hi=clip_hi_percentile)
    Cg_plot = clip_ttc(Cg, p_hi=clip_hi_percentile)

    # Diagnostic map: how far from G = Corr - 1
    D = Cg - (Cc - 1.0)
    D_plot = clip_ttc(D, p_hi=clip_hi_percentile)

    fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"wspace": 0.30})

    im0 = axs[0].imshow(Cc_plot, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    axs[0].set_title("Corr-TTC")
    axs[0].set_xlabel("t₁ index")
    axs[0].set_ylabel("t₂ index")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(Cg_plot, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    axs[1].set_title("G-TTC")
    axs[1].set_xlabel("t₁ index")
    axs[1].set_ylabel("t₂ index")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(D_plot, origin="lower", cmap="magma", interpolation="nearest", aspect="equal")
    axs[2].set_title("Diagnostic:  G - (Corr - 1)")
    axs[2].set_xlabel("t₁ index")
    axs[2].set_ylabel("t₂ index")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    plt.tight_layout()
    plt.show()

def _slice_to_start_stop_stride(frame_slice: slice | None, n_frames: int) -> tuple[int, int, int]:
    """
    Convert a Python slice into (start, stop, stride) with bounds clipped to [0, n_frames].

    Examples
    --------
    None             -> (0, n_frames, 1)
    slice(0, 4800)   -> (0, 4800, 1)
    slice(100, 2000, 2) -> (100, 2000, 2)
    """
    if frame_slice is None:
        return 0, int(n_frames), 1

    start = 0 if frame_slice.start is None else int(frame_slice.start)
    stop = n_frames if frame_slice.stop is None else int(frame_slice.stop)
    stride = 1 if frame_slice.step is None else int(frame_slice.step)

    if stride == 0:
        raise ValueError("slice.step cannot be 0")

    # Clip to valid range
    start = max(0, min(int(n_frames), start))
    stop = max(0, min(int(n_frames), stop))

    # Ensure forward slicing (you can add reverse support later if you want)
    if stride < 0:
        raise ValueError("Negative slice.step not supported here. Use positive step.")

    if stop <= start:
        raise ValueError(f"Empty frame_slice after clipping: start={start}, stop={stop}, n_frames={n_frames}")

    return start, stop, stride

def compare_ttc_methods_from_raw(
    run: RunData,
    *,
    mask_n: int,
    frame_slice: slice | None = None,
    clip_hi_percentile: float = 99.9,
):
    """
    Build I(t,p) from raw frames for ROI mask_n, compute Corr and G TTCs, and plot.

    Uses your existing functions:
      - extract_roi_intensity_matrix(...)
      - ttc_corr_and_g_from_I(...)
      - plot_corr_vs_g_ttc(...)
    """
    n_frames = int(run.dset_raw.shape[0])
    start, stop, stride = _slice_to_start_stop_stride(frame_slice, n_frames)

    I, frame_idxs = extract_roi_intensity_matrix(
        run.dset_raw,
        dynamic_roi_map=run.dynamic_roi_map,
        mask_n=int(mask_n),
        start=start,
        stop=stop,
        stride=stride,
    )

    Corr, G = ttc_corr_and_g_from_I(I)

    plot_corr_vs_g_ttc(
        Corr,
        G,
        clip_hi_percentile=clip_hi_percentile,
    )

    return {
        "I": I,
        "frame_idxs": frame_idxs,
        "Corr": Corr,
        "G": G,
    }

def compare_existing_processed_ttc_with_corr_from_raw(
    run: RunData,
    *,
    mask_n: int,
    frame_slice: slice | None = None,
    clip_hi_percentile: float = 99.9,
    diff_symmetric: bool = True,
    cmap: str = "plasma",
    figsize=(14.2, 4.8),
):
    """
    Side-by-side comparison:

      [0] Existing processed TTC (run.ttc)
      [1] Corr TTC computed directly from raw frames for the same ROI
      [2] Difference map: Corr - Existing

    Notes
    -----
    - We compute Corr on a selected set of frames (frame_slice).
    - We compare to the corresponding submatrix of the processed TTC using those same frame indices.
    - By default, we symmetrize the maps for plotting (your preference).
    """
    # ---- compute Corr from raw for requested frames ----
    n_frames = int(run.dset_raw.shape[0])
    start, stop, stride = _slice_to_start_stop_stride(frame_slice, n_frames)

    I, frame_idxs = extract_roi_intensity_matrix(
        run.dset_raw,
        dynamic_roi_map=run.dynamic_roi_map,
        mask_n=int(mask_n),
        start=start,
        stop=stop,
        stride=stride,
    )

    Corr, _G = ttc_corr_and_g_from_I(I)  # we only need Corr here

    # ---- pull matching submatrix from existing processed TTC ----
    C_exist = np.asarray(run.ttc, dtype=np.float64)
    if C_exist.ndim != 2 or C_exist.shape[0] != C_exist.shape[1]:
        raise ValueError(f"run.ttc must be square, got {C_exist.shape}")

    # frame_idxs refer to raw frames. We assume processed TTC uses the same frame indexing.
    # So we take the corresponding rows/cols.
    if np.max(frame_idxs) >= C_exist.shape[0]:
        raise ValueError(
            f"Processed TTC size {C_exist.shape[0]} is smaller than max frame index {np.max(frame_idxs)}. "
            f"Check whether processed TTC was computed on fewer frames than raw."
        )

    C_exist_sub = C_exist[np.ix_(frame_idxs, frame_idxs)]

    # ---- symmetrize (optional) ----
    if diff_symmetric:
        C_exist_sub = symmetrize_ttc(C_exist_sub)
        Corr = symmetrize_ttc(Corr)

    D = Corr - C_exist_sub
    if diff_symmetric:
        D = symmetrize_ttc(D)

    # ---- clip for display ----
    C0 = clip_ttc(C_exist_sub, p_hi=clip_hi_percentile)
    C1 = clip_ttc(Corr, p_hi=clip_hi_percentile)
    Dp = clip_ttc(D, p_hi=clip_hi_percentile)

    # ---- plot ----
    fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"wspace": 0.28})

    im0 = axs[0].imshow(C0, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    axs[0].set_title("Existing processed TTC (submatrix)")
    axs[0].set_xlabel("t₁ index")
    axs[0].set_ylabel("t₂ index")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(C1, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    axs[1].set_title("Corr TTC from raw")
    axs[1].set_xlabel("t₁ index")
    axs[1].set_ylabel("t₂ index")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(Dp, origin="lower", cmap="magma", interpolation="nearest", aspect="equal")
    axs[2].set_title("Difference: Corr - Existing")
    axs[2].set_xlabel("t₁ index")
    axs[2].set_ylabel("t₂ index")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    for ax in axs:
        ax.grid(False)

    plt.tight_layout()
    plt.show()

    return {
        "frame_idxs": frame_idxs,
        "I": I,
        "Corr": Corr,
        "Existing_sub": C_exist_sub,
        "Diff": D,
    }


# ============================================================
# Execution functions
# ============================================================

def data_structure_viewer():

    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    print("Loaded:")
    print("  raw:", run.raw_path)
    print("    entry/data/data shape:", run.dset_raw.shape, "dtype:", run.dset_raw.dtype)
    print("  meta:", run.meta_path)
    print("  results:", run.results_path)
    print("  dynamic_roi_map:", run.dynamic_roi_map.shape, run.dynamic_roi_map.dtype)
    print("  scattering_2d:", run.scattering_2d.shape, run.scattering_2d.dtype)
    print("  ttc:", run.ttc.shape, run.ttc.dtype)
    print("  g2:", run.g2.shape, run.g2.dtype)

    run.close()

def mask_roi_viewer():

    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    launch_masked_raw_viewer(run, mask_n=MASK_N, start_frame=0, clip_percentile_init=99.9)

    run.close()

def raw_mask_oscillation_inspector():

    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    inspect_raw_mask_oscillations(
        run,
        mask_signal=MASK_N,
        mask_control=CONTROL_MASK_N,
        dt_s=1.0,
        fmin=1 / 1000,
        fmax=1 / 10,
        detrend=True,
        window=True,
    )

    run.close()


def comparison_of_corr_and_g_ttc_plot_methods():
    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    out = compare_ttc_methods_from_raw(
        run,
        mask_n=MASK_N,
        frame_slice=slice(0, 4800),  # or None for all frames
        clip_hi_percentile=99.9,
    )

    run.close()

def compare_existing_vs_corr_entrypoint():

    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)
    try:
        return compare_existing_processed_ttc_with_corr_from_raw(
            run,
            mask_n=MASK_N,
            frame_slice=slice(0, 4800),  # or None
            clip_hi_percentile=99.9,
            diff_symmetric=True,
        )
    finally:
        run.close()


# ============================================================
# User parameters / entry point
# ============================================================

BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006")
SAMPLE_ID = "A073"
MASK_N = 145
CONTROL_MASK_N = 3

if __name__ == "__main__":

    # data_structure_viewer()
    mask_roi_viewer()
    # raw_mask_oscillation_inspector()
    # comparison_of_corr_and_g_ttc_plot_methods()
    # compare_existing_vs_corr_entrypoint()

    pass
