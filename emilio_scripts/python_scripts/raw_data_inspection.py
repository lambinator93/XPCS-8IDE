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


# Example usage (inside your __main__ after you have `run = load_run_data(...)`):
# launch_masked_raw_viewer(run, mask_n=MASK_N, start_frame=0, clip_percentile_init=99.9)


# ============================================================
# User parameters / entry point
# ============================================================

BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006")
SAMPLE_ID = "A073"
MASK_N = 145

if __name__ == "__main__":
    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    launch_masked_raw_viewer(run, mask_n=MASK_N, start_frame=0, clip_percentile_init=99.9)

    print("Loaded:")
    print("  raw:", run.raw_path)
    print("    entry/data/data shape:", run.dset_raw.shape, "dtype:", run.dset_raw.dtype)
    print("  meta:", run.meta_path)
    print("  results:", run.results_path)
    print("  dynamic_roi_map:", run.dynamic_roi_map.shape, run.dynamic_roi_map.dtype)
    print("  scattering_2d:", run.scattering_2d.shape, run.scattering_2d.dtype)
    print("  ttc:", run.ttc.shape, run.ttc.dtype)
    print("  g2:", run.g2.shape, run.g2.dtype)

    # Keep run.f_raw / run.f_meta open for later steps.
    # When you're done:
    # run.close()
