from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import h5py
import numpy as np
from matplotlib.patches import FancyArrowPatch

import matplotlib as mpl
mpl.use("macosx")  # must be set before importing pyplot
import matplotlib.pyplot as plt


# ============================================================
# File finding + loading
# ============================================================

def find_results_hdf(base_dir: Path, sample_id: str) -> Path:
    """
    Find the first file matching: <sample_id>_*_results.hdf
    Raises FileNotFoundError if none exist.
    """
    matches = sorted(base_dir.glob(f"{sample_id}_*_results.hdf"))
    if not matches:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")
    return matches[0]


@dataclass
class XPCSData:
    dynamic_roi_map: np.ndarray
    scattering_2d: np.ndarray
    ttc: np.ndarray
    g2: np.ndarray


def load_xpcs_arrays(
    sample_id: str,
    base_dir: Path,
    *,
    mask_n: int,
    scattering_first_frame_only: bool = True,
) -> XPCSData:
    """
    Load the 4 arrays you listed as NumPy arrays:

        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d   = f["xpcs/temporal_mean/scattering_2d"][...]
        ttc             = f["xpcs/twotime/correlation_map/c2_00{mask_n:03d}"][...]
        g2              = f["xpcs/twotime/normalized_g2"][...]

    Returns XPCSData(dataclass).
    """
    hdf_path = find_results_hdf(base_dir, sample_id)
    ttc_path = f"xpcs/twotime/correlation_map/c2_00{mask_n:03d}"

    with h5py.File(hdf_path, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]

        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        if scattering_first_frame_only and scattering_2d.ndim == 3:
            scattering_2d = scattering_2d[0, :, :]

        ttc = f[ttc_path][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]

    return XPCSData(
        dynamic_roi_map=dynamic_roi_map,
        scattering_2d=scattering_2d,
        ttc=ttc,
        g2=g2,
    )


def save_xpcs_npz(out_path: Path, data: XPCSData) -> Path:
    """Optional cache to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        dynamic_roi_map=data.dynamic_roi_map,
        scattering_2d=data.scattering_2d,
        ttc=data.ttc,
        g2=data.g2,
    )
    return out_path


# ============================================================
# TTC preprocessing + utilities
# ============================================================

def symmetrize_ttc(ttc: np.ndarray) -> np.ndarray:
    """Mirror TTC along x=y: C -> C + C.T - diag(diag(C))."""
    C0 = np.asarray(ttc, dtype=np.float64)
    return C0 + C0.T - np.diag(np.diag(C0))


def clip_ttc(C: np.ndarray, p_hi: float = 99.9) -> np.ndarray:
    """Clip extreme outliers so one pixel doesn't dominate plots/fits."""
    lo, hi = np.percentile(C, [0.0, p_hi])
    return np.clip(C, lo, hi)


def despike_patch_with_local_median(C: np.ndarray, center: Tuple[int, int], halfwidth: int = 1) -> np.ndarray:
    """
    Replace a small (2*halfwidth+1)^2 patch centered at `center` with the median
    of a slightly larger surrounding window (excluding the patch).
    """
    C = C.copy()
    n = C.shape[0]
    cy, cx = center

    y0 = max(cy - halfwidth, 0)
    y1 = min(cy + halfwidth + 1, n)
    x0 = max(cx - halfwidth, 0)
    x1 = min(cx + halfwidth + 1, n)

    pad = max(halfwidth + 2, 3)
    Y0 = max(cy - pad, 0)
    Y1 = min(cy + pad + 1, n)
    X0 = max(cx - pad, 0)
    X1 = min(cx + pad + 1, n)

    window = C[Y0:Y1, X0:X1].copy()

    py0, py1 = y0 - Y0, y1 - Y0
    px0, px1 = x0 - X0, x1 - X0
    window[py0:py1, px0:px1] = np.nan

    med = np.nanmedian(window)
    if np.isfinite(med):
        C[y0:y1, x0:x1] = med

    return C


def arrow_endpoint_to_edge(start_x: int, start_y: int, dx: int, dy: int, n: int) -> Tuple[int, int]:
    """
    For a direction (dx,dy) in {-1,0,1}, return the endpoint on the image edge
    when stepping from (start_x,start_y) until leaving bounds.
    """
    if dx == 0 and dy == 0:
        return start_x, start_y

    steps = []
    if dx > 0:
        steps.append(n - 1 - start_x)
    elif dx < 0:
        steps.append(start_x)

    if dy > 0:
        steps.append(n - 1 - start_y)
    elif dy < 0:
        steps.append(start_y)

    kmax = min(steps) if steps else 0
    return start_x + dx * kmax, start_y + dy * kmax


# ============================================================
# Lineouts (TTC + annotated figure)
# ============================================================

def plot_ttc_with_lineouts(
    data: XPCSData,
    start: int,
    *,
    clip_percentile: Optional[float] = 99.9,
    cmap: str = "plasma",
    add_antidiag_se: bool = True,
    despike_at_start: bool = True,
    despike_halfwidth: int = 1,
) -> None:
    """
    Left : symmetrized TTC with arrows for lineouts (to edges)
    Right: lineout curves

    - TTC is mirrored about x=y: C -> C + C.T - diag(diag(C))
    - Vertical lineout goes DOWN in time (toward x-axis) => decreasing t2 index
    - Adds anti-diagonal lineout from (start,start) toward SE: (x+, y-)
    """
    C = symmetrize_ttc(data.ttc)
    n = C.shape[0]
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"TTC must be square, got shape {C.shape}")
    if not (0 <= start < n):
        raise ValueError(f"start must be in [0, {n-1}]")

    if despike_at_start:
        C = despike_patch_with_local_median(C, center=(start, start), halfwidth=despike_halfwidth)

    # Lineouts (anchored at (start,start))
    # Horizontal right: (x+, y)
    x_h = np.arange(start, n)
    y_h = C[start, start:n]

    # Vertical DOWN: (x, y-) -> decreasing row index
    x_v = np.arange(start, -1, -1)
    y_v = C[start::-1, start]

    # Main diagonal forward: (x+, y+)
    x_d = np.arange(start, n)
    y_d = np.diag(C)[start:n]

    lineouts = [
        dict(x=x_h, y=y_h, color="tab:blue",   label="horizontal (→)"),
        dict(x=x_v, y=y_v, color="tab:orange", label="vertical (↓)"),
        dict(x=x_d, y=y_d, color="tab:green",  label="diag x=y (↗)"),
    ]

    # Anti-diagonal SE (x+, y-) from (start,start): (start+k, start-k)
    if add_antidiag_se:
        kmax = min(n - 1 - start, start)
        xs = start + np.arange(0, kmax + 1)
        ys = start - np.arange(0, kmax + 1)
        y_ad = C[ys, xs]  # C[row=y, col=x]
        lineouts.append(dict(x=xs, y=y_ad, color="tab:purple", label="anti-diag (↘)"))

    Cplot = C.copy()
    if clip_percentile is not None:
        Cplot = clip_ttc(Cplot, p_hi=float(clip_percentile))

    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)

    # Left: TTC + arrows
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title("TTC with lineouts")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    arrows = [
        ("horizontal", (1, 0),  "tab:blue"),
        ("vertical",   (0, -1), "tab:orange"),  # down
        ("diag x=y",   (1, 1),  "tab:green"),
    ]
    if add_antidiag_se:
        arrows.append(("anti-diag", (1, -1), "tab:purple"))  # SE

    for _, (dx, dy), col in arrows:
        ex, ey = arrow_endpoint_to_edge(start, start, dx, dy, n)
        ax0.annotate(
            "",
            xy=(ex, ey),
            xytext=(start, start),
            arrowprops=dict(arrowstyle="->", lw=3, color=col),
        )

    ax0.plot([start], [start], marker="o", markersize=4, color="white", zorder=5)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # Right: lineout curves
    ax1 = fig.add_subplot(gs[1])
    for d in lineouts:
        ax1.plot(d["x"], d["y"], lw=2, color=d["color"], label=d["label"])
    ax1.set_title(f"Lineouts from t₁=t₂={start}")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("TTC value")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Fitting: Option 3 (baseline term) using grid over (omega, tau)
# ============================================================

def fit_damped_cosine_with_linear_baseline(
    t: np.ndarray,
    y: np.ndarray,
    omega_grid: np.ndarray,
    tau_grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit:
      y(t)= C + b*t + exp(-t/tau)*(a*cos(omega*t) + s*sin(omega*t))

    For any fixed (omega, tau), (C, b, a, s) are solved by weighted least squares.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if weights is None:
        w = np.ones_like(t)
    else:
        w = np.asarray(weights, float)
        w = np.clip(w, 0.0, np.inf)

    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(w)
    t, y, w = t[m], y[m], w[m]

    sw = np.sqrt(w)

    best: Dict[str, Any] = {"sse": np.inf}

    for tau in tau_grid:
        tau = float(tau)
        if tau <= 0:
            continue
        e = np.exp(-t / tau)

        for omega in omega_grid:
            omega = float(omega)
            coswt = np.cos(omega * t)
            sinwt = np.sin(omega * t)

            # [1, t, e*cos, e*sin] * [C, b, a, s]^T
            A = np.column_stack([np.ones_like(t), t, e * coswt, e * sinwt])

            Aw = A * sw[:, None]
            yw = y * sw

            coeff, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
            C, b, a, s = coeff

            yhat = A @ coeff
            resid = y - yhat
            sse = float(np.sum(w * resid * resid))

            if sse < best["sse"]:
                R = float(np.hypot(a, s))
                # a*cos + s*sin = R*cos(ωt + φ) with φ = atan2(-s, a)
                phi = float(np.arctan2(-s, a))

                best = dict(
                    C=float(C),
                    b=float(b),
                    a=float(a),
                    s=float(s),
                    R=R,
                    phi=phi,
                    omega=omega,
                    tau=tau,
                    yhat=yhat,
                    sse=sse,
                )

    return best


def evaluate_model(t: np.ndarray, C: float, b: float, omega: float, tau: float, a: float, s: float) -> np.ndarray:
    t = np.asarray(t, float)
    e = np.exp(-t / tau)
    return C + b * t + e * (a * np.cos(omega * t) + s * np.sin(omega * t))


# ============================================================
# Extract + fit + plot (anti-diagonal through diagonal anchor)
# ============================================================

def extract_fit_antidiagonal_with_ttc_plot(
    sample_id: str,
    base_dir: Path,
    *,
    mask_n: int,
    start_time_idx: int,
    dt_s: float = 1.0,
    omega_min: float = 2 * np.pi / 500.0,
    omega_max: float = 2 * np.pi / 20.0,
    n_omega: int = 260,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    n_tau: int = 120,
    clip_hi_percentile: float = 99.9,
    despike_at_start: bool = True,
    despike_halfwidth: int = 1,
    use_weights: bool = True,
) -> Dict[str, Any]:
    """
    Extract the anti-diagonal lineout passing through (i,i) where i=start_time_idx:
      (t1, t2) = (i-k, i+k), k>=0, staying in-bounds

    Then fit Option 3:
      y(t)= C + b*t + exp(-t/tau)*(a*cos(omega*t) + s*sin(omega*t))

    Produces a 1x2 figure:
      Left: TTC with arrow along the extracted anti-diagonal
      Right: lineout + best-fit curve
    """
    data = load_xpcs_arrays(sample_id, base_dir, mask_n=mask_n)

    C = symmetrize_ttc(data.ttc)
    if despike_at_start:
        C = despike_patch_with_local_median(C, center=(start_time_idx, start_time_idx), halfwidth=despike_halfwidth)
    C = clip_ttc(C, p_hi=clip_hi_percentile)

    N = C.shape[0]
    i = int(start_time_idx)
    if not (0 <= i < N):
        raise ValueError(f"start_time_idx must be in [0, {N-1}]")

    # Anti-diagonal through (i,i): (i-k, i+k)
    kmax = min(i, N - 1 - i)
    ks = np.arange(0, kmax + 1, dtype=int)
    t1 = i - ks
    t2 = i + ks
    y = C[t1, t2]

    tau_idx = 2 * ks
    t = tau_idx.astype(np.float64) * float(dt_s)

    # Grids for omega/tau
    omega_grid = np.linspace(float(omega_min), float(omega_max), int(n_omega))

    if tau_min is None:
        tau_min = max(2.0 * dt_s, 0.02 * (t.max() if t.size else 1.0))
    if tau_max is None:
        tau_max = max(5.0 * tau_min, 2.0 * (t.max() if t.size else 1.0))
    tau_grid = np.geomspace(float(tau_min), float(tau_max), int(n_tau))

    # ----------------------------
    # FFT analysis (NEW – put it HERE)
    # ----------------------------
    freqs, power, f_peak, period_s, p_peak = fft_peak_from_lineout(
        y,
        dt_s,
        detrend=True,
        window="hann",
        fmin=1 / 1000,  # periods ≤ 1000 s
        fmax=1 / 10,  # periods ≥ 10 s
    )

    print(
        f"FFT peak: f = {f_peak:.4g} Hz  "
        f"(period ≈ {period_s:.2f} s)"
    )

    # Optional: weight early times more (often helps)
    weights = None
    if use_weights and t.size:
        weights = np.exp(-t / (0.7 * t.max()))

    fit = fit_damped_cosine_with_linear_baseline(t, y, omega_grid, tau_grid, weights=weights)

    # Summary
    period = (2 * np.pi / fit["omega"]) if fit["omega"] != 0 else np.inf
    print("Fit parameters (baseline + damped cosine):")
    print(f"  C      = {fit['C']:.6g}")
    print(f"  b      = {fit['b']:.6g}")
    print(f"  tau    = {fit['tau']:.6g} s")
    print(f"  omega  = {fit['omega']:.6g} rad/s  (period ≈ {period:.2f} s)")
    print(f"  R      = {fit['R']:.6g}")
    print(f"  phi    = {fit['phi']:.6g} rad")
    print(f"  SSE    = {fit['sse']:.6g}")

    # Plot
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.35}
    )

    # Left: TTC + arrow
    im = ax0.imshow(C, origin="lower", cmap="plasma", interpolation="nearest")
    ax0.set_title(f"{sample_id}  M{mask_n}  TTC")
    ax0.set_xlabel("t₂ index")
    ax0.set_ylabel("t₁ index")

    end_t1 = i - kmax
    end_t2 = i + kmax
    ax0.annotate(
        "",
        xy=(end_t2, end_t1),
        xytext=(i, i),
        arrowprops=dict(arrowstyle="->", lw=2.8, color="tab:purple"),
    )
    ax0.plot(i, i, "o", color="white", ms=6)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # Right: lineout + fit
    ax1.plot(t, y, lw=2, color="tab:purple", label="anti-diagonal lineout")
    ax1.plot(t, fit["yhat"], lw=2, color="black", ls="--", label="fit")
    ax1.set_title("Anti-diagonal lineout + fit")
    ax1.set_xlabel("τ (s)")
    ax1.set_ylabel("TTC value")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.show()

    return fit

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def symmetrize_ttc(ttc: np.ndarray) -> np.ndarray:
    """Mirror TTC along x=y."""
    ttc = np.asarray(ttc, dtype=np.float64)
    return ttc + ttc.T - np.diag(np.diag(ttc))


def clip_ttc(C: np.ndarray, p_hi: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(C, [0.0, p_hi])
    return np.clip(C, lo, hi)

def extract_antidiagonal_lineout(
    ttc: np.ndarray,
    *,
    start_idx: int,
    dt_s: float,
    clip_percentile: float | None = 99.9,
):
    C = symmetrize_ttc(ttc)

    if clip_percentile is not None:
        lo, hi = np.percentile(C, [0, clip_percentile])
        C = np.clip(C, lo, hi)

    N = C.shape[0]
    i = int(start_idx)
    if not (0 <= i < N):
        raise ValueError("start_idx outside TTC bounds")

    kmax = min(i, N - 1 - i)
    ks = np.arange(0, kmax + 1)

    t1 = i - ks
    t2 = i + ks
    y = C[t1, t2]

    tau_idx = 2 * ks
    t = tau_idx * dt_s

    return t, y


def estimate_fft(t: np.ndarray, y: np.ndarray, *, drop_first: int = 0, detrend: bool = True, window: bool = True):
    """
    Returns freqs (Hz), power, f_peak (Hz), period (s)
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if drop_first > 0:
        t = t[drop_first:]
        y = y[drop_first:]

    if len(t) < 4:
        raise ValueError("Need at least 4 points for FFT.")

    dt = float(np.median(np.diff(t)))
    yy = y.copy()

    if detrend:
        A = np.column_stack([np.ones_like(t), t])
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
        yy = yy - (A @ beta)

    if window:
        yy = yy * np.hanning(len(yy))

    Y = np.fft.rfft(yy)
    freqs = np.fft.rfftfreq(len(yy), d=dt)
    power = (Y.real**2 + Y.imag**2)

    if len(power) > 0:
        power[0] = 0.0  # remove DC

    k = int(np.argmax(power)) if len(power) else 0
    f_peak = float(freqs[k]) if len(freqs) else 0.0
    period = (1.0 / f_peak) if f_peak > 0 else np.inf
    return freqs, power, f_peak, period


# ------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------

def plot_ttc_lineout_fft(
    data,
    *,
    start_idx: int,
    dt_s: float = 1.0,
    clip_hi_percentile: float = 99.9,
    cmap: str = "plasma",
    arrow_color: str = "C2",
    drop_first: int = 0,
    detrend: bool = True,
    window: bool = True,
    figsize=(13, 5.5),
):
    """
    Figure layout:
      - Left (spans full height): TTC plot with anti-diagonal arrow (C2)
      - Right-top: lineout vs tau
      - Right-bottom: FFT power spectrum

    Uses data = load_xpcs_arrays(sample_id, BASE_DIR, mask_n=mask_n)
    where data.ttc is the TTC array.
    """
    C = symmetrize_ttc(data.ttc)
    Cplot = clip_ttc(C, p_hi=clip_hi_percentile)

    t, y = extract_antidiagonal_lineout(
        C,
        start_idx=start_idx,
        dt_s=dt_s,
        clip_percentile=clip_hi_percentile,
    )
    # keep kmax separately for the arrow length:
    N = C.shape[0]
    i = int(start_idx)
    kmax = min(i, N - 1 - i)

    freqs, power, f_peak, period = estimate_fft(
        t, y, drop_first=drop_first, detrend=detrend, window=window
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], wspace=0.35, hspace=0.35)

    # -------------------------
    # LEFT: TTC + arrow
    # -------------------------
    ax0 = fig.add_subplot(gs[:, 0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title(f"TTC, start t1=t2={start_idx}")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    # anti-diagonal through (i,i): endpoints (i-kmax, i+kmax)
    i = int(start_idx)
    end_t1 = i - kmax
    end_t2 = i + kmax

    ax0.add_patch(
        FancyArrowPatch(
            (i, i),
            (end_t2, end_t1),
            arrowstyle="->",
            linewidth=3,
            mutation_scale=14,
            color=arrow_color,
        )
    )
    ax0.plot([i], [i], marker="o", markersize=5, color="white", zorder=5)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # -------------------------
    # RIGHT-TOP: lineout
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, y, lw=2, color=arrow_color, label="anti-diagonal lineout")
    ax1.set_title("Anti-diagonal lineout")
    ax1.set_xlabel("τ (s)")
    ax1.set_ylabel("TTC value")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # -------------------------
    # RIGHT-BOTTOM: FFT
    # -------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(freqs, power, lw=1.8, color="black")
    ax2.set_title(f"FFT power (peak period ≈ {period:.2f} s)" if np.isfinite(period) else "FFT power")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_yscale("log")
    ax2.set_xlim([-0.001, 0.01])
    ax2.grid(True, alpha=0.3)

    # mark peak
    if f_peak > 0:
        ax2.axvline(f_peak, lw=1.5, color=arrow_color, alpha=0.9)

    plt.show()
    return {
        "t": t,
        "y": y,
        "freqs": freqs,
        "power": power,
        "f_peak_hz": f_peak,
        "period_s": period,
    }

def fft_peak_from_lineout(
    y: np.ndarray,
    dt_s: float,
    *,
    detrend: bool = True,
    window: str = "hann",
    fmin: float | None = None,
    fmax: float | None = None,
):
    """
    Compute FFT power spectrum for a 1D signal and return the dominant peak.

    No zero-padding (n_fft = N), so no artificial interpolation.
    Uses mean removal + optional linear detrend + optional windowing.
    """
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    if N < 8:
        raise ValueError("Need at least ~8 points for a meaningful FFT")

    # 1) remove DC
    x = y - np.mean(y)

    # 2) optional linear detrend
    if detrend:
        t = np.arange(N, dtype=np.float64)
        # Fit x ~ a*t + b, subtract
        a, b = np.polyfit(t, x, 1)
        x = x - (a * t + b)

    # 3) window (reduces leakage => typically makes the main peak taller/cleaner)
    if window is None or window.lower() == "none":
        w = np.ones(N, dtype=np.float64)
    elif window.lower() in ("hann", "hanning"):
        w = np.hanning(N)
    elif window.lower() == "hamming":
        w = np.hamming(N)
    elif window.lower() == "blackman":
        w = np.blackman(N)
    else:
        raise ValueError(f"Unknown window: {window}")

    xw = x * w

    # 4) rFFT (real FFT)
    F = np.fft.rfft(xw)  # length N//2+1
    freqs = np.fft.rfftfreq(N, d=dt_s)

    # Power spectrum (magnitude^2). Normalize by window power so comparisons are saner.
    # This does NOT change the peak frequency, just makes amplitudes comparable.
    W = np.sum(w**2)
    power = (np.abs(F) ** 2) / W

    # 5) Peak pick (ignore DC bin at 0 Hz)
    mask = np.ones_like(freqs, dtype=bool)
    mask[0] = False

    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    if not np.any(mask):
        raise ValueError("No frequencies left after applying fmin/fmax")

    k_peak = np.argmax(power[mask])
    idxs = np.flatnonzero(mask)
    k = idxs[k_peak]

    f_peak = freqs[k]
    p_peak = power[k]
    period_s = (1.0 / f_peak) if f_peak > 0 else np.inf

    return freqs, power, f_peak, period_s, p_peak

def detrend_linear(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    A = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - (A @ beta)

def window_fn(name: str, n: int) -> np.ndarray:
    name = (name or "").lower()
    if name in ("hann", "hanning"):
        return np.hanning(n)
    if name in ("hamming",):
        return np.hamming(n)
    return np.ones(n)

def segment_indices(n: int, seg_len: int, overlap: float) -> list[tuple[int,int]]:
    step = max(1, int(round(seg_len * (1 - overlap))))
    idx = []
    for s in range(0, n - seg_len + 1, step):
        idx.append((s, s + seg_len))
    return idx

def periodogram(y: np.ndarray, dt: float, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    n = len(y)
    w = window_fn(window, n)
    yw = (y - np.mean(y)) * w
    # rfft
    Y = np.fft.rfft(yw)
    # power (not absolute scaling-critical for peak picking)
    P = (np.abs(Y) ** 2)
    f = np.fft.rfftfreq(n, d=dt)
    return f, P

def peak_frequency_from_psd(f: np.ndarray, P: np.ndarray, fmin: float, fmax: float) -> float:
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        raise ValueError("No frequencies in band")
    i = np.argmax(P[m])
    return f[m][i]

def bootstrap_peak_frequency(
    y: np.ndarray,
    dt: float,
    *,
    fmin: float,
    fmax: float,
    seg_len: int | None = None,
    overlap: float = 0.5,
    window: str = "hann",
    detrend: bool = True,
    n_boot: int = 2000,
    ci: float = 0.68,
    rng_seed: int = 0,
):
    """
    Returns:
      f_hat (median), f_lo, f_hi, f_samples
    """
    y = np.asarray(y, float)
    if detrend:
        y = detrend_linear(y)

    n = len(y)
    if seg_len is None:
        # sensible default: ~1/4 of record (at least 64 points)
        seg_len = max(64, n // 4)
    seg_len = min(seg_len, n)

    idx = segment_indices(n, seg_len, overlap)
    if len(idx) < 3:
        # fallback: use whole record as one segment (uncertainty will be meaningless)
        f, P = periodogram(y, dt, window=window)
        f0 = peak_frequency_from_psd(f, P, fmin, fmax)
        return f0, f0, f0, np.array([f0])

    # Compute PSD per segment
    f_ref = None
    Ps = []
    for (a, b) in idx:
        f, P = periodogram(y[a:b], dt, window=window)
        if f_ref is None:
            f_ref = f
        else:
            # same length => same freq grid
            pass
        Ps.append(P)
    Ps = np.stack(Ps, axis=0)  # (K, nf)
    f_ref = np.asarray(f_ref)

    # Bootstrap resample segments
    rng = np.random.default_rng(rng_seed)
    K = Ps.shape[0]
    f_samp = np.empty(n_boot, float)

    for i in range(n_boot):
        picks = rng.integers(0, K, size=K)  # resample K segments with replacement
        Pmean = Ps[picks].mean(axis=0)
        f_samp[i] = peak_frequency_from_psd(f_ref, Pmean, fmin, fmax)

    f_hat = float(np.median(f_samp))
    alpha = (1 - ci) / 2
    f_lo = float(np.quantile(f_samp, alpha))
    f_hi = float(np.quantile(f_samp, 1 - alpha))
    return f_hat, f_lo, f_hi, f_samp

# def fft_peak_with_bin_uncertainty(y: np.ndarray, dt_s: float, *, drop_first: int = 0):
#     """
#     Peak frequency from plain FFT + uncertainty from bin width (no segmentation).
#     Returns f_peak, sigma_f, period, sigma_period, delta_f.
#     """
#     y = np.asarray(y, dtype=np.float64)
#     if drop_first:
#         y = y[drop_first:]
#
#     N = y.size
#     if N < 8:
#         raise ValueError("Need at least ~8 points for FFT")
#
#     delta_f = 1.0 / (N * dt_s)          # FFT bin spacing
#     sigma_f = 0.5 * delta_f             # ~half-bin uncertainty
#
#     # Use your existing peak picker so it's consistent
#     freqs, power, f_peak, period_s, _ = fft_peak_from_lineout(
#         y, dt_s,
#         detrend=True,
#         window="hann",
#         fmin=1/1000,
#         fmax=1/10,
#     )
#
#     sigma_T = (sigma_f / (f_peak**2)) if f_peak > 0 else np.inf
#     return f_peak, sigma_f, period_s, sigma_T, delta_f

def fft_peak_with_bin_uncertainty(
    y: np.ndarray,
    dt_fft: float,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
    detrend: bool = True,
    window: str = "hann",
):
    """
    Peak frequency from FFT, plus bin-width uncertainty.

    Uncertainty model:
      df = 1 / (N * dt_fft)
      f ~ f_peak ± df/2

    Returns
    -------
    f_peak, f_lo, f_hi, period, period_lo, period_hi, df
    """
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    if N < 8:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    freqs, power, f_peak, period_s, p_peak = fft_peak_from_lineout(
        y, dt_fft,
        detrend=detrend,
        window=window,
        fmin=fmin,
        fmax=fmax,
    )

    df = 1.0 / (N * dt_fft)  # FFT frequency bin spacing
    f_lo = f_peak - 0.5 * df
    f_hi = f_peak + 0.5 * df

    # Respect any user band limits (optional)
    if fmin is not None:
        f_lo = max(f_lo, fmin)
        f_hi = max(f_hi, fmin)
    if fmax is not None:
        f_lo = min(f_lo, fmax)
        f_hi = min(f_hi, fmax)

    # Convert to period band. (Note inversion flips bounds.)
    # Use conservative mapping: period_hi corresponds to the lower frequency bound.
    if f_lo <= 0:
        period_hi = np.nan
    else:
        period_hi = 1.0 / f_lo

    if f_hi <= 0:
        period_lo = np.nan
    else:
        period_lo = 1.0 / f_hi

    period = 1.0 / f_peak if f_peak > 0 else np.nan

    return f_peak, f_lo, f_hi, period, period_lo, period_hi, df


def extract_antidiagonal_lineout_y_only(Csym: np.ndarray, start_idx: int, *, drop_first: int = 0):
    """
    Extract y along anti-diagonal through (i,i): (t1=i-k, t2=i+k).
    Returns y (optionally dropping first points).
    """
    Csym = np.asarray(Csym, dtype=np.float64)
    n = Csym.shape[0]
    i = int(start_idx)
    if Csym.ndim != 2 or Csym.shape[0] != Csym.shape[1]:
        raise ValueError(f"TTC must be square, got {Csym.shape}")
    if not (0 <= i < n):
        raise ValueError(f"start_idx must be in [0, {n-1}]")

    kmax = min(i, n - 1 - i)
    ks = np.arange(0, kmax + 1, dtype=int)
    t1 = i - ks
    t2 = i + ks
    y = Csym[t1, t2]

    if drop_first > 0:
        y = y[drop_first:]

    return y


def plot_period_vs_diagonal_start(
    data,
    *,
    dt_s: float,
    start_idxs: np.ndarray | None = None,
    # preprocessing
    clip_hi_percentile: float = 99.9,
    drop_first_lineout: int = 0,
    # smoothing
    half_window: int = 5,            # <-- NEW: n lineouts either side
    band_ci: float = 0.68,           # <-- NEW: central CI for shaded band (0.68 ~ 1σ)
    # FFT options
    fmin: float | None = 1/1000,
    fmax: float | None = 1/10,
    detrend: bool = True,
    window: str = "hann",
    # plotting
    cmap: str = "plasma",
    figsize=(13, 5.5),
):
    """
    One figure:
      Left: TTC with diagonal start positions marked in C2
      Right: smoothed peak period vs start time with shaded uncertainty band

    Notes
    -----
    Anti-diagonal lag is tau = 2*k*dt_s, so FFT sampling interval is:
      dt_fft = 2*dt_s
    """
    # --- TTC prep (symmetrize + clip for display/lineouts) ---
    C = symmetrize_ttc(data.ttc)
    Cplot = clip_ttc(C, p_hi=clip_hi_percentile)

    n = C.shape[0]
    if start_idxs is None:
        lo = int(0.05 * (n - 1))
        hi = int(0.95 * (n - 1))
        start_idxs = np.linspace(lo, hi, 80).astype(int)
        start_idxs = np.unique(start_idxs)
    else:
        start_idxs = np.unique(np.asarray(start_idxs, dtype=int))

    # FFT sampling interval for anti-diagonal lineout
    dt_fft = 2.0 * float(dt_s)

    # ------------------------------------------------------------
    # Step 1: compute a RAW period estimate at each start index
    # ------------------------------------------------------------
    raw_period = np.full(start_idxs.shape, np.nan, dtype=float)

    for k, i in enumerate(start_idxs):
        y = extract_antidiagonal_lineout_y_only(C, int(i), drop_first=drop_first_lineout)

        if y.size < 16:
            continue

        f_peak, f_lo, f_hi, period, period_lo, period_hi, df = fft_peak_with_bin_uncertainty(
            y, dt_fft,
            fmin=fmin, fmax=fmax,
            detrend=detrend,
            window=window,
        )

        if np.isfinite(period) and period > 0:
            raw_period[k] = period

    # ------------------------------------------------------------
    # Step 2: smooth by pooling +/- half_window neighbors
    #         and compute a robust uncertainty band from quantiles
    # ------------------------------------------------------------
    half_window = int(max(0, half_window))
    smooth_period = np.full_like(raw_period, np.nan)
    band_lo = np.full_like(raw_period, np.nan)
    band_hi = np.full_like(raw_period, np.nan)

    # quantiles for central CI
    alpha = (1.0 - float(band_ci)) / 2.0
    q_lo = alpha
    q_hi = 1.0 - alpha

    for k in range(len(start_idxs)):
        a = max(0, k - half_window)
        b = min(len(start_idxs), k + half_window + 1)

        window_vals = raw_period[a:b]
        window_vals = window_vals[np.isfinite(window_vals)]

        if window_vals.size < 3:
            continue

        # center estimate
        smooth_period[k] = float(np.median(window_vals))

        # uncertainty band (robust)
        band_lo[k] = float(np.quantile(window_vals, q_lo))
        band_hi[k] = float(np.quantile(window_vals, q_hi))

    # Convert start idx -> seconds
    starts_s = start_idxs.astype(float) * float(dt_s)

    # Keep only finite points for plotting
    m = np.isfinite(smooth_period)
    starts_s_m = starts_s[m]
    smooth_m = smooth_period[m]
    blo_m = band_lo[m]
    bhi_m = band_hi[m]
    start_idxs_used = start_idxs[m]

    # --- plot ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)

    # Left: TTC with start points
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title("TTC + diagonal start positions")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    ax0.plot(start_idxs_used, start_idxs_used, "o", ms=3.5, color="C2", alpha=0.9)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # Right: smoothed period vs start time with band
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(starts_s_m, smooth_m, lw=2.4, color="C2", label=f"median over ±{half_window}")

    mm = np.isfinite(blo_m) & np.isfinite(bhi_m)
    ax1.fill_between(
        starts_s_m[mm],
        blo_m[mm],
        bhi_m[mm],
        color="C2",
        alpha=0.25,
        linewidth=0,
        label=f"{int(band_ci*100)}% window band",
    )

    ax1.set_title("Peak period vs diagonal start time")
    ax1.set_xlabel("Diagonal start time  (t = start_idx · dt_s)  [s]")
    ax1.set_ylabel("Peak period  [s]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return {
        "start_idx": start_idxs_used,
        "start_time_s": starts_s_m,
        "period_raw_s": raw_period,
        "period_smooth_s": smooth_m,
        "band_lo_s": blo_m,
        "band_hi_s": bhi_m,
        "dt_fft_used_s": dt_fft,
        "half_window": half_window,
        "band_ci": band_ci,
    }

def peak_period_from_start_idx(
    data,
    start_idx,
    dt_s,
    *,
    fmin,
    fmax,
):
    C = symmetrize_ttc(data.ttc)
    tau_idx, y, _ = extract_antidiagonal_lineout(C, start_idx)
    _, _, f_peak, period_s, _ = fft_peak_from_lineout(
        y, dt_s,
        detrend=True,
        window="hann",
        fmin=fmin,
        fmax=fmax,
    )
    return period_s

def smoothed_peak_period_vs_time(
    data,
    start_idxs: np.ndarray,
    *,
    dt_s: float,
    half_window: int = 5,
    fmin: float | None = 1 / 1000,
    fmax: float | None = 1 / 10,
    drop_first: int = 0,
    detrend: bool = True,
    window: str = "hann",
    clip_hi_percentile: float = 99.9,
):
    """
    For each diagonal start index i, average FFT peak periods over
    start indices [i-half_window, ..., i+half_window].

    Returns:
        period, period_lo, period_hi  (arrays, same length as start_idxs)
    """
    C = symmetrize_ttc(data.ttc)
    C = clip_ttc(C, p_hi=clip_hi_percentile)

    start_idxs = np.asarray(start_idxs, dtype=int)
    n = C.shape[0]

    dt_fft = 2.0 * float(dt_s)

    period = np.full(len(start_idxs), np.nan)
    period_lo = np.full(len(start_idxs), np.nan)
    period_hi = np.full(len(start_idxs), np.nan)

    for j, i0 in enumerate(start_idxs):
        # neighborhood of diagonal starts
        neigh = start_idxs[
            (start_idxs >= i0 - half_window) &
            (start_idxs <= i0 + half_window)
        ]

        periods_local = []

        for i in neigh:
            if i < 0 or i >= n:
                continue

            y = extract_antidiagonal_lineout_y_only(
                C, int(i), drop_first=drop_first
            )

            if y.size < 16:
                continue

            try:
                f_peak, _, _, p, _, _, _ = fft_peak_with_bin_uncertainty(
                    y,
                    dt_fft,
                    fmin=fmin,
                    fmax=fmax,
                    detrend=detrend,
                    window=window,
                )
            except Exception:
                continue

            if np.isfinite(p):
                periods_local.append(p)

        if len(periods_local) == 0:
            continue

        periods_local = np.asarray(periods_local)
        period[j] = np.median(periods_local)
        period_lo[j] = np.percentile(periods_local, 16)
        period_hi[j] = np.percentile(periods_local, 84)

    return period, period_lo, period_hi




# ============================================================
# Main
# ============================================================

BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
SAMPLE_ID = "A073"
MASK_N = 145

if __name__ == "__main__":
    # If your TTC time step is known (seconds per TTC index step), set dt_s appropriately.
    DT_S = 1.0

    # "Start point" along t1=t2
    START_DIAG_IDX = 2500  # e.g. "t=t=600s" if dt_s=1s (otherwise it's an index)

    # Example: constrain possible periods
    # period ~100 s => omega ~ 2π/100 ~ 0.0628 rad/s
    # fit = extract_fit_antidiagonal_with_ttc_plot(
    #     SAMPLE_ID,
    #     BASE_DIR,
    #     mask_n=MASK_N,
    #     start_time_idx=START_DIAG_IDX,
    #     dt_s=DT_S,
    #     omega_min=2 * np.pi / 500.0,  # periods up to 500 s
    #     omega_max=2 * np.pi / 20.0,   # periods down to 20 s
    #     n_omega=260,
    #     n_tau=120,
    #     despike_at_start=True,
    #     despike_halfwidth=1,
    #     use_weights=True,
    # )


    # If you also want the multi-lineout plot:
    # data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)
    # plot_ttc_with_lineouts(data, start=START_DIAG_IDX, add_antidiag_se=True, despike_at_start=True, despike_halfwidth=1)

    DT_S = 1.0
    START_DIAG_IDX = 2000

    data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)

    # --- extract lineout ---
    t, y = extract_antidiagonal_lineout(
        data.ttc,
        start_idx=START_DIAG_IDX,
        dt_s=DT_S,
    )

    # --- bootstrap FFT peak ---
    # fmin, fmax = 1 / 1000, 1 / 10  # Hz
    #
    # f_hat, f_lo, f_hi, f_samp = bootstrap_peak_frequency(
    #     y, DT_S,
    #     fmin=fmin,
    #     fmax=fmax,
    #     seg_len=None,  # auto
    #     overlap=0.5,
    #     window="hann",
    #     detrend=True,
    #     n_boot=2000,
    #     ci=0.68,
    #     rng_seed=1,
    # )
    #
    #
    # print(f"Peak frequency: {f_hat:.6g} Hz  (68% CI [{f_lo:.6g}, {f_hi:.6g}])")
    # print(f"Period: {1 / f_hat:.2f} s  (CI [{1 / f_hi:.2f}, {1 / f_lo:.2f}] s)")

    # drop_first = 5
    # dt_lineout = DT_S * 2.0  # because tau = 2*k*dt_s along anti-diagonal
    # f_peak, sf, T, sT, df = fft_peak_with_bin_uncertainty(y, dt_lineout, drop_first=drop_first)
    #
    # print(f"FFT peak: {f_peak:.6g} Hz")
    # print(f"Bin spacing Δf = {df:.3g} Hz  -> σ_f ≈ {sf:.3g} Hz")
    # print(f"Period: {T:.2f} s  -> σ_T ≈ {sT:.2f} s")

    data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)

    out = plot_period_vs_diagonal_start(
        data,
        dt_s=DT_S,
        drop_first_lineout=5,
        half_window=5,  # <- choose smoothing strength
        band_ci=0.68,  # <- 0.68 or 0.95 etc
        fmin=1 / 1000,
        fmax=1 / 10,
        detrend=True,
        window="hann",
    )

    # out = plot_ttc_lineout_fft(
    #     data,
    #     start_idx=START_DIAG_IDX,
    #     dt_s=DT_S,
    #     drop_first=5,     # try 0, 5, 10
    #     detrend=True,
    #     window=True,
    # )
    # print("Peak period (s):", out["period_s"])