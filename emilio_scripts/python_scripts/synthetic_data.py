import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")  # requires PyQt6; change to "macosx" if you prefer
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def make_ttc_product_gui_periods(
    *,
    n: int = 900,
    dt_s: float = 1.0,
    clip_percentile: float = 99.5,
    cmap: str = "plasma",
):
    """
    Interactive TTC model GUI (product-form G) with *period* sliders.

    Model:
      u = (t1+t2)/2
      v = (t2-t1)
      wv = 2π/Tv
      wu = 2π/Tu

      C(t1,t2) = C0
               + A*cos(wv*v)
               + m*cos(wu*u + phi) * cos(kv*wu*v)

    Notes
    -----
    - Tv, Tu are in seconds (if dt_s is seconds per index).
    - kv controls "diamond stretch" (how fast modulation varies along v).
    - TTC plot is square (aspect="equal").
    """

    # -------------------------
    # grid
    # -------------------------
    t = np.arange(n, dtype=np.float64) * float(dt_s)
    T1, T2 = np.meshgrid(t, t, indexing="xy")
    V = (T2 - T1)
    U = (T1 + T2) * 0.5

    def omega_from_period(T):
        T = float(T)
        if T <= 0:
            return 0.0
        return 2.0 * np.pi / T

    def model(C0, A, Tv, m, Tu, kv, phi):
        wv = omega_from_period(Tv)
        wu = omega_from_period(Tu)
        stripes = A * np.cos(wv * V)
        Gprod = np.cos(wu * U + phi) * np.cos((kv * wu) * V)
        return C0 + stripes + m * Gprod

    def clip_for_display(C):
        if clip_percentile is None:
            return C
        hi = np.percentile(C, float(clip_percentile))
        lo = np.percentile(C, 0.0)
        return np.clip(C, lo, hi)

    # -------------------------
    # defaults: ~250 s periods
    # -------------------------
    p0 = dict(
        C0=1.0,
        A=0.6,
        Tv=250.0,
        m=0.6,
        Tu=250.0,
        kv=0.5,    # 0.5 mimics cos(wu*v/2) style; adjust for stretch
        phi=0.0,
    )

    # -------------------------
    # layout
    # -------------------------
    fig = plt.figure(figsize=(12.8, 7.2))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 0.28],
        left=0.06, right=0.985, top=0.93, bottom=0.09,
        hspace=0.24,
    )

    ax_img = fig.add_subplot(gs[0, 0])
    ax_sl = fig.add_subplot(gs[1, 0])
    ax_sl.axis("off")

    C = model(**p0)
    Cplot = clip_for_display(C)

    im = ax_img.imshow(
        Cplot,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        extent=[t[0], t[-1], t[0], t[-1]],
    )
    ax_img.set_xlabel(r"$t_1$ (s)")
    ax_img.set_ylabel(r"$t_2$ (s)")
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.03)
    cbar.set_label("C(t₁,t₂) (a.u.)")

    # slider grid (wider spacing, narrower axes so labels don't collide)
    gs_sl = gs[1, 0].subgridspec(2, 4, wspace=0.40, hspace=1.05)

    def slider_ax(r, c, pad=0.018):
        ax = fig.add_subplot(gs_sl[r, c])
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
        return ax

    # Row 1: offsets/amplitudes
    s_C0 = Slider(slider_ax(0, 0), "C0", 0.0, 3.0, valinit=p0["C0"], valfmt="%.3f")
    s_A  = Slider(slider_ax(0, 1), "A", -2.0, 2.0, valinit=p0["A"], valfmt="%.3f")
    s_m  = Slider(slider_ax(0, 2), "m", -2.0, 2.0, valinit=p0["m"], valfmt="%.3f")
    s_phi = Slider(slider_ax(0, 3), "φ", -np.pi, np.pi, valinit=p0["phi"], valfmt="%.3f")

    # Row 2: periods + stretch factor
    # Keep ranges centered on your use-case (~250 s), but still flexible
    s_Tv = Slider(slider_ax(1, 0), "Tv (s)", 50.0, 600.0, valinit=p0["Tv"], valfmt="%.1f")
    s_Tu = Slider(slider_ax(1, 1), "Tu (s)", 50.0, 600.0, valinit=p0["Tu"], valfmt="%.1f")
    s_kv = Slider(slider_ax(1, 2), "kv", 0.0, 2.0, valinit=p0["kv"], valfmt="%.3f")

    # a spare slot (you can add another slider later if wanted)
    ax_spare = fig.add_subplot(gs_sl[1, 3])
    ax_spare.axis("off")

    for s in (s_C0, s_A, s_m, s_phi, s_Tv, s_Tu, s_kv):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # buttons
    ax_reset = fig.add_axes([0.06, 0.02, 0.08, 0.045])
    b_reset = Button(ax_reset, "Reset")

    ax_save = fig.add_axes([0.15, 0.02, 0.10, 0.045])
    b_save = Button(ax_save, "Save PNG")

    busy = {"flag": False}

    def update(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        try:
            C = model(
                C0=float(s_C0.val),
                A=float(s_A.val),
                Tv=float(s_Tv.val),
                m=float(s_m.val),
                Tu=float(s_Tu.val),
                kv=float(s_kv.val),
                phi=float(s_phi.val),
            )
            Cplot = clip_for_display(C)
            im.set_data(Cplot)

            vmin = float(np.min(Cplot))
            vmax = float(np.max(Cplot))
            if vmax <= vmin:
                vmax = vmin + 1e-12
            im.set_clim(vmin, vmax)

            ax_img.set_title(
                f"TTC product model | Tv={s_Tv.val:.1f}s, Tu={s_Tu.val:.1f}s, kv={s_kv.val:.3f}"
            )
            fig.canvas.draw_idle()
        finally:
            busy["flag"] = False

    for s in (s_C0, s_A, s_m, s_phi, s_Tv, s_Tu, s_kv):
        s.on_changed(update)

    def on_reset(_event):
        for s in (s_C0, s_A, s_m, s_phi, s_Tv, s_Tu, s_kv):
            s.reset()
        update()

    def on_save(_event):
        fname = "ttc_product_gui_periods.png"
        fig.savefig(fname, dpi=200)
        print(f"Saved: {fname}")

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(on_save)

    update()
    plt.show()

if __name__ == "__main__":
    make_ttc_product_gui_periods(n=900, dt_s=1.0, clip_percentile=99.5)