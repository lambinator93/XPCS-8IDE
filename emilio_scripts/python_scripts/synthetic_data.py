# ============================================================
# Interactive TTC toy model GUI (Matplotlib sliders) – MIXED kernel
# ============================================================

from __future__ import annotations

import numpy as np
import matplotlib as mpl
mpl.use("macosx")
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, RadioButtons


# ----------------------------
# Model helpers
# ----------------------------
def make_x_t(n, dt, v, A, P, phi, sigma_rw, seed):
    rng = np.random.default_rng(int(seed))
    t = np.arange(n) * dt

    if sigma_rw > 0:
        rw = np.cumsum(rng.normal(0.0, sigma_rw, n))
        rw -= rw[0]
    else:
        rw = np.zeros(n)

    periodic = A * np.sin(2 * np.pi * t / P + phi) if (A != 0 and P > 0) else 0.0
    x = v * t + periodic + rw
    return t, x


def make_c2_mixed(
    t, x,
    *, beta0, tau_beta,
    w, sigma_s,
    tau_c, alpha,
    p_tau, use_tau_cos
):
    t1 = t[:, None]
    t2 = t[None, :]
    tau = t2 - t1
    T = 0.5 * (t1 + t2)

    betaT = beta0 * np.exp(-T / tau_beta) if tau_beta > 0 else beta0

    dx = x[None, :] - x[:, None]
    Kx = np.exp(-(dx**2) / (2 * sigma_s**2))

    Ktau = np.exp(-(np.abs(tau) / tau_c)**alpha)
    if use_tau_cos and p_tau > 0:
        Ktau *= np.cos(2 * np.pi * np.abs(tau) / p_tau)

    K = w * Kx + (1 - w) * Ktau
    return 1.0 + betaT * K


def robust_clim(C, plo=0.5, phi=99.5):
    lo, hi = np.percentile(C, [plo, phi])
    if lo == hi:
        lo -= 1e-6
        hi += 1e-6
    return lo, hi


# ----------------------------
# Main GUI
# ----------------------------
def main():
    n = 420
    dt = 1.0

    # initial parameters
    beta0, tau_beta = 0.25, 800.0
    w0, sigma_s0 = 0.35, 1.2
    tau_c0, alpha0, p_tau0 = 90.0, 1.0, 120.0
    v0, A0, P0, phi0 = 0.0, 2.0, 120.0, 0.0
    sigma_rw0, seed0 = 0.03, 1
    use_tau_cos0 = True

    # ----------------------------
    # Figure layout
    # ----------------------------
    fig = plt.figure(figsize=(13.6, 8.8))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.5, 1.7],
        width_ratios=[1.0, 1.3],
        hspace=0.25,
        wspace=0.28,
    )

    ax_xt = fig.add_subplot(gs[0, 0])
    ax_ttc = fig.add_subplot(gs[0, 1])
    ax_eq = fig.add_subplot(gs[1, 0])
    ax_panel = fig.add_subplot(gs[1, 1])

    ax_eq.axis("off")
    ax_panel.axis("off")

    # ----------------------------
    # Equation box
    # ----------------------------
    eq_text = (
        "Toy TTC model (mixed kernel)\n\n"
        "c2(t1,t2) = 1 + β(T)[ w·Kx(Δx) + (1−w)·Kτ(τ) ]\n"
        "Δx = x(t2) − x(t1),   τ = t2 − t1,   T = (t1+t2)/2\n\n"
        "Kx(Δx) = exp(−Δx² / 2σs²)\n"
        "Kτ(τ)  = exp(−(|τ|/τc)^α) · cos(2π|τ|/Pτ)\n\n"
        "β(T) = β0 exp(−T/τβ)\n\n"
        "x(t) = v t + A sin(2πt/P + ϕ) + RW(t)\n\n"
        "Diagonal stripes → lag kernel (small w)\n"
        "Checker/grid → periodic x(t) or cos(Kτ)"
    )

    ax_eq.text(
        0.02, 0.98, eq_text,
        va="top", ha="left",
        fontsize=10.2,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.96),
        transform=ax_eq.transAxes,
    )

    # ----------------------------
    # Initial data
    # ----------------------------
    t, x = make_x_t(n, dt, v0, A0, P0, phi0, sigma_rw0, seed0)
    C = make_c2_mixed(
        t, x,
        beta0=beta0, tau_beta=tau_beta,
        w=w0, sigma_s=sigma_s0,
        tau_c=tau_c0, alpha=alpha0,
        p_tau=p_tau0, use_tau_cos=use_tau_cos0,
    )

    (line_xt,) = ax_xt.plot(t, x, lw=2)
    ax_xt.set_title("x(t)")
    ax_xt.grid(True, alpha=0.3)

    vmin, vmax = robust_clim(C)
    im = ax_ttc.imshow(C, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)
    ax_ttc.set_title("c2(t1,t2)")
    fig.colorbar(im, ax=ax_ttc, fraction=0.04, pad=0.02)

    # ----------------------------
    # Buttons UNDER TEXT BOX (new)
    # ----------------------------
    eq_bbox = ax_eq.get_position()
    bx, by, bw, bh = eq_bbox.x0, eq_bbox.y0, eq_bbox.width, eq_bbox.height

    ax_btn_cos = fig.add_axes([bx + 0.05 * bw, by + 0.02 * bh, 0.40 * bw, 0.12 * bh])
    ax_btn_seed = fig.add_axes([bx + 0.52 * bw, by + 0.02 * bh, 0.40 * bw, 0.12 * bh])

    btn_cos = Button(ax_btn_cos, "cos in Kτ: ON")
    btn_seed = Button(ax_btn_seed, "Reseed RW")

    # ----------------------------
    # Sliders (right panel unchanged)
    # ----------------------------
    pb = ax_panel.get_position()
    x0, y0, w, h = pb.x0, pb.y0, pb.width, pb.height

    def add_slider(y, label, vmin, vmax, vinit):
        ax = fig.add_axes([x0 + 0.08 * w, y, 0.88 * w, 0.038])
        return Slider(ax, label, vmin, vmax, valinit=vinit)

    ytop = y0 + 0.94 * h
    dy = 0.085 * h

    s_beta0 = add_slider(ytop - 0*dy, "β0", 0, 1, beta0)
    s_tau_beta = add_slider(ytop - 1*dy, "τβ", 0, 4000, tau_beta)
    s_w = add_slider(ytop - 2*dy, "w", 0, 1, w0)
    s_sigma_s = add_slider(ytop - 3*dy, "σs", 0.05, 10, sigma_s0)
    s_tau_c = add_slider(ytop - 4*dy, "τc", 1, 800, tau_c0)
    s_alpha = add_slider(ytop - 5*dy, "α", 0.2, 2.5, alpha0)
    s_p_tau = add_slider(ytop - 6*dy, "Pτ", 5, 800, p_tau0)
    s_v = add_slider(ytop - 7*dy, "v", -0.02, 0.02, v0)
    s_A = add_slider(ytop - 8*dy, "A", 0, 10, A0)
    s_P = add_slider(ytop - 9*dy, "P", 5, 600, P0)
    s_phi = add_slider(ytop - 10*dy, "ϕ", -np.pi, np.pi, phi0)
    s_sigma_rw = add_slider(ytop - 11*dy, "σrw", 0, 0.25, sigma_rw0)
    s_seed = add_slider(ytop - 12*dy, "seed", 0, 999, seed0)

    # ----------------------------
    # Update logic
    # ----------------------------
    state = {"use_tau_cos": True, "seed_bump": 0}

    def recompute():
        seed = int(s_seed.val) + state["seed_bump"]
        t, x = make_x_t(
            n, dt, s_v.val, s_A.val, s_P.val,
            s_phi.val, s_sigma_rw.val, seed
        )
        C = make_c2_mixed(
            t, x,
            beta0=s_beta0.val, tau_beta=s_tau_beta.val,
            w=s_w.val, sigma_s=s_sigma_s.val,
            tau_c=s_tau_c.val, alpha=s_alpha.val,
            p_tau=s_p_tau.val, use_tau_cos=state["use_tau_cos"]
        )
        return t, x, C

    def update(_=None):
        t, x, C = recompute()
        line_xt.set_data(t, x)
        ax_xt.relim()
        ax_xt.autoscale_view()
        im.set_data(C)
        im.set_clim(*robust_clim(C))
        fig.canvas.draw_idle()

    def toggle_cos(_):
        state["use_tau_cos"] = not state["use_tau_cos"]
        btn_cos.label.set_text("cos in Kτ: ON" if state["use_tau_cos"] else "cos in Kτ: OFF")
        update()

    def reseed(_):
        state["seed_bump"] += 1
        update()

    for s in (
        s_beta0, s_tau_beta, s_w, s_sigma_s, s_tau_c, s_alpha,
        s_p_tau, s_v, s_A, s_P, s_phi, s_sigma_rw, s_seed
    ):
        s.on_changed(update)

    btn_cos.on_clicked(toggle_cos)
    btn_seed.on_clicked(reseed)

    plt.show()


if __name__ == "__main__":
    main()