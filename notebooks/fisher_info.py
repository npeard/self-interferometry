import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Fisher Information for Phase Estimation in Michelson Interferometers

    ## Single Interferometer

    A Michelson interferometer converts a phase perturbation $\delta\varphi(t)$ into a
    detected intensity:

    $$I(\varphi) = \frac{I_0}{2}\bigl(1 + V\cos(\varphi_0 + \delta\varphi)\bigr)$$

    where $\varphi_0$ is the static operating-point phase offset and $V$ is the fringe
    visibility (set to 1 for an ideal interferometer).

    For shot-noise-limited detection (Poisson statistics), the Fisher Information on
    $\delta\varphi$ from a single intensity measurement is:

    $$\mathcal{F}(\delta\varphi) = \frac{\bigl(\partial I / \partial \delta\varphi\bigr)^2}{I}
    = \frac{I_0}{2} \cdot \frac{V^2 \sin^2(\varphi_0 + \delta\varphi)}{1 + V\cos(\varphi_0 + \delta\varphi)}$$

    - **Minimum FI**: at $\varphi_0 + \delta\varphi = 0, \pm\pi$ (bright/dark fringe — flat intensity, zero slope)
    - **Maximum FI**: near quadrature $\varphi_0 + \delta\varphi \approx \pm\pi/2$ (steepest fringe slope)

    ## Multiple Interferometers

    With $K$ interferometers probing the **same** surface vibration $\delta\varphi(t)$, each
    with a different static offset $\varphi_0^{(k)}$, the total Fisher Information is additive:

    $$\mathcal{F}_\text{total}(\delta\varphi) = \sum_{k=1}^{K} \mathcal{F}_k(\delta\varphi)$$

    If the offsets are uniformly spaced, there is always at least one interferometer near
    quadrature, so the total FI stays well above the single-interferometer minimum for all $\delta\varphi$.
    """)
    return


@app.cell
def _(np):
    I0 = 1.0
    V = 1.0
    N_phase = 1000
    delta_phi = np.linspace(-np.pi, np.pi, N_phase)

    def michelson_intensity(dphi, p0, i0=1.0, vis=1.0):
        """Detected intensity for a Michelson interferometer."""
        return (i0 / 2) * (1 + vis * np.cos(p0 + dphi))

    def fisher_info(dphi, p0, i0=1.0, vis=1.0):
        """Shot-noise-limited Fisher Information on delta_phi."""
        total_phase = p0 + dphi
        numerator = (i0 / 2) ** 2 * vis**2 * np.sin(total_phase) ** 2
        denominator = michelson_intensity(dphi, p0, i0, vis)
        denominator = np.maximum(denominator, 1e-15)
        return numerator / denominator

    return I0, V, delta_phi, fisher_info, michelson_intensity


@app.cell
def _(I0, V, delta_phi, fisher_info, michelson_intensity, np, plt):
    _offsets_single = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    _offset_labels = ["0", "\u03c0/6", "\u03c0/4", "\u03c0/3", "\u03c0/2"]

    _fig1, _axes1 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    _colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(_offsets_single)))

    for _p0, _label, _c in zip(_offsets_single, _offset_labels, _colors):
        _I_det = michelson_intensity(delta_phi, _p0, I0, V)
        _fi = fisher_info(delta_phi, _p0, I0, V)
        _axes1[0].plot(delta_phi, _I_det, color=_c, label=rf"$\varphi_0 = {_label}$")
        _axes1[1].plot(delta_phi, _fi, color=_c, label=rf"$\varphi_0 = {_label}$")

    _axes1[0].set_ylabel("Intensity $I$")
    _axes1[0].set_title("Single Interferometer: Intensity vs Phase Perturbation")
    _axes1[0].legend(fontsize=8, ncol=2)
    _axes1[0].grid(True, alpha=0.3)

    _axes1[1].set_xlabel(r"Phase perturbation $\delta\varphi$ (rad)")
    _axes1[1].set_ylabel(r"Fisher Information $\mathcal{F}(\delta\varphi)$")
    _axes1[1].set_title("Single Interferometer: Fisher Information vs Phase Perturbation")
    _axes1[1].legend(fontsize=8, ncol=2)
    _axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Top:** Detected intensity for different operating-point offsets $\varphi_0$.
    When the intensity is flat (bright or dark fringe), the slope is zero and we
    have no phase sensitivity.

    **Bottom:** Fisher Information peaks where the fringe slope is steepest
    ($\varphi_0 + \delta\varphi \approx \pm\pi/2$) and vanishes at the fringe
    extrema where $\sin(\varphi_0 + \delta\varphi) = 0$.
    """)
    return


@app.cell
def _(I0, V, delta_phi, fisher_info, np, plt):
    _N_offset = 500
    _phi0_vals = np.linspace(-np.pi, np.pi, _N_offset)

    _phi0_grid, _dphi_grid = np.meshgrid(_phi0_vals, delta_phi, indexing="ij")
    _FI_map = fisher_info(_dphi_grid, _phi0_grid, I0, V)

    _fig2, _ax2 = plt.subplots(figsize=(10, 5))
    _im = _ax2.imshow(
        _FI_map.T,
        extent=[_phi0_vals[0], _phi0_vals[-1], delta_phi[0], delta_phi[-1]],
        aspect="auto",
        origin="lower",
        cmap="inferno",
    )
    _ax2.set_xlabel(r"Operating-point offset $\varphi_0$ (rad)")
    _ax2.set_ylabel(r"Phase perturbation $\delta\varphi$ (rad)")
    _ax2.set_title(r"Fisher Information $\mathcal{F}(\delta\varphi;\,\varphi_0)$")
    _fig2.colorbar(_im, ax=_ax2, label=r"$\mathcal{F}$")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The 2D map shows FI as a function of both the static offset $\varphi_0$ and
    the perturbation $\delta\varphi$. The dark bands (zero FI) lie along
    $\varphi_0 + \delta\varphi = n\pi$ — exactly the bright and dark fringes
    where the intensity slope vanishes. A single interferometer always has blind
    spots in this map.
    """)
    return


@app.cell
def _(I0, V, delta_phi, fisher_info, np, plt):
    _fig3, _axes3 = plt.subplots(1, 2, figsize=(14, 5))

    _K_values = [1, 2, 3, 4, 5]
    _colors3 = plt.cm.plasma(np.linspace(0.15, 0.85, len(_K_values)))

    for _K, _c in zip(_K_values, _colors3):
        _offsets = np.linspace(0, np.pi, _K, endpoint=False)
        _FI_total = np.zeros_like(delta_phi)
        for _p0 in _offsets:
            _FI_total += fisher_info(delta_phi, _p0, I0, V)
        _axes3[0].plot(delta_phi, _FI_total, color=_c, label=f"K = {_K}", linewidth=1.5)

    _FI_quad = fisher_info(delta_phi, np.pi / 2, I0, V)
    _axes3[0].axhline(
        np.max(_FI_quad),
        color="gray",
        linestyle="--",
        alpha=0.6,
        label="Single max (quadrature)",
    )

    _axes3[0].set_xlabel(r"Phase perturbation $\delta\varphi$ (rad)")
    _axes3[0].set_ylabel(r"Total Fisher Information $\mathcal{F}_\mathrm{total}$")
    _axes3[0].set_title("Total FI: Uniformly Spaced Interferometers")
    _axes3[0].legend(fontsize=8)
    _axes3[0].grid(True, alpha=0.3)

    _K_range = np.arange(1, 21)
    _min_FI = []
    _mean_FI = []
    for _K in _K_range:
        _offsets = np.linspace(0, np.pi, _K, endpoint=False)
        _FI_total = np.zeros_like(delta_phi)
        for _p0 in _offsets:
            _FI_total += fisher_info(delta_phi, _p0, I0, V)
        _min_FI.append(np.min(_FI_total))
        _mean_FI.append(np.mean(_FI_total))

    _axes3[1].plot(_K_range, _min_FI, "o-", color="tab:red", label=r"Worst-case (min over $\delta\varphi$)")
    _axes3[1].plot(_K_range, _mean_FI, "s-", color="tab:blue", label=r"Average over $\delta\varphi$")
    _axes3[1].axhline(
        np.max(_FI_quad),
        color="gray",
        linestyle="--",
        alpha=0.6,
        label="Single max (quadrature)",
    )
    _axes3[1].set_xlabel("Number of interferometers K")
    _axes3[1].set_ylabel(r"Fisher Information $\mathcal{F}$")
    _axes3[1].set_title("Scaling of FI with Number of Interferometers")
    _axes3[1].legend(fontsize=8)
    _axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Left:** Total Fisher Information from $K$ uniformly-spaced interferometers.
    With $K=1$, the FI drops to zero at the fringe extrema. With $K \geq 2$ and
    different offsets, the blind spots of one interferometer are filled in by the
    others. The total FI stays above the single-interferometer maximum for all
    $\delta\varphi$ once $K \geq 3$.

    **Right:** The worst-case (minimum over all $\delta\varphi$) and average FI as
    a function of $K$. Even two interferometers eliminate the zero-FI blind spots.
    The worst-case FI converges rapidly as $K$ grows, confirming that diverse phase
    offsets guarantee robust phase sensitivity at all operating points.
    """)
    return


@app.cell
def _(I0, V, fisher_info, np, plt):
    _t = np.linspace(0, 4 * np.pi, 2000)
    _amplitude = 0.8 * np.pi
    _delta_phi_t = _amplitude * np.sin(_t)

    _fig4, _axes4 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    _axes4[0].plot(_t, _delta_phi_t, "k-", linewidth=1)
    _axes4[0].set_ylabel(r"$\delta\varphi(t)$ (rad)")
    _axes4[0].set_title("Surface Vibration (Phase Perturbation)")
    _axes4[0].grid(True, alpha=0.3)

    _FI_single_0 = fisher_info(_delta_phi_t, 0.0, I0, V)
    _axes4[1].plot(_t, _FI_single_0, color="tab:red", label=r"$K=1$, $\varphi_0=0$", linewidth=1)

    _FI_single_quad = fisher_info(_delta_phi_t, np.pi / 2, I0, V)
    _axes4[1].plot(
        _t,
        _FI_single_quad,
        color="tab:orange",
        label=r"$K=1$, $\varphi_0=\pi/2$",
        linewidth=1,
    )

    _offsets_3 = np.linspace(0, np.pi, 3, endpoint=False)
    _FI_multi = np.zeros_like(_delta_phi_t)
    for _p0 in _offsets_3:
        _FI_multi += fisher_info(_delta_phi_t, _p0, I0, V)
    _axes4[1].plot(_t, _FI_multi, color="tab:blue", label=r"$K=3$, uniform", linewidth=1.5)

    _axes4[1].set_ylabel(r"$\mathcal{F}(t)$")
    _axes4[1].set_title("Fisher Information Over Time")
    _axes4[1].legend(fontsize=8)
    _axes4[1].grid(True, alpha=0.3)

    for _idx, _p0 in enumerate(_offsets_3):
        _I_t = (I0 / 2) * (1 + np.cos(_p0 + _delta_phi_t))
        _axes4[2].plot(_t, _I_t, linewidth=1, label=rf"$\varphi_0 = {_p0:.2f}$")

    _axes4[2].set_xlabel("Time (a.u.)")
    _axes4[2].set_ylabel("Intensity")
    _axes4[2].set_title(r"Detected Intensities ($K=3$ Interferometers)")
    _axes4[2].legend(fontsize=8)
    _axes4[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Top:** A sinusoidal surface vibration $\delta\varphi(t)$.

    **Middle:** FI over time. A single interferometer at $\varphi_0=0$ loses all
    phase information whenever $\delta\varphi$ passes through $0$ or $\pm\pi$
    (bright/dark fringes). Even the optimally-placed single interferometer
    ($\varphi_0=\pi/2$) still has periodic blind spots. With $K=3$ uniformly
    spaced interferometers, the total FI never drops to zero — at least one
    channel is always near quadrature.

    **Bottom:** The three detected intensity signals. When one channel is flat
    (zero slope), the others are changing rapidly, providing complementary
    information about the phase.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | Configuration | Worst-case FI | Sensitivity |
    |---|---|---|
    | Single interferometer at fringe peak/null | 0 | **Blind** — no phase info |
    | Single interferometer at quadrature | Maximum for 1 channel | Good, but periodic blind spots |
    | $K$ interferometers, same offset | $K \times$ single (but still has zeros) | Scales amplitude, not robustness |
    | $K$ interferometers, diverse offsets | $> 0$ for all $\delta\varphi$ | **Robust** — no blind spots |

    The key insight: multiple interferometers with **different** phase offsets
    provide complementary information. The combined Fisher Information exceeds
    what any single interferometer can achieve, not just in magnitude but in
    **coverage** — eliminating the blind spots where a single interferometer
    has zero sensitivity to phase displacements.
    """)
    return


if __name__ == "__main__":
    app.run()
