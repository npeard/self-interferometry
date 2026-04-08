import marimo

__generated_with = '0.20.4'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Fisher Information for Phase Estimation in Michelson Interferometers

    ## Single Interferometer

    A Michelson interferometer converts a phase perturbation $\delta\varphi(t)$ into a
    detected intensity:

    $$I(\varphi) = \frac{I_0}{2}\bigl(1 + V\cos(\varphi_0 + \delta\varphi)\bigr) + \mathcal{N}(0,\sigma)$$

    where $\varphi_0$ is the static operating-point phase offset and $V$ is the fringe
    visibility (set to 1 for an ideal interferometer). $\mathcal{N}(0,\sigma)$ represents our independent
    zero-mean Gaussian noise on the channel for shot-noise-limited detection (Poisson statistics).

    The log likelihood is then
    $$ \log f(I; \varphi) = \frac{I_0^2}{4\sigma^2}\bigl(I - 1 - V\cos(\varphi_0 + \delta\varphi)\bigr)^2$$

    The Fisher Information on $\delta\varphi$ from a single intensity measurement is
    the expectation of the squared derivative of the log-likelihood with respect to $\delta\varphi$:

    $$\mathcal{I}(I; \delta\varphi) = \bigl(\partial \log f / \partial \delta\varphi\bigr)^2
    = \frac{I_0^2 V^2}{4\sigma^2} \cdot \sin^2(\varphi_0 + \delta\varphi) $$

    - **Minimum FI**: at $\varphi_0 + \delta\varphi = 0, \pm\pi$ (bright/dark fringe — flat intensity, zero slope)
    - **Maximum FI**: near quadrature $\varphi_0 + \delta\varphi \approx \pm\pi/2$ (steepest fringe slope)

    ## Multiple Interferometers

    With $K$ interferometers probing the **same** surface vibration $\delta\varphi(t)$, each
    with a different static offset $\varphi_0^{(k)}$, the total Fisher Information is additive:

    $$\mathcal{F}_\text{total}(\delta\varphi) = \sum_{k=1}^{K} \mathcal{F}_k(\delta\varphi)$$

    If the offsets are uniformly spaced, there is always at least one interferometer near
    quadrature, so the total FI stays well above the single-interferometer minimum for all $\delta\varphi$.
    """)


@app.cell
def _(np):
    I0 = 1.0
    V = 1.0
    N_phase = 1000
    delta_phi = np.linspace(-np.pi, np.pi, N_phase)

    def michelson_intensity(dphi, p0, i0=1.0, vis=1.0):
        """Detected intensity for a Michelson interferometer."""
        return (i0 / 2) * (1 + vis * np.cos(p0 + dphi))

    def fisher_info(dphi, p0, i0=1.0, vis=1.0, noise_level=1.0):
        """Shot-noise-limited Fisher Information on delta_phi."""
        total_phase = p0 + dphi
        numerator = (i0 / 2) ** 2 * vis**2 * np.sin(total_phase) ** 2
        # denominator should be noise level, variance of the Gaussian noise
        return numerator / noise_level

    def fisher_info_displacement(d, p0, wavelength, i0=1.0, vis=1.0, noise_level=1.0):
        """Fisher Information on physical displacement d (in nm).

        Phase is phi = 4*pi*d / wavelength (Michelson round-trip).
        FI w.r.t. d picks up the Jacobian factor (4*pi/wavelength)^2.
        """
        phase = 4 * np.pi * d / wavelength
        jacobian_sq = (4 * np.pi / wavelength) ** 2
        return (
            (i0 / 2) ** 2 * vis**2 * np.sin(p0 + phase) ** 2 * jacobian_sq / noise_level
        )

    return I0, V, delta_phi, fisher_info, michelson_intensity


@app.cell
def _(I0, V, delta_phi, fisher_info, michelson_intensity, np, plt):
    _offsets_single = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    _offset_labels = ['0', '\u03c0/6', '\u03c0/4', '\u03c0/3', '\u03c0/2']

    _fig1, _axes1 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    _colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(_offsets_single)))

    for _p0, _label, _c in zip(_offsets_single, _offset_labels, _colors, strict=False):
        _I_det = michelson_intensity(delta_phi, _p0, I0, V)
        _fi = fisher_info(delta_phi, _p0, I0, V)
        _axes1[0].plot(delta_phi, _I_det, color=_c, label=rf'$\varphi_0 = {_label}$')
        _axes1[1].plot(delta_phi, _fi, color=_c, label=rf'$\varphi_0 = {_label}$')

    _axes1[0].set_ylabel('Intensity $I$')
    _axes1[0].set_title('Single Interferometer: Intensity vs Phase Perturbation')
    _axes1[0].legend(fontsize=8, ncol=2)
    _axes1[0].grid(True, alpha=0.3)

    _axes1[1].set_xlabel(r'Phase perturbation $\delta\varphi$ (rad)')
    _axes1[1].set_ylabel(r'Fisher Information $\mathcal{F}(\delta\varphi)$')
    _axes1[1].set_title(
        'Single Interferometer: Fisher Information vs Phase Perturbation'
    )
    _axes1[1].legend(fontsize=8, ncol=2)
    _axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


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


@app.cell
def _(I0, V, delta_phi, fisher_info, np, plt):
    _N_offset = 500
    _phi0_vals = np.linspace(-np.pi, np.pi, _N_offset)

    _phi0_grid, _dphi_grid = np.meshgrid(_phi0_vals, delta_phi, indexing='ij')
    _FI_map = fisher_info(_dphi_grid, _phi0_grid, I0, V)

    _fig2, _ax2 = plt.subplots(figsize=(10, 5))
    _im = _ax2.imshow(
        _FI_map.T,
        extent=[_phi0_vals[0], _phi0_vals[-1], delta_phi[0], delta_phi[-1]],
        aspect='auto',
        origin='lower',
        cmap='inferno',
    )
    _ax2.set_xlabel(r'Operating-point offset $\varphi_0$ (rad)')
    _ax2.set_ylabel(r'Phase perturbation $\delta\varphi$ (rad)')
    _ax2.set_title(r'Fisher Information $\mathcal{F}(\delta\varphi;\,\varphi_0)$')
    _fig2.colorbar(_im, ax=_ax2, label=r'$\mathcal{F}$')
    plt.tight_layout()
    plt.show()


@app.cell
def _(mo):
    mo.md(r"""
    The 2D map shows FI as a function of both the static offset $\varphi_0$ and
    the perturbation $\delta\varphi$. The dark bands (zero FI) lie along
    $\varphi_0 + \delta\varphi = n\pi$ — exactly the bright and dark fringes
    where the intensity slope vanishes. A single interferometer always has blind
    spots in this map.
    """)


@app.cell
def _(I0, V, delta_phi, fisher_info, np, plt):
    _fig3, _axes3 = plt.subplots(1, 2, figsize=(14, 5))

    _K_values = [1, 2, 3, 4, 5]
    _colors3 = plt.cm.plasma(np.linspace(0.15, 0.85, len(_K_values)))

    for _K, _c in zip(_K_values, _colors3, strict=False):
        _offsets = np.linspace(0, np.pi, _K, endpoint=False)
        _FI_total = np.zeros_like(delta_phi)
        for _p0 in _offsets:
            _FI_total += fisher_info(delta_phi, _p0, I0, V)
        _axes3[0].plot(delta_phi, _FI_total, color=_c, label=f'K = {_K}', linewidth=1.5)

    _FI_quad = fisher_info(delta_phi, np.pi / 2, I0, V)
    _axes3[0].axhline(
        np.max(_FI_quad),
        color='gray',
        linestyle='--',
        alpha=0.6,
        label='Single max (quadrature)',
    )

    _axes3[0].set_xlabel(r'Phase perturbation $\delta\varphi$ (rad)')
    _axes3[0].set_ylabel(r'Total Fisher Information $\mathcal{F}_\mathrm{total}$')
    _axes3[0].set_title('Total FI: Uniformly Spaced Interferometers')
    _axes3[0].legend(fontsize=8)
    _axes3[0].grid(True, alpha=0.3)

    _K_range = np.arange(1, 21)

    # Uniform unit-circle [0, 2π) spacing
    _min_FI_uc = []
    for _K in _K_range:
        _offsets = np.linspace(0, np.pi, _K, endpoint=False)
        _FI_total = np.zeros_like(delta_phi)
        for _p0 in _offsets:
            _FI_total += fisher_info(delta_phi, _p0, I0, V)
        _min_FI_uc.append(np.min(_FI_total))

    # Monte Carlo: random phase offsets drawn uniformly from [0, 2π)
    _N_mc = 1000
    _rng = np.random.default_rng(42)
    _mc_min_mean = np.zeros(len(_K_range))
    _mc_min_std = np.zeros(len(_K_range))
    for _i, _K in enumerate(_K_range):
        _rand_offsets = _rng.uniform(0, np.pi, (_N_mc, _K))
        _all_phases = (
            _rand_offsets[:, :, np.newaxis] + delta_phi[np.newaxis, np.newaxis, :]
        )
        _FI_trials = np.sum((I0 / 2) ** 2 * V**2 * np.sin(_all_phases) ** 2, axis=1)
        _mc_min_mean[_i] = np.mean(np.min(_FI_trials, axis=1))
        _mc_min_std[_i] = np.std(np.min(_FI_trials, axis=1))

    _axes3[1].plot(
        _K_range,
        _min_FI_uc,
        'o-',
        color='tab:blue',
        label=r'Uniform $[0,\pi)$: worst-case',
        linewidth=1.5,
    )
    _axes3[1].plot(
        _K_range,
        _mc_min_mean,
        's-',
        color='tab:orange',
        label=r'Monte Carlo: worst-case (mean $\pm 1\sigma$)',
        linewidth=1.5,
    )
    _axes3[1].fill_between(
        _K_range,
        np.maximum(_mc_min_mean - _mc_min_std, 0),
        _mc_min_mean + _mc_min_std,
        color='tab:orange',
        alpha=0.2,
    )
    _axes3[1].axhline(
        np.max(_FI_quad),
        color='gray',
        linestyle='--',
        alpha=0.6,
        label='Single max (quadrature)',
    )
    _axes3[1].set_xlabel('Number of interferometers K')
    _axes3[1].set_ylabel(
        r'Minimum Fisher Information $\min_{\delta\varphi}\,\mathcal{F}$'
    )
    _axes3[1].set_title('Scaling of Min(FI) with Number of Interferometers')
    _axes3[1].legend(fontsize=8)
    _axes3[1].grid(True, alpha=0.3)
    _axes3[1].set_xticks(_K_range)

    plt.tight_layout()
    plt.show()


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


@app.cell
def _(I0, V, fisher_info, np, plt):
    _t = np.linspace(0, 4 * np.pi, 2000)
    _amplitude = 0.8 * np.pi
    _delta_phi_t = _amplitude * np.sin(_t)

    _fig4, _axes4 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    _axes4[0].plot(_t, _delta_phi_t, 'k-', linewidth=1)
    _axes4[0].set_ylabel(r'$\delta\varphi(t)$ (rad)')
    _axes4[0].set_title('Surface Vibration (Phase Perturbation)')
    _axes4[0].grid(True, alpha=0.3)

    _FI_single_0 = fisher_info(_delta_phi_t, 0.0, I0, V)
    _axes4[1].plot(
        _t, _FI_single_0, color='tab:red', label=r'$K=1$, $\varphi_0=0$', linewidth=1
    )

    _FI_single_quad = fisher_info(_delta_phi_t, np.pi / 2, I0, V)
    _axes4[1].plot(
        _t,
        _FI_single_quad,
        color='tab:orange',
        label=r'$K=1$, $\varphi_0=\pi/2$',
        linewidth=1,
    )

    _offsets_3 = np.linspace(0, np.pi, 3, endpoint=False)
    _FI_multi = np.zeros_like(_delta_phi_t)
    for _p0 in _offsets_3:
        _FI_multi += fisher_info(_delta_phi_t, _p0, I0, V)
    _axes4[1].plot(
        _t, _FI_multi, color='tab:blue', label=r'$K=3$, uniform', linewidth=1.5
    )

    _axes4[1].set_ylabel(r'$\mathcal{F}(t)$')
    _axes4[1].set_title('Fisher Information Over Time')
    _axes4[1].legend(fontsize=8)
    _axes4[1].grid(True, alpha=0.3)

    for _idx, _p0 in enumerate(_offsets_3):
        _I_t = (I0 / 2) * (1 + np.cos(_p0 + _delta_phi_t))
        _axes4[2].plot(_t, _I_t, linewidth=1, label=rf'$\varphi_0 = {_p0:.2f}$')

    _axes4[2].set_xlabel('Time (a.u.)')
    _axes4[2].set_ylabel('Intensity')
    _axes4[2].set_title(r'Detected Intensities ($K=3$ Interferometers)')
    _axes4[2].legend(fontsize=8)
    _axes4[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


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


@app.cell
def _(mo):
    mo.md(r"""
    ## Multi-Color Interferometer Arrays

    When interferometers operate at **different wavelengths**, each maps the same physical
    displacement $d$ to a different phase:

    $$\varphi_k = \frac{4\pi}{\lambda_k}\,d + \varphi_0^{(k)}$$

    The Fisher Information on displacement $d$ for interferometer $k$ acquires a
    Jacobian factor from the wavelength-dependent phase sensitivity:

    $$\mathcal{F}_k(d) = \frac{I_0^2 V^2}{4\sigma^2}
    \sin^2\!\Bigl(\varphi_0^{(k)} + \frac{4\pi d}{\lambda_k}\Bigr)
    \cdot \Bigl(\frac{4\pi}{\lambda_k}\Bigr)^2$$

    Wavelength diversity provides two advantages over phase-offset diversity alone:

    1. **Incommensurate fringe periods** — the zeros of $\sin^2(\cdot)$ for different
       $\lambda$ do not coincide, so blind spots are suppressed more effectively than
       with a single color at different offsets.
    2. **Shorter wavelengths contribute more FI per channel** via the $(4\pi/\lambda)^2$
       prefactor.

    Below we compare how the mean Fisher Information (averaged over displacement) scales
    with array size $K$ for multi-color arrays vs. single-color phase-offset-diverse arrays.
    """)


@app.cell
def _(I0, V, np, plt):
    _N_d = 1000
    _d_grid = np.linspace(0, 1000, _N_d)  # displacement in nm, 0 to 2 µm

    _K_range = np.arange(1, 21)
    _N_mc = 1000
    _rng = np.random.default_rng(42)

    _lambda_ref = 635.0  # reference wavelength in nm

    # --- Multi-color: random wavelengths in [450, 780] nm + random phase offsets ---
    _mc_multi_min_mean = np.zeros(len(_K_range))
    _mc_multi_min_std = np.zeros(len(_K_range))
    _mc_multi_avg_mean = np.zeros(len(_K_range))
    _mc_multi_avg_std = np.zeros(len(_K_range))
    for _i, _K in enumerate(_K_range):
        _lambdas = _rng.uniform(450, 675, (_N_mc, _K))  # nm
        _phi0s = _rng.uniform(0, np.pi, (_N_mc, _K))
        # Shape: (N_mc, K, N_d)
        _phases = (
            _phi0s[:, :, np.newaxis]
            + 4
            * np.pi
            * _d_grid[np.newaxis, np.newaxis, :]
            / _lambdas[:, :, np.newaxis]
        )
        _jacobian_sq = (4 * np.pi / _lambdas) ** 2  # (N_mc, K)
        _FI_per_channel = (
            (I0 / 2) ** 2 * V**2 * np.sin(_phases) ** 2 * _jacobian_sq[:, :, np.newaxis]
        )
        _FI_total = np.sum(_FI_per_channel, axis=1)  # (N_mc, N_d)
        _min_over_d = np.min(_FI_total, axis=1)  # (N_mc,)
        _avg_over_d = np.mean(_FI_total, axis=1)  # (N_mc,)
        _mc_multi_min_mean[_i] = np.mean(_min_over_d)
        _mc_multi_min_std[_i] = np.std(_min_over_d)
        _mc_multi_avg_mean[_i] = np.mean(_avg_over_d)
        _mc_multi_avg_std[_i] = np.std(_avg_over_d)

    # --- Phase-offset only: all at 635 nm, random phase offsets ---
    _mc_phase_min_mean = np.zeros(len(_K_range))
    _mc_phase_min_std = np.zeros(len(_K_range))
    _mc_phase_avg_mean = np.zeros(len(_K_range))
    _mc_phase_avg_std = np.zeros(len(_K_range))
    for _i, _K in enumerate(_K_range):
        _phi0s = _rng.uniform(0, np.pi, (_N_mc, _K))
        _phases = (
            _phi0s[:, :, np.newaxis]
            + 4 * np.pi * _d_grid[np.newaxis, np.newaxis, :] / _lambda_ref
        )
        _jacobian_sq_ref = (4 * np.pi / _lambda_ref) ** 2
        _FI_per_channel = (I0 / 2) ** 2 * V**2 * np.sin(_phases) ** 2 * _jacobian_sq_ref
        _FI_total = np.sum(_FI_per_channel, axis=1)
        _min_over_d = np.min(_FI_total, axis=1)
        _avg_over_d = np.mean(_FI_total, axis=1)
        _mc_phase_min_mean[_i] = np.mean(_min_over_d)
        _mc_phase_min_std[_i] = np.std(_min_over_d)
        _mc_phase_avg_mean[_i] = np.mean(_avg_over_d)
        _mc_phase_avg_std[_i] = np.std(_avg_over_d)

    # --- Baselines at K=3: specific single-color arrays ---
    _baseline_lambdas = np.array([635.0, 675.0, 515.0])  # (3_wl,)
    _baseline_colors = ['tab:red', 'tab:purple', 'tab:green']
    _baseline_labels = ['635 nm', '675 nm', '515 nm']

    _phi0s = _rng.uniform(0, np.pi, (_N_mc, 3))  # (N_mc, 3_channels)
    # (3_wl, N_mc, 3_ch, N_d)
    _phases = (
        _phi0s[np.newaxis, :, :, np.newaxis]
        + 4
        * np.pi
        * _d_grid[np.newaxis, np.newaxis, np.newaxis, :]
        / _baseline_lambdas[:, np.newaxis, np.newaxis, np.newaxis]
    )
    _jac_sq = (4 * np.pi / _baseline_lambdas) ** 2  # (3_wl,)
    _FI_per_channel = (
        (I0 / 2) ** 2
        * V**2
        * np.sin(_phases) ** 2
        * _jac_sq[:, np.newaxis, np.newaxis, np.newaxis]
    )
    _FI_total = np.sum(_FI_per_channel, axis=2)  # (3_wl, N_mc, N_d)
    _FI_min_over_d = np.min(_FI_total, axis=2)  # (3_wl, N_mc)
    _baseline_min_means = np.mean(_FI_min_over_d, axis=1)  # (3_wl,)
    _FI_avg_over_d = np.mean(_FI_total, axis=2)  # (3_wl, N_mc)
    _baseline_avg_means = np.mean(_FI_avg_over_d, axis=1)  # (3_wl,)

    # --- Plot: side-by-side min vs mean FI ---
    _fig, (_ax_min, _ax_avg) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mean of Min FI (worst-case robustness)
    _ax_min.plot(
        _K_range,
        _mc_multi_min_mean,
        's-',
        color='tab:blue',
        linewidth=1.5,
        label=r'Multi-color ($\lambda \sim U[450,675]$ nm)',
    )
    _ax_min.fill_between(
        _K_range,
        _mc_multi_min_mean - _mc_multi_min_std,
        _mc_multi_min_mean + _mc_multi_min_std,
        color='tab:blue',
        alpha=0.2,
    )
    _ax_min.plot(
        _K_range,
        _mc_phase_min_mean,
        'o-',
        color='tab:orange',
        linewidth=1.5,
        label=r'Phase-offset only (635 nm)',
    )
    _ax_min.fill_between(
        _K_range,
        _mc_phase_min_mean - _mc_phase_min_std,
        _mc_phase_min_mean + _mc_phase_min_std,
        color='tab:orange',
        alpha=0.2,
    )
    for _bm, _bc, _bl in zip(
        _baseline_min_means, _baseline_colors, _baseline_labels, strict=False
    ):
        _ax_min.scatter(
            3,
            _bm,
            s=100,
            color=_bc,
            zorder=5,
            edgecolors='k',
            linewidths=0.8,
            label=rf'$K=3$ baseline ({_bl})',
        )
    _ax_min.set_xlabel('Number of interferometers $K$')
    _ax_min.set_ylabel(r'$\langle \min_d \mathcal{F}(d) \rangle$ (nm$^{-2}$)')
    _ax_min.set_title('Worst-Case FI (Mean of Min over $d$)')
    _ax_min.legend(fontsize=8)
    _ax_min.grid(True, alpha=0.3)
    _ax_min.set_xticks(_K_range)

    # Right: Mean of Mean FI (average sensitivity)
    _ax_avg.plot(
        _K_range,
        _mc_multi_avg_mean,
        's-',
        color='tab:blue',
        linewidth=1.5,
        label=r'Multi-color ($\lambda \sim U[450,675]$ nm)',
    )
    _ax_avg.fill_between(
        _K_range,
        _mc_multi_avg_mean - _mc_multi_avg_std,
        _mc_multi_avg_mean + _mc_multi_avg_std,
        color='tab:blue',
        alpha=0.2,
    )
    _ax_avg.plot(
        _K_range,
        _mc_phase_avg_mean,
        'o-',
        color='tab:orange',
        linewidth=1.5,
        label=r'Phase-offset only (635 nm)',
    )
    _ax_avg.fill_between(
        _K_range,
        _mc_phase_avg_mean - _mc_phase_avg_std,
        _mc_phase_avg_mean + _mc_phase_avg_std,
        color='tab:orange',
        alpha=0.2,
    )
    for _bm, _bc, _bl in zip(
        _baseline_avg_means, _baseline_colors, _baseline_labels, strict=False
    ):
        _ax_avg.scatter(
            3,
            _bm,
            s=100,
            color=_bc,
            zorder=5,
            edgecolors='k',
            linewidths=0.8,
            label=rf'$K=3$ baseline ({_bl})',
        )
    _ax_avg.set_xlabel('Number of interferometers $K$')
    _ax_avg.set_ylabel(r'$\langle \overline{\mathcal{F}}(d) \rangle$ (nm$^{-2}$)')
    _ax_avg.set_title('Average FI (Mean of Mean over $d$)')
    _ax_avg.legend(fontsize=8)
    _ax_avg.grid(True, alpha=0.3)
    _ax_avg.set_xticks(_K_range)

    plt.tight_layout()
    plt.show()


@app.cell
def _(mo):
    mo.md(r"""
    The two panels reveal a fundamental trade-off between wavelength diversity and
    worst-case robustness:

    **Left (Worst-Case FI):** Single-color arrays with diverse phase offsets
    outperform multi-color arrays, especially at small $K$. This is because all
    channels at the same wavelength oscillate at the **same spatial frequency** —
    their zeros are fixed relative to each other, so well-spread phase offsets
    guarantee at least one channel is always near quadrature. With different
    wavelengths, the fringe periods are **incommensurate**, and by Kronecker's
    theorem the joint phase trajectory is dense on the torus: there always exist
    displacements where all channels are simultaneously near their zeros, driving
    the worst-case FI down.

    **Right (Average FI):** Multi-color arrays achieve higher average FI because
    shorter wavelengths contribute more information per channel via the
    $(4\pi/\lambda)^2$ Jacobian factor. The harmonic mean of the wavelength
    distribution weights shorter $\lambda$ more heavily, boosting the multi-color
    curve above the single-color reference.

    The **baseline points** at $K=3$ confirm the $1/\lambda^2$ scaling: shorter
    wavelengths (515 nm) yield more FI per channel than longer ones (675 nm) in
    both metrics.
    """)


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


if __name__ == '__main__':
    app.run()
