import marimo

__generated_with = '0.20.4'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Fabry-Perot Cavity Reflectance: Fresnel Coefficient Model

        We model a two-mirror cavity where:
        - **R1** (front mirror reflectance) is scanned from 0 to 1
        - **R2** (back mirror reflectance) is fixed at 1
        - **φ** (round-trip phase from back-surface vibrations) varies from −π to π

        The reflected electric field from the cavity (Fresnel coefficients) is:

        $$E_r = r_1 + \frac{t_1^2 \, r_2 \, e^{i\varphi}}{1 - r_1 \, r_2 \, e^{i\varphi}}$$

        where $r_i = \sqrt{R_i}$, $t_1^2 = 1 - R_1$, and the detected intensity is $I = |E_r|^2$.
        """
    )
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    # Parameters
    N_R1 = 500
    N_phi = 500
    R2 = 1.0

    R1_vals = np.linspace(
        0, 1, N_R1, endpoint=False
    )  # exclude R1=1 to avoid singularity
    phi_vals = np.linspace(-np.pi, np.pi, N_phi)

    R1_grid, phi_grid = np.meshgrid(R1_vals, phi_vals, indexing='ij')

    # Fresnel amplitude coefficients
    r1 = np.sqrt(R1_grid)
    t1_sq = 1 - R1_grid  # t1^2 = 1 - R1 (intensity transmission)
    r2 = np.sqrt(R2)

    # Reflected field from cavity
    round_trip = r2 * np.exp(1j * phi_grid)
    E_ref = r1 + t1_sq * round_trip / (1 - r1 * round_trip)

    # Detected intensity (photodiode signal)
    I_detected = np.abs(E_ref) ** 2

    # Fringe slope: dI/dphi
    fringe_slope = np.gradient(I_detected, phi_vals, axis=1)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Detected intensity
    im0 = axes[0].imshow(
        I_detected.T,
        extent=[R1_vals[0], R1_vals[-1], -np.pi, np.pi],
        aspect='auto',
        origin='lower',
        cmap='inferno',
    )
    axes[0].set_xlabel('R1 (front mirror reflectance)')
    axes[0].set_ylabel('Phase φ (rad)')
    axes[0].set_title('Detected Intensity  $|E_r|^2$')
    fig.colorbar(im0, ax=axes[0], label='Intensity (a.u.)')

    # Fringe slope
    im1 = axes[1].imshow(
        np.log(np.abs(fringe_slope) + 1e-10).T,
        extent=[R1_vals[0], R1_vals[-1], -np.pi, np.pi],
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
    )
    axes[1].set_xlabel('R1 (front mirror reflectance)')
    axes[1].set_ylabel('Phase φ (rad)')
    axes[1].set_title(r'Fringe Slope  $\partial I / \partial \varphi$')
    fig.colorbar(im1, ax=axes[1], label='dI/dφ')

    plt.tight_layout()
    plt.show()


@app.cell
def _(mo):
    mo.md(r"""
    **Left:** Photodiode intensity as a function of front-mirror reflectance R1 and round-trip phase φ (R2 = {R2}).

    **Right:** Fringe slope (dI/dφ), showing how sensitive the detected signal is to phase changes at each operating point. Higher absolute slope = greater displacement sensitivity.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ## Key Observations

    - At **R1 ≈ 0**, the cavity has no front mirror — light passes through and reflects off R2 with no interference. The fringe slope is modest.
    - As **R1 increases**, the cavity finesse grows and fringes sharpen, increasing the peak slope.
    - Near **R1 → 1**, the cavity becomes highly resonant — extremely sharp fringes with very high slope at resonance, but the signal is very sensitive to the operating point.
    - The **optimal R1** for displacement sensing balances fringe sharpness against dynamic range.
    """)


if __name__ == '__main__':
    app.run()
