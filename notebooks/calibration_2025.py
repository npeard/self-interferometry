import marimo

__generated_with = '0.20.4'
app = marimo.App()


@app.cell
def _():
    from dataclasses import dataclass

    import lmfit
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from uncertainties import ufloat
    from uncertainties import unumpy as unp

    return (
        FuncFormatter,
        MultipleLocator,
        dataclass,
        lmfit,
        mo,
        mpl,
        np,
        plt,
        ufloat,
        unp,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Voice Coil Calibration

    Joint fit of amplitude and phase frequency response data to a
    second-order damped harmonic oscillator model.
    """)


@app.cell
def _(np):
    frequency = np.array(
        [
            1,
            2,
            5,
            10,
            20,
            50,
            80,
            90,
            100,
            110,
            130,
            150,
            175,
            200,
            210,
            220,
            230,
            240,
            275,
            300,
            310,
            320,
            330,
            350,
            400,
            500,
            1000,
            846,
            797,
            703,
            567,
        ]
    )

    amplitude = np.array(
        [
            17.36478205,
            15.75459508,
            16.49235272,
            16.62064382,
            15.45089948,
            15.66495982,
            16.50981465,
            16.51475312,
            17.6839671,
            17.33336867,
            19.06888131,
            22.31569461,
            27.272504,
            35.4933552,
            40.96141627,
            51.80674343,
            66.47242315,
            96.30803206,
            102.4183677,
            41.78399195,
            32.02110103,
            26.45377421,
            21.65020221,
            14.66699953,
            12.82340615,
            5.895646196,
            1.236,
            1.668016194,
            2.090097403,
            2.94045677,
            4.63963964,
        ]
    )

    amplitude_error = np.array(
        [
            0.9011097879,
            1.545,
            0.6680331672,
            1.418199416,
            0.4994050903,
            1.545,
            0.7139004612,
            1.321387371,
            0.5931253222,
            1.04067386,
            0.7007313871,
            1.180013241,
            0.125427047,
            1.296054911,
            0.6615122291,
            1.808620561,
            0.2712039057,
            3.429038981,
            3.282011781,
            1.07205799,
            0.413200227,
            1.296054911,
            0.4372347327,
            1.03,
            0.3838222201,
            0.6480274557,
            1.340117371,
            1.340117371,
            1.340117371,
            1.340117371,
            1.340117371,
        ]
    )

    phase = np.array(
        [
            -0.04775220833,
            -0.04775220833,
            -0.04775220833,
            -0.04775220833,
            -0.04775220833,
            -0.04775220833,
            -0.04775220833,
            -0.0552920307,
            -0.06283185307,
            -0.06408849013,
            0.03015928947,
            -0.08168140899,
            -0.1193805208,
            -0.163362818,
            -0.1985486557,
            -0.24392582,
            -0.3387196084,
            -0.6142492222,
            -2.876137499,
            -3.045774078,
            -3.1033934,
            -3.095790746,
            -3.093663259,
            -3.051592307,
            -3.153430175,
            -3.20624663,
            -3.300305914,
            -3.382025274,
            -3.281376122,
            -3.191466882,
            -3.182679345,
        ]
    )

    # Based on estimated 10us error in estimation of time shifts
    phase_error = 10 ** (-8) * 2 * np.pi * frequency
    return amplitude, amplitude_error, frequency, phase, phase_error


@app.cell
def _(mpl, np):
    def apply_pub_style():
        """Set matplotlib rcParams for a clean, publication-ready style."""
        mpl.rcParams.update(
            {
                'figure.dpi': 150,
                'savefig.dpi': 200,
                'font.size': 10,
                'axes.labelsize': 10,
                'axes.titlesize': 11,
                'legend.fontsize': 9,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'figure.constrained_layout.use': True,
            }
        )

    def pi_over_two_formatter(x, pos):
        """Format axis ticks as multiples of pi/2."""
        k = x / (np.pi / 2)
        n = int(np.round(k))

        if not np.isclose(k, n):
            return ''
        if n == 0:
            return r'$0$'

        sign = '-' if n < 0 else ''
        a = abs(n)

        if a % 2 == 0:
            m = a // 2
            if m == 1:
                return rf'${sign}\pi$'
            return rf'${sign}{m}\pi$'
        if a == 1:
            return rf'${sign}\pi/2$'
        return rf'${sign}{a}\pi/2$'

    return apply_pub_style, pi_over_two_formatter


@app.cell
def _(dataclass, lmfit, np, ufloat):
    @dataclass
    class CalibrationParams:
        """Calibration parameters for the coil driver.

        Attributes:
            f0: Resonant frequency in Hz
            Q: Quality factor, equivalent to 1/(2*zeta_m)
            k: Gain factor, equivalent to alpha/k_m
            c: Phase offset in rad
        """

        f0: ufloat
        Q: ufloat
        k: ufloat
        c: ufloat

    def magnitude_func(f: np.ndarray, f0: float, Q: float, k: float) -> np.ndarray:
        return (k * f0**2) / np.sqrt((f0**2 - f**2) ** 2 + f0**2 * f**2 / Q**2)

    def phase_func(f: np.ndarray, f0: float, Q: float, c: float) -> np.ndarray:
        return np.arctan2(f0 / Q * f, f**2 - f0**2) + c

    def mag_phase_funcs(
        f: np.ndarray, f0: float, Q: float, k: float, c: float
    ) -> np.ndarray:
        """Joint magnitude and phase model for simultaneous fitting.

        Args:
            f: Frequency array in Hz
            f0: Resonant frequency in Hz
            Q: Quality factor, equivalent to 1/(2*zeta_m)
            k: Gain factor, equivalent to alpha/k_m
            c: Phase offset in rad
        """
        mag = magnitude_func(f, f0, Q, k)
        ph = phase_func(f, f0, Q, c)
        return np.append(mag, ph)

    def fit_data(
        frequency_hz: np.ndarray,
        amplitude: np.ndarray,
        amplitude_error: np.ndarray,
        phase_rad: np.ndarray,
        phase_error: np.ndarray,
    ) -> CalibrationParams:
        """Perform weighted least-squares joint fit of amplitude and phase."""
        weights = np.append(1 / amplitude_error, 1 / phase_error)
        model = lmfit.Model(mag_phase_funcs, weights=weights, scale_covar=False)
        params = model.make_params(f0=1, Q=dict(value=1, min=0), k=1, c=0)
        y_tot = np.append(amplitude, phase_rad)
        result = model.fit(y_tot, params, f=frequency_hz)
        return CalibrationParams(
            f0=result.uvars['f0'],
            Q=result.uvars['Q'],
            k=result.uvars['k'],
            c=result.uvars['c'],
        )

    return fit_data, magnitude_func, phase_func


@app.cell
def _(
    FuncFormatter,
    MultipleLocator,
    amplitude,
    amplitude_error,
    apply_pub_style,
    frequency,
    np,
    phase,
    phase_error,
    pi_over_two_formatter,
    plt,
):
    apply_pub_style()

    fig_data = plt.figure(figsize=(6.4, 4.8))
    gs = fig_data.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05)

    ax0 = fig_data.add_subplot(gs[0, 0])
    ax1 = fig_data.add_subplot(gs[1, 0], sharex=ax0)

    ax0.errorbar(
        frequency,
        amplitude,
        yerr=amplitude_error,
        linestyle='none',
        marker='o',
        markersize=3,
    )
    ax0.set_ylabel(r'$\left| H^{vc}_{x} \right| \; (\mu\mathrm{m}/\mathrm{V})$')
    ax0.tick_params(labelbottom=False)

    ax1.errorbar(
        frequency, phase, yerr=phase_error, linestyle='none', marker='o', markersize=3
    )
    ax1.set_ylabel(r'$\phi^{vc}_{x}$ (rad)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
    ax1.yaxis.set_major_formatter(FuncFormatter(pi_over_two_formatter))

    fig_data.suptitle('Voice Coil Calibration Data')
    fig_data


@app.cell
def _(
    FuncFormatter,
    MultipleLocator,
    amplitude,
    amplitude_error,
    apply_pub_style,
    fit_data,
    frequency,
    magnitude_func,
    np,
    phase,
    phase_error,
    phase_func,
    pi_over_two_formatter,
    plt,
    unp,
):
    params = fit_data(frequency, amplitude, amplitude_error, phase, phase_error)

    apply_pub_style()
    fig_fit = plt.figure(figsize=(6.4, 4.8))
    gs_fit = fig_fit.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05)

    ax_mag = fig_fit.add_subplot(gs_fit[0, 0])
    ax_ph = fig_fit.add_subplot(gs_fit[1, 0], sharex=ax_mag)

    fit_freq = np.geomspace(frequency.min(), frequency.max(), 1000)

    # Amplitude
    ax_mag.errorbar(
        frequency,
        amplitude,
        yerr=amplitude_error,
        marker='o',
        ms=3,
        capsize=2,
        linestyle='none',
    )
    ax_mag.plot(
        fit_freq,
        magnitude_func(
            fit_freq,
            unp.nominal_values(params.f0),
            unp.nominal_values(params.Q),
            unp.nominal_values(params.k),
        ),
    )
    ax_mag.set_ylabel(r'$\left| H^{vc}_{x} \right|$ ($\mu$m/V)')
    ax_mag.tick_params(labelbottom=False)
    ax_mag.loglog()
    ax_mag.text(0.03, 0.95, '(a)', transform=ax_mag.transAxes, ha='left', va='top')
    eq_mag = r'$\left| H^{vc}_{x} (f)\right| = \frac{K f_0^2}{\sqrt{(f_0^2 - f^2)^2 + \frac{f_0^2}{Q^2}f^2}}$'
    ax_mag.text(
        3,
        8,
        eq_mag,
        transform=ax_mag.transData,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', alpha=0.9),
    )

    # Phase
    ax_ph.errorbar(
        frequency,
        phase,
        yerr=phase_error,
        marker='o',
        ms=3,
        capsize=2,
        linestyle='none',
    )
    ax_ph.plot(
        fit_freq,
        phase_func(
            fit_freq,
            unp.nominal_values(params.f0),
            unp.nominal_values(params.Q),
            unp.nominal_values(params.c),
        ),
    )
    ax_ph.set_ylabel(r'$\phi^{vc}_{x}$ (rad)')
    ax_ph.set_xlabel('Frequency (Hz)')
    ax_ph.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))
    ax_ph.yaxis.set_major_formatter(FuncFormatter(pi_over_two_formatter))
    ax_ph.text(0.03, 0.22, '(b)', transform=ax_ph.transAxes, ha='left', va='top')
    eq_ph = r'$\phi^{vc}_{x}(f) = \mathrm{atan2} \left( \frac{\frac{f_0}{Q} f}{f^2 - f_0^2} \right) + c$'
    ax_ph.text(
        3,
        -1.5,
        eq_ph,
        transform=ax_ph.transData,
        ha='left',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', alpha=0.9),
    )

    fig_fit.suptitle(
        'Voice Coil Transfer Function Joint Fit of Amplitude and Phase', y=1.05
    )
    fig_fit
    return (params,)


@app.cell
def _(mo, params):
    fspec = '.1uSL'
    mo.md(
        f"""
        ## Fitted Parameters

        | Parameter | Value |
        |-----------|-------|
        | $f_0$ (Hz) | {params.f0:{fspec}} |
        | $Q = (2\\,\\zeta_{{\\mathrm{{m}}}})^{{-1}}$ | {params.Q:{fspec}} |
        | $K = \\alpha/k_{{\\mathrm{{m}}}}$ | {params.k:{fspec}} |
        | $c$ (rad) | {params.c:{fspec}} |
        """
    )


if __name__ == '__main__':
    app.run()
