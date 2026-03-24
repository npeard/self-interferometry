import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.collections import LineCollection

    return LineCollection, mo, np, plt, torch


@app.cell
def _():
    from self_interferometry.analysis.datasets import get_data_loaders
    from self_interferometry.analysis.generate_data import generate_synthetic_test_data
    from self_interferometry.analysis.lit_module import LitModule
    from self_interferometry.analysis.synthetic_lit_module import DEFAULT_WAVELENGTHS_NM

    return (
        DEFAULT_WAVELENGTHS_NM,
        LitModule,
        generate_synthetic_test_data,
        get_data_loaders,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Model Predictions Viewer

    Load a trained checkpoint and visualize predictions against target data.
    Supports both real HDF5 datasets and synthetic test data.
    """)
    return


@app.cell
def _():
    freespace_datapath = '/Users/nolanpeard/Documents/Projects/self-interferometry/self_interferometry/analysis/data/free-space-synchro_10k.h5'
    mmfiber_datapath = '/Users/nolanpeard/Documents/Projects/self-interferometry/self_interferometry/analysis/data/mmfiber-synchro_10k.h5'
    return (freespace_datapath,)


@app.cell
def _(freespace_datapath, mo):
    checkpoint_path = mo.ui.text(
        label='Checkpoint path',
        value='/Users/nolanpeard/Documents/Projects/self-interferometry/self_interferometry/analysis/models/checkpoints/1rjw9mid_epoch=199-val_total_unweighted_loss=0.0000.ckpt',
    )
    dataset_path = mo.ui.text(
        label="Dataset path (or 'synthetic')", value=freespace_datapath
    )
    batch_size = mo.ui.number(label='Batch size', value=64, start=1, stop=512, step=1)
    sample_index = mo.ui.slider(label='Sample index', start=0, stop=9, value=0)
    mo.vstack([checkpoint_path, dataset_path, batch_size, sample_index])
    return batch_size, checkpoint_path, dataset_path, sample_index


@app.cell
def _(LitModule, checkpoint_path, mo, torch):
    mo.stop(not checkpoint_path.value, mo.md('Enter a checkpoint path above.'))
    _map_location = 'cpu' if not torch.cuda.is_available() else None
    # Manual loading to strip torch.compile _orig_mod. key prefixes
    _ckpt = torch.load(
        checkpoint_path.value, weights_only=False, map_location=_map_location
    )
    ckpt_hparams = _ckpt.get('hyper_parameters', {})
    _state_dict = {
        k.replace('._orig_mod.', '.'): v for k, v in _ckpt['state_dict'].items()
    }
    model = LitModule(**ckpt_hparams)
    model.load_state_dict(_state_dict)
    in_channels = model.hparams['model_hparams']['in_channels']
    return ckpt_hparams, in_channels, model


@app.cell
def _(
    DEFAULT_WAVELENGTHS_NM,
    batch_size,
    ckpt_hparams,
    dataset_path,
    generate_synthetic_test_data,
    get_data_loaders,
    in_channels,
):
    if dataset_path.value == 'synthetic':
        _wavelengths_nm = ckpt_hparams.get(
            'wavelengths_nm', DEFAULT_WAVELENGTHS_NM[:in_channels]
        )
        _start_freq = ckpt_hparams.get('start_freq', 1.0)
        _end_freq = ckpt_hparams.get('end_freq', 1000.0)
        _max_displacement_um = ckpt_hparams.get('max_displacement_um', 5.0)
        test_loader = generate_synthetic_test_data(
            num_samples=batch_size.value * 4,
            wavelengths_nm=_wavelengths_nm,
            batch_size=batch_size.value,
            start_freq=_start_freq,
            end_freq=_end_freq,
            max_displacement_um=_max_displacement_um,
        )
    else:
        _, _, test_loader = get_data_loaders(
            dataset_path.value, batch_size=batch_size.value
        )
    return (test_loader,)


@app.cell
def _(model, test_loader, torch):
    # Run predictions on a single batch only
    model.eval()
    _batch = next(iter(test_loader))
    with torch.no_grad():
        _result = model.predict_step(_batch, 0)

    velocity_hat = _result[0].cpu().numpy()
    velocity_target = _result[1].cpu().numpy()
    displacement_hat = _result[2].cpu().numpy()
    displacement_target = _result[3].cpu().numpy()
    signals = _result[4].cpu().numpy()

    # Batch-level MSE
    _mse = torch.nn.MSELoss()
    batch_velocity_mse = _mse(
        torch.tensor(velocity_hat), torch.tensor(velocity_target)
    ).item()
    batch_displacement_mse = _mse(
        torch.tensor(displacement_hat), torch.tensor(displacement_target)
    ).item()
    return (
        batch_displacement_mse,
        batch_velocity_mse,
        displacement_hat,
        displacement_target,
        signals,
        velocity_hat,
        velocity_target,
    )


@app.cell
def _(np, torch):
    def compute_input_gradients(mdl, sigs):
        """Compute gradients of model output with respect to input signals.

        For each sample, computes d(output[t])/d(input) for all timesteps t.
        Returns the sum of absolute gradients across all output timesteps,
        showing which input points have the strongest overall influence.

        Args:
            mdl: The trained model
            sigs: Input signals tensor [batch_size, num_channels, signal_length]

        Returns:
            Gradients array [batch_size, num_channels, signal_length]
        """
        mdl.training = True
        signals_input = torch.tensor(
            sigs.detach().cpu().numpy(),
            dtype=torch.float32,
            requires_grad=True,
            device='cpu',
        )
        with torch.enable_grad():
            output = mdl(signals_input)
            grad_outputs = torch.ones_like(output)
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=signals_input,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )[0]
        return np.abs(gradients.detach().numpy())

    return (compute_input_gradients,)


@app.cell
def _(compute_input_gradients, model, signals, torch):
    _model_cpu = model.cpu()
    _signals_tensor = torch.tensor(signals, dtype=torch.float32, requires_grad=True)
    abs_gradients = compute_input_gradients(_model_cpu, _signals_tensor)

    _vmean = abs_gradients.mean()
    _vstd = abs_gradients.std()
    grad_vmin = 0
    grad_vmax = _vmean + 2 * _vstd
    return abs_gradients, grad_vmax, grad_vmin


@app.cell
def _(mo):
    mo.md(r"""
    ## Raw Data
    """)
    return


@app.cell
def _(np, plt, sample_index, signals):
    _i = sample_index.value
    _batch_size, _num_channels, _signal_length = signals.shape
    _hw_names = ['PD1 (RP1_CH2)', 'PD2 (RP2_CH1)', 'PD3 (RP2_CH2)']
    _channel_names = [
        _hw_names[_j] if _j < len(_hw_names) else f'CH{_j + 1}'
        for _j in range(_num_channels)
    ]

    _fig, _axs = plt.subplots(
        _num_channels, 1, figsize=(12, 3 * _num_channels), sharex=True
    )
    if _num_channels == 1:
        _axs = [_axs]
    for _j in range(_num_channels):
        _axs[_j].plot(np.arange(_signal_length), signals[_i, _j, :])
        _axs[_j].set_title(f'Input Signal - {_channel_names[_j]}')
        _axs[_j].set_ylabel('Amplitude')
        _axs[_j].grid(True, alpha=0.3)
    _axs[-1].set_xlabel('Sample')
    _fig.suptitle(f'Raw Input Signals - Sample {_i + 1}')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Predictions
    """)
    return


@app.cell
def _(
    displacement_hat,
    displacement_target,
    plt,
    sample_index,
    torch,
    velocity_hat,
    velocity_target,
):
    _i = sample_index.value
    _mse = torch.nn.MSELoss()
    _v_mse = _mse(
        torch.tensor(velocity_hat[_i]), torch.tensor(velocity_target[_i])
    ).item()
    _d_mse = _mse(
        torch.tensor(displacement_hat[_i]), torch.tensor(displacement_target[_i])
    ).item()

    _fig, _axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    _axs[0].plot(velocity_target[_i], label='Target Velocity', color='blue')
    _axs[0].plot(
        velocity_hat[_i], label='Predicted Velocity', color='red', linestyle='--'
    )
    _axs[0].set_title(f'Velocity (MSE: {_v_mse:.2e})')
    _axs[0].set_ylabel('Velocity (μm/s)')
    _axs[0].grid(True, alpha=0.3)
    _axs[0].legend(loc='upper right')

    _axs[1].plot(displacement_target[_i], label='Target Displacement', color='green')
    _axs[1].plot(
        displacement_hat[_i],
        label='Predicted Displacement',
        color='orange',
        linestyle='--',
    )
    _axs[1].set_title(f'Displacement (MSE: {_d_mse:.2e})')
    _axs[1].set_ylabel('Displacement (μm)')
    _axs[1].grid(True, alpha=0.3)
    _axs[1].legend(loc='upper right')

    _axs[-1].set_xlabel('Sample')
    _fig.suptitle(f'Predictions - Sample {_i + 1}')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Residuals
    """)
    return


@app.cell
def _(
    displacement_hat,
    displacement_target,
    plt,
    sample_index,
    torch,
    velocity_hat,
    velocity_target,
):
    _i = sample_index.value
    _mse = torch.nn.MSELoss()
    _v_mse = _mse(
        torch.tensor(velocity_hat[_i]), torch.tensor(velocity_target[_i])
    ).item()
    _d_mse = _mse(
        torch.tensor(displacement_hat[_i]), torch.tensor(displacement_target[_i])
    ).item()
    _v_residual = velocity_hat[_i] - velocity_target[_i]
    _d_residual = displacement_hat[_i] - displacement_target[_i]

    _fig, _axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    _axs[0].plot(_v_residual, label='Velocity Residual', color='purple')
    _axs[0].axhline(y=0, color='black', linestyle=':', alpha=0.5)
    _axs[0].set_title(f'Velocity Residual (MSE: {_v_mse:.2e})')
    _axs[0].set_ylabel('Residual (μm/s)')
    _axs[0].grid(True, alpha=0.3)
    _axs[0].legend(loc='upper right')

    _axs[1].plot(_d_residual, label='Displacement Residual', color='purple')
    _axs[1].axhline(y=0, color='black', linestyle=':', alpha=0.5)
    _axs[1].set_title(f'Displacement Residual (MSE: {_d_mse:.2e})')
    _axs[1].set_ylabel('Residual (μm)')
    _axs[1].grid(True, alpha=0.3)
    _axs[1].legend(loc='upper right')

    _axs[-1].set_xlabel('Sample')
    _fig.suptitle(f'Residuals - Sample {_i + 1}')
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Combined View
    """)
    return


@app.cell
def _(
    LineCollection,
    abs_gradients,
    batch_displacement_mse,
    batch_velocity_mse,
    displacement_hat,
    displacement_target,
    grad_vmax,
    grad_vmin,
    np,
    plt,
    sample_index,
    signals,
    torch,
    velocity_hat,
    velocity_target,
):
    _i = sample_index.value
    _batch_size, _num_channels, _signal_length = signals.shape
    _mse = torch.nn.MSELoss()
    _v_mse = _mse(
        torch.tensor(velocity_hat[_i]), torch.tensor(velocity_target[_i])
    ).item()
    _d_mse = _mse(
        torch.tensor(displacement_hat[_i]), torch.tensor(displacement_target[_i])
    ).item()
    _v_residual = velocity_hat[_i] - velocity_target[_i]
    _d_residual = displacement_hat[_i] - displacement_target[_i]

    _hw_names = ['PD1 (RP1_CH2)', 'PD2 (RP2_CH1)', 'PD3 (RP2_CH2)']
    _channel_names = [
        _hw_names[_j] if _j < len(_hw_names) else f'CH{_j + 1}'
        for _j in range(_num_channels)
    ]

    _fig, _axs = plt.subplots(4 + _num_channels, 1, figsize=(12, 12), sharex=True)

    # Velocity predictions
    _axs[0].plot(velocity_target[_i], label='Target Velocity', color='blue')
    _axs[0].plot(
        velocity_hat[_i], label='Predicted Velocity', color='red', linestyle='--'
    )
    _axs[0].set_title(f'Velocity (MSE: {_v_mse:.2e})')
    _axs[0].set_ylabel('Velocity (μm/s)')
    _axs[0].grid(True, alpha=0.3)
    _axs[0].legend(loc='upper right')

    # Velocity residuals
    _axs[1].plot(_v_residual, label='Velocity Residual', color='purple')
    _axs[1].axhline(y=0, color='black', linestyle=':', alpha=0.5)
    _axs[1].set_title(f'Velocity Residual (MSE: {_v_mse:.2e})')
    _axs[1].set_ylabel('Residual (μm/s)')
    _axs[1].grid(True, alpha=0.3)
    _axs[1].legend(loc='upper right')

    # Displacement predictions
    _axs[2].plot(displacement_target[_i], label='Target Displacement', color='green')
    _axs[2].plot(
        displacement_hat[_i],
        label='Predicted Displacement',
        color='orange',
        linestyle='--',
    )
    _axs[2].set_title(f'Displacement (MSE: {_d_mse:.2e})')
    _axs[2].set_ylabel('Displacement (μm)')
    _axs[2].grid(True, alpha=0.3)
    _axs[2].legend(loc='upper right')

    # Displacement residuals
    _axs[3].plot(_d_residual, label='Displacement Residual', color='purple')
    _axs[3].axhline(y=0, color='black', linestyle=':', alpha=0.5)
    _axs[3].set_title(f'Displacement Residual (MSE: {_d_mse:.2e})')
    _axs[3].set_ylabel('Residual (μm)')
    _axs[3].grid(True, alpha=0.3)
    _axs[3].legend(loc='upper right')

    # Signal channels colored by gradient magnitude
    _line = None
    for _j in range(_num_channels):
        _signal = signals[_i, _j, :]
        _grad_mag = abs_gradients[_i, _j, :]
        _x = np.arange(_signal_length)
        _points = np.array([_x, _signal]).T.reshape(-1, 1, 2)
        _segments = np.concatenate([_points[:-1], _points[1:]], axis=1)
        _colors = _grad_mag[:-1]

        _lc = LineCollection(
            _segments, cmap='hot', norm=plt.Normalize(vmin=grad_vmin, vmax=grad_vmax)
        )
        _lc.set_array(_colors)
        _lc.set_linewidth(2)
        _line = _axs[_j + 4].add_collection(_lc)
        _axs[_j + 4].set_xlim(0, _signal_length - 1)
        _axs[_j + 4].set_ylim(_signal.min() * 1.1, _signal.max() * 1.1)
        _axs[_j + 4].set_title(
            f'Input Signal - {_channel_names[_j]} (colored by |doutput/dinput|)'
        )
        _axs[_j + 4].set_ylabel('Amplitude')
        _axs[_j + 4].grid(True, alpha=0.3)

    _axs[-1].set_xlabel('Sample')
    _fig.suptitle(
        f'Sample {_i + 1}\n'
        f'Batch MSE: Velocity={batch_velocity_mse:.2e}, '
        f'Displacement={batch_displacement_mse:.2e}'
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    # Colorbar spanning signal subplots
    if _line is not None:
        _first_bbox = _axs[4].get_position()
        _last_bbox = _axs[-1].get_position()
        _cbar_ax = _fig.add_axes(
            [0.92, _last_bbox.y0, 0.02, _first_bbox.y1 - _last_bbox.y0]
        )
        _fig.colorbar(_line, cax=_cbar_ax, label='Gradient Magnitude |doutput/dinput|')

    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    *Gradient visualization shows |doutput/dinput| magnitude, indicating which input
    time points most strongly influence the model's predictions.*
    """)
    return


if __name__ == "__main__":
    app.run()
