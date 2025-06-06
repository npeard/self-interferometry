#!/usr/bin/env python3
"""Coil Driver - A class for calibrating and computing displacement and velocity
from voltage waveforms.

This module provides a class that encapsulates functionality for converting voltage
waveforms to displacement and velocity waveforms using calibration data.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.fft import fft, fftfreq, ifft

from self_interferometry.redpitaya.waveform import Waveform


@dataclass
class CalibrationParameters:
    """Calibration parameters for the coil driver.

    Attributes:
        f0: Resonant frequency in Hz
        Q: Quality factor
        k: Gain factor
        c: Phase offset
        speaker_part_number: Optional part number of the speaker
    """

    f0: float = 257.20857316296724
    Q: float = 15.804110908084784
    k: float = 33.42493417407945
    c: float = -3.208233068626455
    speaker_part_number: str | None = None


class CoilDriver:
    """A class for calibrating and computing displacement and velocity from voltage
    waveforms.

    This class provides methods for:
    - Converting voltage waveforms to displacement waveforms
    - Converting voltage waveforms to velocity waveforms
    - Calculating transfer functions for the coil driver
    """

    def __init__(self, calibration_params: CalibrationParameters | None = None):
        """Initialize the CoilDriver with calibration parameters.

        Args:
            calibration_params: Calibration parameters for the coil driver.
                                If None, default parameters are used.
        """
        if calibration_params is None:
            self.params = CalibrationParameters()
        else:
            self.params = calibration_params

    def sample(
        self,
        waveform: Waveform,
        randomize_phase_only: bool = False,
        random_single_tone: bool = False,
        normalize_gain: bool = False,
        test_mode: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a sample waveform using the Waveform generator, with optional gain
        equalization.

        When normalize_gain is True, the spectrum is pre-compensated by dividing by the
        amplitude transfer function, so that when the displacement_spectrum is computed
        via multiplication, the spectral components are the same as originally generated
        by Waveform.sample().

        Args:
            waveform: Waveform generator instance
            randomize_phase_only: If True, only randomize the phases while keeping the
                spectrum amplitudes the same
            random_single_tone: If True, generate a single tone at a randomly selected
                valid frequency
            normalize_gain: If True, pre-compensate the spectrum to normalize the gain
                across frequencies
            test_mode: If True, do not randomize the spectrum and phases. Used to
                repeatedly generate the same waveform during testing.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - Array of time points
                - Amplitude points in time domain
                - Spectrum (amplitude)
                - Spectral phases (phase)
        """
        # If normalize_gain is False, simply return the output of Waveform.sample()
        if not normalize_gain:
            return waveform.sample(
                randomize_phase_only=randomize_phase_only,
                random_single_tone=random_single_tone,
            )

        # Generate a sample from the waveform
        t, voltage, voltage_spectrum = waveform.sample(
            randomize_phase_only=randomize_phase_only,
            random_single_tone=random_single_tone,
            test_mode=test_mode,
        )

        # Calculate frequencies for the full spectrum
        sample_rate = 1 / (t[1] - t[0])
        n = len(t)
        freq = fftfreq(n, d=1 / sample_rate)

        # Get the complex transfer function
        transfer_function = self.get_transfer_function(freq)

        # Compute compensation for velocity transform (because velocity is the quantity
        # we care about, we'd like for its spectrum to be the same shape as the original
        # waveform (flat)
        velocity_transfer = transfer_function * 1j * 2 * np.pi * freq
        # We apply a scaling factor here so that we don't have to apply large amounts of
        # gain to the original spectrum. This is OK because ultimately we just want the
        # shape of the velocity spectrum to match the original waveform (flat) and we
        # don't care so much about the overall scaling as long as it is within the
        # hardware constraints and allows us to scan over multiple fringes
        scaling_factor = np.abs(velocity_transfer[freq == waveform.valid_freqs[0]])
        # The extra factor of 10 here is manually selected and seems to produce voltages
        # in a compatible range
        velocity_transfer /= scaling_factor * 10

        # Pre-compensate the spectrum by dividing by the complex transfer function
        nonzero_mask = velocity_transfer != 0
        normalized_spectrum = np.ones_like(voltage_spectrum)
        normalized_spectrum[nonzero_mask] = (
            voltage_spectrum[nonzero_mask] / velocity_transfer[nonzero_mask]
        )

        # Convert to time domain
        normalized_voltage = np.real(ifft(normalized_spectrum, norm='ortho'))
        normalized_voltage = np.fft.fftshift(normalized_voltage)

        # Recompute the spectrum and phase for use in other modules
        # First, shift back to match the original order
        y_unshifted = np.fft.ifftshift(normalized_voltage)

        # Compute the FFT to get the normalized spectrum
        normalized_spectrum = fft(y_unshifted, norm='ortho')

        return t, normalized_voltage, normalized_spectrum

    def _calculate_amplitude_transfer(self, f: np.ndarray) -> np.ndarray:
        """Calculate the amplitude transfer function for the coil driver.

        Args:
            f: Frequencies at which to calculate the transfer function (Hz)

        Returns:
            Amplitude transfer function (microns/V)
        """
        return (self.params.k * self.params.f0**2) / np.sqrt(
            (self.params.f0**2 - f**2) ** 2
            + self.params.f0**2 * f**2 / self.params.Q**2
        )

    def _calculate_phase_transfer(self, f: np.ndarray) -> np.ndarray:
        """Calculate the phase transfer function for the coil driver.

        Args:
            f: Frequencies at which to calculate the transfer function (Hz)

        Returns:
            Phase transfer function (radians)
        """
        return (
            np.arctan2(self.params.f0 / self.params.Q * f, f**2 - self.params.f0**2)
            + self.params.c
        )

    def get_transfer_function(
        self, freq: np.ndarray, max_freq: float | None = None
    ) -> np.ndarray:
        """Calculate the complex transfer function for the given frequencies.

        Args:
            freq: Frequency array (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included

        Returns:
            Complex transfer function as a numpy array
        """
        # Calculate the amplitude transfer function
        amplitude_transfer = self._calculate_amplitude_transfer(np.abs(freq))

        # Apply frequency limit if specified
        if max_freq is not None:
            amplitude_transfer = np.where(
                np.abs(freq) <= max_freq, amplitude_transfer, 0
            )

        # Calculate the phase transfer function
        # For negative frequencies, we need to conjugate the phase
        phase_transfer = np.where(
            freq < 0,
            np.exp(-1j * self._calculate_phase_transfer(-freq)),
            np.exp(1j * self._calculate_phase_transfer(freq)),
        )

        # Return the complex transfer function
        return amplitude_transfer * phase_transfer

    def get_displacement(
        self,
        voltage_waveform: np.ndarray,
        sample_rate: float,
        max_freq: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the displacement waveform from a voltage waveform using the
        calibration parameters.

        Args:
            voltage_waveform: Voltage waveform in time domain (V)
            sample_rate: Sample rate of the waveform (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included

        Returns:
            Tuple containing:
            - Displacement waveform in time domain (microns)
            - Displacement waveform in frequency domain
            - Frequency array (Hz)
        """
        # Calculate the spectrum of the voltage waveform
        voltage_spectrum = fft(voltage_waveform, norm='ortho')
        n = voltage_waveform.size
        sample_spacing = 1 / sample_rate
        freq = fftfreq(n, d=sample_spacing)  # units: cycles/s = Hz

        # Get the complex transfer function
        transfer_function = self.get_transfer_function(freq, max_freq)

        # Multiply by the transfer function
        displacement_spectrum = voltage_spectrum * transfer_function

        # Divide by 2pi to account for change of variables used when fitting the
        # transfer functions
        # Need to do this whenever we inverse FFT the spectrum
        displacement_spectrum /= 2 * np.pi
        # Multiply by 2 since we changed how spectra are generated (two-sided instead of
        # one-sided)
        displacement_spectrum *= 2

        # Convert back to time domain
        displacement_waveform = np.real(ifft(displacement_spectrum, norm='ortho'))

        return displacement_waveform, displacement_spectrum, freq

    def get_velocity(
        self,
        voltage_waveform: np.ndarray,
        sample_rate: float,
        max_freq: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the velocity waveform from a voltage waveform using the calibration
        parameters.

        Args:
            voltage_waveform: Voltage waveform in time domain (V)
            sample_rate: Sample rate of the waveform (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included

        Returns:
            Tuple containing:
            - Velocity waveform in time domain (microns/s)
            - Velocity waveform in frequency domain
            - Frequency array (Hz)
        """
        # Calculate the spectrum of the voltage waveform
        voltage_spectrum = fft(voltage_waveform, norm='ortho')
        n = voltage_waveform.size
        sample_spacing = 1 / sample_rate
        freq = fftfreq(n, d=sample_spacing)  # units: cycles/s = Hz

        # Get the complex transfer function
        transfer_function = self.get_transfer_function(freq, max_freq)

        # Multiply by the transfer function
        displacement_spectrum = voltage_spectrum * transfer_function

        # Divide by 2pi to account for change of variables used when fitting the
        # transfer functions
        # Need to do this whenever we inverse FFT the spectrum
        displacement_spectrum /= 2 * np.pi
        # Multiply by 2 since we changed how spectra are generated (two-sided instead of
        # one-sided)
        displacement_spectrum *= 2

        # Calculate velocity spectrum by multiplying displacement spectrum by j*omega
        velocity_spectrum = displacement_spectrum * 1j * 2 * np.pi * freq

        # Convert back to time domain
        velocity_waveform = np.real(ifft(velocity_spectrum, norm='ortho'))

        return velocity_waveform, velocity_spectrum, freq

    @staticmethod
    def integrate_velocity(
        velocity_waveform: np.ndarray | torch.Tensor, sample_rate: float = 125e6 / 256
    ) -> np.ndarray | torch.Tensor:
        """Integrate velocity waveform to get displacement waveform using
        cumulative integration.

        Compatible with both NumPy arrays and PyTorch tensors.

        Args:
            velocity_waveform: Velocity waveform (microns/s)
                              Can be a NumPy array or PyTorch tensor
                              For PyTorch tensors, supports batch dimensions [batch_size, channels, signal_length]
            sample_rate: Sample rate of the velocity waveform (Hz)
                        Default is Red Pitaya sample rate with decimation
            high_pass_freq: High-pass filter cutoff frequency (Hz) to remove DC drift
                           Set to 0 to disable high-pass filtering

        Returns:
            Displacement waveform (microns) in the same format as input
        """
        # Check if input is a PyTorch tensor
        # More robust check using module name instead of attribute
        is_torch = 'torch' in str(type(velocity_waveform).__module__)

        if is_torch:
            # Get the shape for proper reshaping after integration
            original_shape = velocity_waveform.shape

            # Handle batch dimensions if present
            if len(original_shape) > 1:
                batch_size = original_shape[0]
                signal_length = original_shape[-1]

                # Reshape for integration (flatten batch dimensions)
                velocity_flat = velocity_waveform.reshape(-1, signal_length)
                displacement_flat = torch.zeros_like(velocity_flat)

                # Time step
                dt = 1.0 / sample_rate

                # Perform integration for each sample in batch
                for i in range(batch_size):
                    # Cumulative sum for integration
                    displacement_flat[i] = torch.cumsum(velocity_flat[i], dim=0) * dt

                    # Shift displacement to start at zero
                    displacement_flat[i] = (
                        displacement_flat[i] - displacement_flat[i][0]
                    )

                # Reshape back to original shape
                displacement = displacement_flat.reshape(original_shape)

            else:
                # Single waveform case
                dt = 1.0 / sample_rate
                displacement = torch.cumsum(velocity_waveform, dim=0) * dt

                # Shift displacement to start at zero
                displacement = displacement - displacement[0]

            return displacement

        else:
            # NumPy implementation (original)
            # Time step
            dt = 1.0 / sample_rate

            # Simple cumulative integration (cumulative sum * dt)
            displacement = np.cumsum(velocity_waveform) * dt

            # Shift displacement to start at zero at the beginning of the trace
            return displacement - displacement[0]

    @staticmethod
    def derivative_displacement(
        displacement_waveform: np.ndarray, sample_rate: float
    ) -> np.ndarray:
        """Calculate the time derivative of a displacement waveform to get velocity.

        This method uses central differences to compute the derivative.

        Args:
            displacement_waveform: Displacement waveform (microns)
            sample_rate: Sample rate of the displacement waveform (Hz)

        Returns:
            Velocity waveform (microns/s)
        """
        # Time step
        dt = 1.0 / sample_rate

        # Use central differences for better accuracy
        # For the first point, use forward difference
        # For the last point, use backward difference
        # For all other points, use central difference
        velocity = np.zeros_like(displacement_waveform)

        # First point (forward difference)
        velocity[0] = (displacement_waveform[1] - displacement_waveform[0]) / dt

        # Middle points (central difference)
        velocity[1:-1] = (displacement_waveform[2:] - displacement_waveform[:-2]) / (
            2 * dt
        )

        # Last point (backward difference)
        velocity[-1] = (displacement_waveform[-1] - displacement_waveform[-2]) / dt

        return velocity

    def set_calibration_parameters(self, params: CalibrationParameters):
        """Update the calibration parameters.

        Args:
            params: New calibration parameters
        """
        self.params = params

    def get_calibration_parameters(self) -> CalibrationParameters:
        """Get the current calibration parameters.

        Returns:
            Current calibration parameters
        """
        return self.params

    def save_calibration_to_dict(self) -> dict:
        """Save the calibration parameters to a dictionary.

        Returns:
            Dictionary containing the calibration parameters
        """
        return {
            'f0': self.params.f0,
            'Q': self.params.Q,
            'k': self.params.k,
            'c': self.params.c,
            'speaker_part_number': self.params.speaker_part_number,
        }

    @classmethod
    def from_dict(cls, params_dict: dict) -> 'CoilDriver':
        """Create a CoilDriver from a dictionary of parameters.

        Args:
            params_dict: Dictionary containing the calibration parameters

        Returns:
            New CoilDriver instance with the specified parameters
        """
        params = CalibrationParameters(
            f0=params_dict.get('f0', 257.20857316296724),
            Q=params_dict.get('Q', 15.804110908084784),
            k=params_dict.get('k', 33.42493417407945),
            c=params_dict.get('c', -3.208233068626455),
            speaker_part_number=params_dict.get('speaker_part_number'),
        )
        return cls(params)
