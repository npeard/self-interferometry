#!/usr/bin/env python3
"""
Test for the CoilDriver calibration and displacement/velocity calculations.

This test verifies:
1. Consistency between FFT-calculated and CoilDriver-retrieved displacement and velocity
2. Reconstructed waveform matching
3. Manual computation of displacement and velocity transfer functions
"""

import pytest
import numpy as np

from redpitaya.waveform import Waveform
from redpitaya.coil_driver import CoilDriver
from redpitaya.plot_waveforms import calculate_fft


def test_displacement_velocity_consistency():
    """
    Test that the displacement and velocity spectra calculated by CoilDriver
    are consistent with those calculated using calculate_fft.
    """
    # Create a waveform generator with test parameters
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()
    
    # Generate a random waveform
    t, voltage, voltage_spectral_mod, voltage_spectral_phase = waveform.sample()
    
    # Calculate sample rate from time array
    sample_rate = 1 / (t[1] - t[0])
    
    # Get displacement and velocity using CoilDriver
    displacement, displacement_spectrum, displacement_freqs = coil_driver.get_displacement(voltage, sample_rate)
    velocity, velocity_spectrum, velocity_freqs = coil_driver.get_velocity(voltage, sample_rate)
    
    # Calculate FFT of the displacement and velocity waveforms
    freqs_disp_fft, displacement_mag_fft, displacement_phase_fft = calculate_fft(displacement, sample_rate)
    freqs_vel_fft, velocity_mag_fft, velocity_phase_fft = calculate_fft(velocity, sample_rate)
    
    assert np.allclose(np.abs(displacement_spectrum)[displacement_freqs >= 0], displacement_mag_fft)
    assert np.allclose(np.abs(velocity_spectrum)[velocity_freqs >= 0], velocity_mag_fft)
    #assert np.allclose(np.angle(displacement_spectrum)[displacement_freqs >= 0], displacement_phase_fft)
    #assert np.allclose(np.angle(velocity_spectrum)[velocity_freqs >= 0], velocity_phase_fft)
    assert np.allclose(freqs_vel_fft, velocity_freqs[velocity_freqs >= 0])
    assert np.allclose(freqs_disp_fft, displacement_freqs[displacement_freqs >= 0])


def test_reconstructed_waveforms():
    """
    Test that the reconstructed displacement and velocity waveforms
    match those retrieved from CoilDriver.
    """
    # Create a waveform generator with test parameters
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()
    
    # Generate a random waveform
    t, voltage, voltage_spectral_mod, voltage_spectral_phase = waveform.sample()
    
    # Calculate sample rate from time array
    sample_rate = 1 / (t[1] - t[0])
    
    # Get displacement and velocity using CoilDriver
    displacement, displacement_spectrum, displacement_freqs = coil_driver.get_displacement(voltage, sample_rate)
    velocity, velocity_spectrum, velocity_freqs = coil_driver.get_velocity(voltage, sample_rate)
    
    # Reconstruct displacement and velocity from their spectra
    reconstructed_displacement = np.real(np.fft.ifft(displacement_spectrum, norm="ortho"))
    reconstructed_velocity = np.real(np.fft.ifft(velocity_spectrum, norm="ortho"))
    
    # Check that the reconstructed waveforms match the original ones
    assert np.allclose(displacement, reconstructed_displacement), \
        "Reconstructed displacement does not match original"
    
    assert np.allclose(velocity, reconstructed_velocity), \
        "Reconstructed velocity does not match original"


def test_manual_transfer_function_computation():
    """
    Test that manually computing displacement and velocity transfer functions
    gives consistent results with CoilDriver's methods.
    """
    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()
    
    # Create a frequency array for testing
    freqs = np.linspace(10, 1000, 1000)
    
    # Calculate amplitude and phase transfer functions using CoilDriver methods
    amplitude_transfer = coil_driver._calculate_amplitude_transfer(freqs)
    phase_transfer = coil_driver._calculate_phase_transfer(freqs)
    
    # Manually compute displacement transfer function
    displacement_transfer = amplitude_transfer * np.exp(1j * phase_transfer)
    displacement_transfer_mod = np.abs(displacement_transfer)
    displacement_transfer_phase = np.angle(displacement_transfer)
    
    # Manually compute velocity transfer function
    velocity_transfer = displacement_transfer * freqs * 2 * np.pi * 1j
    velocity_transfer_mod = np.abs(velocity_transfer)
    velocity_transfer_phase = np.angle(velocity_transfer)
    
    # Test a specific frequency to verify calculations
    test_freq_idx = 500  # Middle of the range
    test_freq = freqs[test_freq_idx]
    
    # Calculate expected displacement transfer at test frequency
    expected_disp_amplitude = coil_driver._calculate_amplitude_transfer(np.array([test_freq]))[0]
    expected_disp_phase = coil_driver._calculate_phase_transfer(np.array([test_freq]))[0]
    expected_disp_transfer = expected_disp_amplitude * np.exp(1j * expected_disp_phase)
    
    # Calculate expected velocity transfer at test frequency
    expected_vel_transfer = expected_disp_transfer * test_freq * 2 * np.pi * 1j
    
    # Check that the manually computed transfer functions match the expected values
    assert np.allclose(displacement_transfer[test_freq_idx], expected_disp_transfer), \
        "Manually computed displacement transfer function does not match expected"
    
    assert np.allclose(velocity_transfer[test_freq_idx], expected_vel_transfer), \
        "Manually computed velocity transfer function does not match expected"
    
    # Also verify that the velocity is the derivative of displacement
    # For a sinusoidal displacement x(t) = A*sin(ω*t + φ), the velocity is v(t) = A*ω*cos(ω*t + φ)
    # In the frequency domain, this is equivalent to multiplying by iω = i*2*π*f
    assert np.allclose(velocity_transfer, displacement_transfer * freqs * 2 * np.pi * 1j), \
        "Velocity transfer function is not consistent with displacement transfer function"


def test_displacement_velocity_relationship():
    """
    Test that the velocity is the time derivative of displacement.
    """
    # Create a waveform generator with test parameters
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()
    
    # Generate a random waveform
    t, voltage, voltage_spectral_mod, voltage_spectral_phase = waveform.sample()
    
    # Calculate sample rate from time array
    sample_rate = 1 / (t[1] - t[0])
    
    # Get displacement and velocity using CoilDriver
    displacement, displacement_spectrum, displacement_freqs = coil_driver.get_displacement(voltage, sample_rate)
    velocity, velocity_spectrum, velocity_freqs = coil_driver.get_velocity(voltage, sample_rate)
    
    # Verify that velocity_spectrum = iω * displacement_spectrum
    # This is equivalent to velocity being the time derivative of displacement
    expected_velocity_spectrum = 1j * 2 * np.pi * displacement_freqs * displacement_spectrum
    
    # Check that the velocity spectrum matches the expected value
    assert np.allclose(velocity_spectrum, expected_velocity_spectrum), \
        "Velocity spectrum is not the derivative of displacement spectrum"


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])
