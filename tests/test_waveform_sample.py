#!/usr/bin/env python3
"""
Test for the Waveform.sample() method.

This test compares the spectrum values returned by the Waveform.sample() method
with those calculated manually using the calculate_fft function.
"""

import pytest
import numpy as np

from redpitaya.waveform import Waveform
from tests.plot_waveforms import calculate_fft


def test_waveform_sample_spectrum_consistency():
    """
    Test that the spectrum values returned by Waveform.sample() are consistent
    with those calculated by calculate_fft().
    
    This test:
    1. Creates a Waveform instance
    2. Calls sample() multiple times (at least 10)
    3. For each sample, compares the spectrum values returned by sample()
       with those calculated by calculate_fft()
    """
    # Create a waveform generator with test parameters
    start_freq = 10
    end_freq = 1000
    waveform = Waveform(
        start_freq=start_freq,
        end_freq=end_freq,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Run the test multiple times with different random waveforms
    num_tests = 10
    for i in range(num_tests):
        # Update frequency range slightly for each test to get different waveforms
        test_start_freq = start_freq + i
        test_end_freq = end_freq + i * 10
        waveform.set_frequency_range(test_start_freq, test_end_freq)
        
        # Generate a random waveform
        t, voltage, voltage_spectral_mod, voltage_spectral_phase = waveform.sample()
        
        # Calculate sample rate from time array
        sample_rate = 1 / (t[1] - t[0])
        
        # Calculate FFT of the voltage waveform using the calculate_fft function
        freqs_fft, voltage_mag_fft, voltage_phase_fft = calculate_fft(voltage, sample_rate)
        
        assert np.allclose(voltage_spectral_mod, voltage_mag_fft, rtol=0.1), \
            f"Test {i+1}: Spectrum mismatch"
        assert np.allclose(voltage_spectral_phase, voltage_phase_fft, rtol=0.1), \
            f"Test {i+1}: Phase mismatch"
            
        print(f"Completed test iteration {i+1}/{num_tests}")


def test_waveform_sample_reproducibility():
    """
    Test that consecutive calls to sample() with the same parameters 
    produce different random waveforms as expected.
    """
    # Create a waveform generator
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Generate two consecutive waveforms
    _, voltage1, _, _ = waveform.sample()
    _, voltage2, _, _ = waveform.sample()
    
    # They should be different (random generation)
    assert not np.allclose(voltage1, voltage2), "Consecutive samples should be different"


def test_waveform_frequency_range():
    """
    Test that the waveform spectrum is confined to the specified frequency range.
    """
    # Create a waveform generator with narrow frequency range
    start_freq = 100
    end_freq = 200
    waveform = Waveform(
        start_freq=start_freq,
        end_freq=end_freq,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Generate a waveform
    _, voltage, voltage_spectral_mod, _ = waveform.sample()
    
    # Calculate sample rate from time array
    sample_rate = 1 / (waveform.t[1] - waveform.t[0])
    
    # Calculate FFT of the voltage waveform using the calculate_fft function
    freqs_fft, voltage_mag, _ = calculate_fft(voltage, sample_rate)
    
    # Check that significant spectral content is within the specified range
    # (allowing for some leakage)
    threshold = 0.1 * np.max(voltage_mag)
    significant_freqs = freqs_fft[voltage_mag > threshold]
    
    # Check lower bound (with some tolerance)
    assert np.min(significant_freqs) >= start_freq * 0.9, \
        f"Significant frequency content found below start_freq: {np.min(significant_freqs)} < {start_freq}"
    
    # Check upper bound (with some tolerance)
    assert np.max(significant_freqs) <= end_freq * 1.1, \
        f"Significant frequency content found above end_freq: {np.max(significant_freqs)} > {end_freq}"


def test_waveform_output_range():
    """
    Test that the waveform output is properly normalized to the [-1, 1] range.
    """
    # Create a waveform generator
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Generate multiple waveforms and check their ranges
    for i in range(5):
        _, voltage, _, _ = waveform.sample()
        assert np.max(np.abs(voltage)) <= 1.0, f"Test {i+1}: Voltage exceeds normalized range [-1, 1]"


def test_waveform_length_consistency():
    """
    Test that the time and voltage arrays have consistent lengths.
    """
    # Create a waveform generator
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Generate multiple waveforms and check their lengths
    for i in range(5):
        t, voltage, voltage_spectral_mod, voltage_spectral_phase = waveform.sample()
        assert len(t) == len(voltage), f"Test {i+1}: Time and voltage arrays have different lengths"
        assert len(voltage_spectral_mod) == len(voltage_spectral_phase), \
            f"Test {i+1}: Spectral modulus and phase arrays have different lengths"


def test_waveform_reconstruction():
    """
    Test that a waveform can be reconstructed from its spectral components.
    """
    # Create a waveform generator
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Generate a waveform
    t, voltage, voltage_spectral_mod, voltage_spectral_phase = waveform.sample()
    
    # Construct complex spectrum
    complex_spectrum = voltage_spectral_mod * np.exp(1j * voltage_spectral_phase)
    
    # Complete the spectrum for ifft (make it symmetric)
    full_spectrum = np.zeros(len(voltage), dtype=complex)
    half_len = len(complex_spectrum)
    full_spectrum[:half_len] = complex_spectrum
    
    # Convert to time domain
    reconstructed = np.real(np.fft.ifft(full_spectrum, norm="ortho"))
    
    # The reconstructed signal should have the same length
    assert len(reconstructed) == len(voltage), "Reconstructed signal has different length"
    
    # The reconstructed signal should be normalized
    assert np.max(np.abs(reconstructed)) <= 1.1, "Reconstructed signal exceeds normalized range"


def test_waveform_energy_in_band():
    """
    Test that at least 90% of the energy is in the specified frequency band.
    """
    # Create a waveform generator
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Generate a waveform
    _, _, voltage_spectral_mod, _ = waveform.sample()
    
    # Calculate energy in the specified band
    freq_mask = (waveform.freq >= waveform.start_freq) & (waveform.freq <= waveform.end_freq)
    in_band_energy = np.sum(voltage_spectral_mod[freq_mask]**2)
    total_energy = np.sum(voltage_spectral_mod**2)
    
    # At least 90% of energy should be in the specified band
    assert in_band_energy / total_energy > 0.9, "Too much energy outside the specified frequency band"


def test_waveform_reconstruction():
    """
    Test that the reconstructed signal is close to the original signal.
    """
    # Create a waveform generator
    waveform = Waveform(
        start_freq=10,
        end_freq=1000,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Generate a waveform
    _, voltage, voltage_spectral_mod, voltage_spectral_phase = waveform.sample()
    
    # Construct complex spectrum
    complex_spectrum = voltage_spectral_mod * np.exp(1j * voltage_spectral_phase)
    
    # Complete the spectrum for ifft (make it symmetric)
    # Only positive frequencies are returned by waveform.sample(), 
    # so half the power is missing and we multiply by two
    # Otherwise, the reconstructed signal would have half the amplitude.
    full_spectrum = np.hstack([2*complex_spectrum, np.zeros_like(complex_spectrum)])
    
    # Convert to time domain
    reconstructed = np.real(np.fft.ifft(full_spectrum, norm="ortho"))
    reconstructed = np.fft.fftshift(reconstructed)
    
    # The reconstructed signal should be normalized
    assert np.max(np.abs(reconstructed)) <= 1.1, "Reconstructed signal exceeds normalized range"
    
    # The reconstructed signal should be close to the original signal
    assert np.allclose(voltage, reconstructed, rtol=0.1), "Signal reconstruction failed"


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])
