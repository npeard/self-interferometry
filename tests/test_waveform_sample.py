#!/usr/bin/env python3
"""
Test for the Waveform.sample() method.

This test compares the spectrum values returned by the Waveform.sample() method
with those calculated manually using the calculate_fft function.
"""

import pytest
import numpy as np

from redpitaya.waveform import Waveform
from plot_waveforms import calculate_fft


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
    
    # Normally, we would only multiply by sqrt(2) to double the power
    # However, waveform.sample takes the FFT again to return the spectrum and then halves
    # the spectrum, so we need to multiply by sqrt(2) twice?
    full_spectrum = np.hstack([2*complex_spectrum, np.zeros_like(complex_spectrum)])
    
    # Convert to time domain
    reconstructed = np.real(np.fft.ifft(full_spectrum, norm="ortho"))
    reconstructed = np.fft.fftshift(reconstructed)
    
    # The reconstructed signal should be normalized
    assert np.max(np.abs(reconstructed)) <= 1.0, "Reconstructed signal exceeds normalized range"
    
    # The reconstructed signal should be close to the original signal
    assert np.allclose(voltage, reconstructed, rtol=0.1), "Signal reconstruction failed"


def test_waveform_statistics():
    """
    Test that the Gaussian distributions overlaid on the histograms in plot_waveform_histograms
    accurately represent the statistical properties of the waveform samples.
    
    This test:
    1. Generates multiple waveform samples with phase randomization only
    2. Computes the expected Gaussian distributions based on the noise variance
    3. Compares the statistical properties of the samples with the expected distributions
    4. Verifies that the computed variance matches the empirical variance of the samples
    """
    # Create a waveform generator with test parameters
    start_freq = 100
    end_freq = 500
    num_samples = 20  # Reduced for test efficiency
    
    waveform = Waveform(
        start_freq=start_freq,
        end_freq=end_freq,
        gen_dec=8192,
        acq_dec=256
    )
    
    # Update frequency range
    waveform.set_frequency_range(start_freq, end_freq)
    
    # Initialize arrays to store all voltage samples and reconstructed complex signals
    all_voltages = []
    all_complex_values = []
    
    # Store the last voltage_spectral_amp for variance calculation
    last_voltage_spectral_amp = None
    last_t = None
    
    # Generate multiple waveform samples
    for i in range(num_samples):
        # Generate a random waveform with phase randomization only
        t, voltage, voltage_spectral_amp, voltage_spectral_phase = waveform.sample(randomize_phase_only=True)
        
        # Store for later use
        last_voltage_spectral_amp = voltage_spectral_amp
        last_t = t
        
        # Store the time-domain voltage values
        all_voltages.extend(voltage)
        
        # Construct complex spectrum
        complex_spectrum = voltage_spectral_amp * np.exp(1j * voltage_spectral_phase)
        
        # Complete the spectrum for ifft (make it symmetric)
        full_spectrum = np.hstack([np.sqrt(2)*complex_spectrum, np.zeros_like(complex_spectrum)])
        
        # Convert to time domain without taking real part
        reconstructed_complex = np.fft.ifft(full_spectrum, norm="ortho")
        reconstructed_complex = np.fft.fftshift(reconstructed_complex)
        
        # Store the complex-valued reconstructed signals
        all_complex_values.extend(reconstructed_complex)
    
    # Convert to numpy arrays
    all_voltages = np.array(all_voltages)
    all_complex_values = np.array(all_complex_values)
    
    # Compute variance for the noise distribution using the same formula as in plot_waveform_histograms
    noise_variance = np.sum(2*last_voltage_spectral_amp**2) * len(last_t) * (last_t[1] - last_t[0])**2/np.max(last_t) * (waveform.freq[1] - waveform.freq[0])
    
    # Test 1: Check that the mean of the time-domain voltage values is close to zero
    assert np.isclose(np.mean(all_voltages), 0, atol=0.05), \
        f"Mean of voltage values ({np.mean(all_voltages)}) is not close to zero"
    
    # Test 2: Check that the empirical variance of the time-domain voltage values is close to the computed noise variance
    empirical_variance = np.var(all_voltages)
    assert np.isclose(empirical_variance, noise_variance, rtol=0.3), \
        f"Empirical variance ({empirical_variance}) doesn't match computed variance ({noise_variance})"
    
    # Test 3: Check that the real and imaginary parts of the complex values have variances close to half the computed noise variance
    # For a complex Gaussian, the variance is split between real and imaginary parts
    real_parts = np.real(all_complex_values)
    imag_parts = np.imag(all_complex_values)
    
    real_variance = np.var(real_parts)
    imag_variance = np.var(imag_parts)
    expected_complex_variance = noise_variance / 2 # Half the variance for each component
    
    assert np.isclose(real_variance, expected_complex_variance, rtol=0.3), \
        f"Real part variance ({real_variance}) doesn't match expected variance ({expected_complex_variance})"
    assert np.isclose(imag_variance, expected_complex_variance, rtol=0.3), \
        f"Imaginary part variance ({imag_variance}) doesn't match expected variance ({expected_complex_variance})"
    
    # Test 4: Check that the real and imaginary parts are uncorrelated (for a circular complex Gaussian)
    correlation = np.corrcoef(real_parts, imag_parts)[0, 1]
    assert abs(correlation) < 0.1, \
        f"Real and imaginary parts are correlated ({correlation}), should be uncorrelated"
    
    # Test 5: Check the quartiles of the data against theoretical quartiles for a Gaussian
    # For a standard normal distribution, the quartiles are approximately -0.67, 0, and 0.67
    
    # Normalize the voltage values
    normalized_voltages = (all_voltages - np.mean(all_voltages)) / np.std(all_voltages)
    
    # Get the empirical quartiles
    q1, q2, q3 = np.percentile(normalized_voltages, [25, 50, 75])
    
    # Check against theoretical quartiles with some tolerance
    assert np.isclose(q1, -0.67, atol=0.1), f"First quartile ({q1}) is not close to expected value (-0.67)"
    assert np.isclose(q2, 0.0, atol=0.1), f"Median ({q2}) is not close to expected value (0.0)"
    assert np.isclose(q3, 0.67, atol=0.1), f"Third quartile ({q3}) is not close to expected value (0.67)"
    
    # Test 6: Check that the range of the data is reasonable for a Gaussian
    # For a large sample from a Gaussian, we expect most values to be within ±3 standard deviations
    normalized_max = np.max(np.abs(normalized_voltages))
    assert normalized_max < 5.0, \
        f"Maximum absolute normalized value ({normalized_max}) is too large for a Gaussian distribution"


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])
