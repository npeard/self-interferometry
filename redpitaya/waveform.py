#!/usr/bin/env python3
"""
Waveform Generator - A class for generating waveforms for Red Pitaya devices
This module provides a class that encapsulates functionality for generating
arbitrary waveforms with specific frequency characteristics.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
from typing import List, Tuple, Optional, Union

class Waveform:
    """
    A class for generating waveforms with specific frequency characteristics for Red Pitaya devices.
    
    This class provides methods for:
    - Generating random waveforms within specific frequency ranges
    - Restricting waveforms to specific frequencies
    - Sampling waveforms for Red Pitaya output
    """
    
    # Constants from RedPitayaManager
    BUFFER_SIZE = 16384  # Number of samples in buffer
    SAMPLE_RATE_DEC1 = 125e6  # Sample rate for decimation=1 in Samples/s (Hz)
    
    def __init__(self, 
                 start_freq: float, 
                 end_freq: float, 
                 gen_dec: int = 8192,
                 acq_dec: int = 256,
                 allowed_freqs: Optional[List[float]] = None,
                 ):
        """
        Initialize the Waveform generator.
        
        Args:
            start_freq: The lower bound of the valid frequency range (Hz)
            end_freq: The upper bound of the valid frequency range (Hz)
            gen_dec: Decimation for generation (default: 8192)
            acq_dec: Decimation for acquisition (default: 256)
            allowed_freqs: List of allowed frequencies to use (Hz), if None, all frequencies in range are allowed
        """
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.gen_dec = gen_dec
        self.acq_dec = acq_dec
        self.allowed_freqs = allowed_freqs
        
        # Calculate sample rates
        self.gen_sample_rate = self.SAMPLE_RATE_DEC1 / self.gen_dec
        self.acq_sample_rate = self.SAMPLE_RATE_DEC1 / self.acq_dec
        
        # Calculate time array and frequency array
        self.burst_time = self.BUFFER_SIZE / self.gen_sample_rate
        self.t = np.linspace(0, self.burst_time, self.BUFFER_SIZE, endpoint=False)
        self.freq = np.linspace(0, self.gen_sample_rate/2, self.BUFFER_SIZE//2, endpoint=False)
        
        # Initialize spectrum and phases to None (will be randomized on first sample)
        self.spectrum = None
        self.phases = None
        
        # If allowed_freqs is not provided, use all frequencies in the range
        if self.allowed_freqs is None:
            # Calculate valid frequencies based on acquisition sample rate
            self.valid_freqs = fftfreq(self.BUFFER_SIZE, d=1/self.acq_sample_rate)
            # Filter to only include frequencies in the specified range
            self.valid_freqs = np.array([f for f in self.valid_freqs if start_freq <= abs(f) <= end_freq and f != 0])
        else:
            self.valid_freqs = np.array(self.allowed_freqs)
    
    def _randomize_spectrum(self):
        """
        Randomize the spectrum and phases for the waveform.
        This is called internally by sample() if needed.
        """
        # Generate a random frequency spectrum between the start and end frequencies
        valid_freqs = self.valid_freqs[:, np.newaxis]
        valid_mask = np.any(np.abs(self.freq - valid_freqs) <= 0.5, axis=0)
        
        spectrum = np.random.uniform(0.0, 1.0, len(self.freq))
        spectrum = np.where((self.freq >= self.start_freq) & 
                            (self.freq <= self.end_freq) & 
                            valid_mask, 
                            spectrum, 0)
        
        # Generate random Rayleigh distributed power spectrum
        rayleigh_spectrum = np.random.rayleigh(np.sqrt(spectrum * (self.freq[1] - self.freq[0])))
        # See Phys. Rev. A 107, 042611 (2023) and https://doi.org/10.1016/0141-1187(84)90050-6
        # for why we use the Rayleigh distribution here
        # If we want Gaussian distributed a_n, b_n in a Fourier series for cosine and sine components,
        # each with variance_n = S(f_n)*\Delta f_n, then c_n = sqrt(a_n**2 + b_n**2) is Rayleigh distributed
        # with variance_n = S(f_n)*\Delta f_n
        # np.random.rayleigh(scale) expects a scale parameter = sqrt(variance) as input
        
        # Generate random phases
        phi = np.random.uniform(0, 2*np.pi, len(self.freq))
        
        # Store the spectrum and phases
        self.spectrum = np.sqrt(rayleigh_spectrum) * np.exp(1j * phi)
    
    def _randomize_phase(self):
        """
        Randomize only the phases for the waveform, keeping the spectrum amplitudes the same.
        This is called internally by sample() if randomize_phase_only=True.
        """
        # Generate random phases
        phi = np.random.uniform(0, 2*np.pi, len(self.freq))
        
        # If no spectrum exists yet, we need to call _randomize_spectrum first
        if self.spectrum is None:
            self._randomize_spectrum()

        # Then override with just the phase randomization
        amplitudes = np.abs(self.spectrum)
        self.spectrum = amplitudes * np.exp(1j * phi)
    
    def sample(self, randomize_phase_only=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a sample waveform using the current configuration.
        Randomizes the spectrum and phases each time it's called.
        
        Args:
            randomize_phase_only: If True, only randomize the phases while keeping the spectrum amplitudes the same
    
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - Array of time points
                - Amplitude points in time domain
                - Spectrum (amplitude)
                - Spectral phases (phase)
        """
        # Randomize the spectrum and/or phases
        if randomize_phase_only:
            self._randomize_phase()
        else:
            self._randomize_spectrum()
        
        # Positive frequencies only, TODO: is this right?
        full_spectrum = np.hstack([2*self.spectrum, np.zeros_like(self.spectrum)])
        
        # Convert to time domain
        y = np.real(ifft(full_spectrum, norm="ortho"))
        y = np.fft.fftshift(y)
        
        # Normalize to [-1, 1] range with random scaling inside those bounds
        normalization_factor = np.max(np.abs(y)) * np.random.uniform(1,10)
        y = y / normalization_factor
        
        # Recompute the spectrum and phase after normalization
        # First, shift back to match the original order
        y_unshifted = np.fft.ifftshift(y)
        
        # Compute the FFT to get the normalized spectrum
        normalized_spectrum = fft(y_unshifted, norm="ortho")
        
        # Extract the first half (positive frequencies)
        normalized_complex_spectrum = normalized_spectrum[:len(self.freq)]
        
        # Extract amplitude and phase
        normalized_amplitude = np.abs(normalized_complex_spectrum)
        normalized_phase = np.angle(normalized_complex_spectrum)
        
        return self.t, y, normalized_amplitude, normalized_phase
    
    def get_valid_frequencies(self) -> np.ndarray:
        """
        Get the list of valid frequencies for this waveform.
        
        Returns:
            np.ndarray: Array of valid frequencies in Hz
        """
        return self.valid_freqs
    
    def set_frequency_range(self, start_freq: float, end_freq: float):
        """
        Update the frequency range for the waveform.
        
        Args:
            start_freq: New start frequency in Hz
            end_freq: New end frequency in Hz
        """
        self.start_freq = start_freq
        self.end_freq = end_freq
        
        # Update valid frequencies if we're not using allowed_freqs
        if self.allowed_freqs is None:
            self.valid_freqs = fftfreq(self.BUFFER_SIZE, d=1/self.acq_sample_rate)
            self.valid_freqs = np.array([f for f in self.valid_freqs if start_freq <= abs(f) <= end_freq and f != 0])
    
    def set_allowed_frequencies(self, allowed_freqs: List[float]):
        """
        Set the list of allowed frequencies.
        
        Args:
            allowed_freqs: List of allowed frequencies in Hz
        """
        self.allowed_freqs = allowed_freqs
        self.valid_freqs = np.array(allowed_freqs)
    
    def set_decimation(self, gen_dec: int, acq_dec: int):
        """
        Update the decimation values and recalculate dependent parameters.
        
        Args:
            gen_dec: New generation decimation value
            acq_dec: New acquisition decimation value
        """
        self.gen_dec = gen_dec
        self.acq_dec = acq_dec
        
        # Recalculate sample rates
        self.gen_sample_rate = self.SAMPLE_RATE_DEC1 / self.gen_dec
        self.acq_sample_rate = self.SAMPLE_RATE_DEC1 / self.acq_dec
        
        # Recalculate time array and frequency array
        self.burst_time = self.BUFFER_SIZE / self.gen_sample_rate
        self.t = np.linspace(0, self.burst_time, self.BUFFER_SIZE, endpoint=False)
        self.freq = np.linspace(0, self.gen_sample_rate/2, self.BUFFER_SIZE//2, endpoint=False)
        
        # Update valid frequencies if we're not using allowed_freqs
        if self.allowed_freqs is None:
            self.valid_freqs = fftfreq(self.BUFFER_SIZE, d=1/self.acq_sample_rate)
            self.valid_freqs = np.array([f for f in self.valid_freqs if 
                                       self.start_freq <= abs(f) <= self.end_freq and f != 0])
