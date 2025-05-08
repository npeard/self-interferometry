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
    
    def _random_single_tone_spectrum(self):
        """
        Generate a spectrum containing a single tone at a randomly selected frequency.
        This is called internally by sample() if random_single_tone=True.
        """
        # Create an empty spectrum
        spectrum = np.zeros(len(self.freq), dtype=complex)
        
        # Randomly select one of the valid frequencies
        if len(self.valid_freqs) > 0:
            selected_freq = np.random.choice(self.valid_freqs[self.valid_freqs > 0])
        else:
            # If no valid frequencies, use the middle of the range
            selected_freq = (self.start_freq + self.end_freq) / 2
        
        # Find the index in the frequency array closest to the selected frequency
        freq_idx = np.argmin(np.abs(self.freq - selected_freq))
        
        # Generate a random phase
        phi = np.random.uniform(0, 2*np.pi)
        
        # Set the amplitude at the selected frequency to 1 with the random phase
        spectrum[freq_idx] = 1.0 * np.exp(1j * phi)
        
        # Store the spectrum
        self.spectrum = spectrum
    
    def sample(self, randomize_phase_only=False, random_single_tone=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a sample waveform using the current configuration.
        Randomizes the spectrum and phases each time it's called.
        
        Args:
            randomize_phase_only: If True, only randomize the phases while keeping the spectrum amplitudes the same
            random_single_tone: If True, generate a single tone at a randomly selected valid frequency
    
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - Array of time points
                - Amplitude points in time domain
                - Spectrum (amplitude)
                - Spectral phases (phase)
        """
        # Generate appropriate spectrum based on parameters
        if random_single_tone:
            self._random_single_tone_spectrum()
        elif randomize_phase_only:
            self._randomize_phase()
        else:
            self._randomize_spectrum()
        
        # Only positive frequencies are specified in the spectrum, so we need to 
        # multiply the amplitude spectrum by sqrt(2) to double the power across the time series
        full_spectrum = np.hstack([np.sqrt(2)*self.spectrum, np.zeros_like(self.spectrum)])
        
        # Convert to time domain
        y = np.real(ifft(full_spectrum, norm="ortho"))
        y = np.fft.fftshift(y)
        
        # Recompute the spectrum and phase for use in other modules
        # First, shift back to match the original order
        y_unshifted = np.fft.ifftshift(y)
        
        # Compute the FFT to get the normalized spectrum
        normalized_spectrum = fft(y_unshifted, norm="ortho")[:len(self.freq)]
        
        return self.t, y, np.abs(normalized_spectrum), np.angle(normalized_spectrum)
    
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
