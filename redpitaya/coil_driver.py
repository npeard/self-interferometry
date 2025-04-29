#!/usr/bin/env python3
"""
Coil Driver - A class for calibrating and computing displacement and velocity from voltage waveforms
This module provides a class that encapsulates functionality for converting voltage waveforms
to displacement and velocity waveforms using calibration data.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq
from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass


@dataclass
class CalibrationParameters:
    """
    Calibration parameters for the coil driver.
    
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
    speaker_part_number: Optional[str] = None


class CoilDriver:
    """
    A class for calibrating and computing displacement and velocity from voltage waveforms.
    
    This class provides methods for:
    - Converting voltage waveforms to displacement waveforms
    - Converting voltage waveforms to velocity waveforms
    - Calculating transfer functions for the coil driver
    """
    
    def __init__(self, calibration_params: Optional[CalibrationParameters] = None):
        """
        Initialize the CoilDriver with calibration parameters.
        
        Args:
            calibration_params: Calibration parameters for the coil driver.
                                If None, default parameters are used.
        """
        if calibration_params is None:
            self.params = CalibrationParameters()
        else:
            self.params = calibration_params
    
    def _calculate_amplitude_transfer(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate the amplitude transfer function for the coil driver.
        
        Args:
            f: Frequencies at which to calculate the transfer function (Hz)
            
        Returns:
            Amplitude transfer function (microns/V)
        """
        return (self.params.k * self.params.f0**2) / np.sqrt(
            (self.params.f0**2 - f**2)**2 + self.params.f0**2 * f**2 / self.params.Q**2
        )
    
    def _calculate_phase_transfer(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate the phase transfer function for the coil driver.
        
        Args:
            f: Frequencies at which to calculate the transfer function (Hz)
            
        Returns:
            Phase transfer function (radians)
        """
        return np.arctan2(
            self.params.f0 / self.params.Q * f, 
            f**2 - self.params.f0**2
        ) + self.params.c
    
    def get_displacement(self, 
                        voltage_waveform: np.ndarray, 
                        sample_rate: float,
                        max_freq: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the displacement waveform from a voltage waveform.
        
        Args:
            voltage_waveform: Voltage waveform for the speaker (V)
            sample_rate: Sample rate used to generate voltage waveform (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included
            
        Returns:
            Tuple containing:
            - Displacement waveform in time domain (microns)
            - Displacement waveform in frequency domain
            - Frequency array (Hz)
        """
        # Calculate the spectrum of the voltage waveform
        voltage_spectrum = fft(voltage_waveform, norm="ortho")
        n = voltage_waveform.size
        sample_spacing = 1 / sample_rate
        freq = fftfreq(n, d=sample_spacing)  # units: cycles/s = Hz
        
        # Calculate the amplitude and phase transfer functions
        amplitude_transfer = self._calculate_amplitude_transfer(np.abs(freq))
        
        # Apply frequency limit if specified
        if max_freq is not None:
            amplitude_transfer = np.where(np.abs(freq) <= max_freq, amplitude_transfer, 0)
        
        # Apply the transfer function in the frequency domain
        # For negative frequencies, we need to conjugate the phase
        phase_transfer = np.where(
            freq < 0,
            np.exp(-1j * self._calculate_phase_transfer(-freq)),
            np.exp(1j * self._calculate_phase_transfer(freq))
        )
        
        # Multiply by the transfer function
        displacement_spectrum = voltage_spectrum * amplitude_transfer * phase_transfer
        
        # Convert back to time domain
        displacement_waveform = np.real(ifft(displacement_spectrum, norm="ortho"))
        
        return displacement_waveform, displacement_spectrum, freq
    
    def get_velocity(self, 
                    voltage_waveform: np.ndarray, 
                    sample_rate: float,
                    max_freq: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the velocity waveform from a voltage waveform.
        
        Args:
            voltage_waveform: Voltage waveform for the speaker (V)
            sample_rate: Sample rate used to generate voltage waveform (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included
            
        Returns:
            Tuple containing:
            - Velocity waveform in time domain (microns/s)
            - Velocity waveform in frequency domain
            - Frequency array (Hz)
        """
        # Calculate the spectrum of the voltage waveform
        voltage_spectrum = fft(voltage_waveform, norm="ortho")
        n = voltage_waveform.size
        sample_spacing = 1 / sample_rate
        freq = fftfreq(n, d=sample_spacing)  # units: cycles/s = Hz
        
        # Calculate the amplitude and phase transfer functions
        amplitude_transfer = self._calculate_amplitude_transfer(np.abs(freq))
        
        # Apply frequency limit if specified
        if max_freq is not None:
            amplitude_transfer = np.where(np.abs(freq) <= max_freq, amplitude_transfer, 0)
        
        # Apply the transfer function in the frequency domain
        # For negative frequencies, we need to conjugate the phase
        phase_transfer = np.where(
            freq < 0,
            np.exp(-1j * self._calculate_phase_transfer(-freq)),
            np.exp(1j * self._calculate_phase_transfer(freq))
        )
        
        # For velocity, we multiply by i*omega = i*2*pi*f in frequency domain
        velocity_spectrum = 1j * 2 * np.pi * freq * voltage_spectrum * amplitude_transfer * phase_transfer
        
        # Convert back to time domain
        velocity_waveform = np.real(ifft(velocity_spectrum, norm="ortho"))
        
        return velocity_waveform, velocity_spectrum, freq
    
    def set_calibration_parameters(self, params: CalibrationParameters):
        """
        Update the calibration parameters.
        
        Args:
            params: New calibration parameters
        """
        self.params = params
    
    def get_calibration_parameters(self) -> CalibrationParameters:
        """
        Get the current calibration parameters.
        
        Returns:
            Current calibration parameters
        """
        return self.params
    
    def save_calibration_to_dict(self) -> Dict:
        """
        Save the calibration parameters to a dictionary.
        
        Returns:
            Dictionary containing the calibration parameters
        """
        return {
            "f0": self.params.f0,
            "Q": self.params.Q,
            "k": self.params.k,
            "c": self.params.c,
            "speaker_part_number": self.params.speaker_part_number
        }
    
    @classmethod
    def from_dict(cls, params_dict: Dict) -> 'CoilDriver':
        """
        Create a CoilDriver from a dictionary of parameters.
        
        Args:
            params_dict: Dictionary containing the calibration parameters
            
        Returns:
            New CoilDriver instance with the specified parameters
        """
        params = CalibrationParameters(
            f0=params_dict.get("f0", 257.20857316296724),
            Q=params_dict.get("Q", 15.804110908084784),
            k=params_dict.get("k", 33.42493417407945),
            c=params_dict.get("c", -3.208233068626455),
            speaker_part_number=params_dict.get("speaker_part_number", None)
        )
        return cls(params)
