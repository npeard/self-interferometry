import numpy as np
from scipy.fftpack import fft, ifft, fftfreq

'''
Generates a random waveform within the given frequency range of a given length. 
'''
def bounded_frequency_waveform(start_frequency, end_frequency, length=1000, sample_rate=1000):
    # Create an evenly spaced time array
    t = np.linspace(0, 1.0, length, False)  # 1 second
    # Generate a random frequency spectrum between the start and end frequencies
    freq = np.linspace(0, sample_rate/2, length//2, False)
    spectrum = np.random.uniform(0, 1, len(freq))
    spectrum = np.where((freq >= start_frequency) & (freq <= end_frequency), spectrum, 0)
    c = np.random.rayleigh(np.sqrt(spectrum*(freq[1]-freq[0])))
    # See Phys. Rev. A 107, 042611 (2023) ref 28 for why we use the Rayleigh distribution here
    # Unless we use this distribution, the random noise will not be Gaussian distributed
    phase = np.random.uniform(-np.pi, np.pi, len(freq))
    # Use the inverse Fourier transform to convert the frequency domain signal back to the time domain
    # Also include a zero phase component
    spectrum = np.hstack([c*spectrum*np.exp(1j*phase), np.zeros_like(spectrum)])
    y = np.real(ifft(spectrum))
    y = np.fft.fftshift(y)
    return t, y

'''
Linearly scales data to a new range. 

@param data: assumed to be 1D array 
'''
def linear_convert(data, new_min=-1, new_max=1):
    old_min = np.min(data)
    old_max = np.max(data)
    old_range = old_max - old_min
    new_range = new_max - new_min
    return new_min + new_range * (data - old_min) / old_range

f0 = 257.1722062654443
Q = 15.680894756413974
k = 32.638601262518705
c = -3.219944358492212

'''
Calculates the expected displacement of the speaker at an inputted drive amplitude 'ampl' 
for a given frequency 'f', based on the calibration fit at 0.2Vpp. 

@param f: optimal range 20Hz-1kHz
@param ampl: optimal range 0-0.6V 
@return: expected displacement in microns 
'''
def A(f, ampl):
    return ampl * (k * f0**2) / np.sqrt((f0**2 - f**2)**2 + f0**2*f**2/Q**2)

'''
Calculates the phase delay between the speaker voltage waveform and the photodiode response
at a given frequency 'f'.

@param f: optimal range 20Hz-1kHz
@return: phase in radians
'''
def phase(f):
    return np.arctan2(f0/Q*f, f**2 - f0**2) + c

'''
Calculates the corresponding displacement waveform based on the given voltage waveform
using calibration. 
'''
def displacement_waveform(speaker_data, sample_rate, ampl, right=True):
    fourier_signal = fft(speaker_data)
    n = speaker_data.size
    sample_spacing = 1/sample_rate 
    freq = fftfreq(n, d=sample_spacing) # units: cycles/s = Hz
    
    # Multiply signal by transfer func in freq domain, then return to time domain
    converted_signal = fourier_signal * A(freq, ampl) * np.where(freq < 0, np.exp(-1j*phase(-freq)), np.exp(1j*phase(freq)))
    y = np.real(ifft(converted_signal))

    return y, converted_signal, freq