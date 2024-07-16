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
    c = np.random.rayleigh(np.sqrt(4*spectrum*(freq[1]-freq[0])))
    # See Jiang 2023 ref 28 for why we use the Rayleigh distribution here
    # Unless we use this distribution, the random noise will not be Gaussian distributed
    phase = np.random.uniform(-np.pi, np.pi, len(freq))

    # Use the inverse Fourier transform to convert the frequency domain signal back to the time domain
    # Also include a zero phase component
    spectrum = np.hstack([spectrum*np.exp(1j*phase), np.zeros_like(spectrum)])
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