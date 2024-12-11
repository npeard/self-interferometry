import numpy as np
from numpy.fft import fft, ifft, fftfreq
import h5py

def bounded_frequency_waveform(start_frequency, end_frequency, length, sample_rate):
    """Generates a random waveform within the given frequency range of a given length.
    Args:
        start_frequency (float): the lower bound of the valid frequency range
        end_frequency (float): the upper bound of the valid frequency range
        length (int): the number of values to generate
        sample_rate (float): the rate at which to sample values

    Returns:
        [1darr, 1darr]: the array of time points and amplitude points in time domain
    """
    # Create an evenly spaced time array
    t = np.linspace(0, 1.0, length, False)  # 1 second
    # Generate a random frequency spectrum between the start and end frequencies
    freq = np.linspace(0, sample_rate / 2, length // 2, False)
    spectrum = np.random.uniform(0, 1, len(freq))
    spectrum = np.where((freq >= start_frequency) & (freq <= end_frequency), spectrum, 0)
    c = np.random.rayleigh(np.sqrt(spectrum * (freq[1] - freq[0])))
    # See Phys. Rev. A 107, 042611 (2023) ref 28 for why we use the Rayleigh distribution here
    # Unless we use this distribution, the random noise will not be Gaussian distributed
    phi = np.random.uniform(-np.pi, np.pi, len(freq))
    # Use the inverse Fourier transform to convert the frequency domain signal back to the time domain
    # Also include a zero phase component
    spectrum = spectrum * c * np.exp(1j*phi)
    if invert:
        spectrum = np.divide(spectrum, A(freq) * np.exp(1j*phase(freq)))
    spectrum = np.hstack([spectrum, np.zeros_like(spectrum)])
    y = np.real(ifft(spectrum, norm="ortho"))
    y = np.fft.fftshift(y)
    return t, y


def linear_convert(data, new_min=-1, new_max=1):
    """Linearly scales data to a new range. Default is [-1, 1].

    Args:
        data (1darr): data to scale
        new_min (float, optional): new minimum value for data. Defaults to -1.
        new_max (float, optional): new maximum value for data. Defaults to 1.

    Returns:
        1darr: the newly scaled data
    """
    old_min = np.min(data)
    old_max = np.max(data)
    old_range = old_max - old_min
    new_range = new_max - new_min
    return new_min + new_range * (data - old_min) / old_range


def write_data(file_path, entries):
    """Add data to a given dataset in 'file'. Creates dataset if it doesn't exist;
        otherwise, appends.
    Args:
        file_path (string): the name of the output HDF5 file to which to append data
        entries (dict<str, 1darr>): dictionary of column name & corresponding data
    """
    with h5py.File(file_path, 'a') as f:
        for col_name, col_data in entries.items():
            if col_name in f.keys():
                f[col_name].resize((f[col_name].shape[0] + 1), axis=0)
                new_data = np.expand_dims(col_data, axis=0)
                f[col_name][-1:] = new_data
            else:
                f.create_dataset(col_name,
                                 data=np.expand_dims(col_data, axis=0),
                                 maxshape=(None, col_data.shape[0],
                                           col_data.shape[1]),
                                 chunks=True)

                
# Constants from calibration_rp using RPRPData.csv
f0 = 257.20857316296724
Q = 15.804110908084784
k = 33.42493417407945
c = -3.208233068626455


def A(f):
    """Calculates the expected displacement of the speaker at an inputted drive amplitude 'ampl' for a given frequency 'f',
        based on the calibration fit at 0.2Vpp.

    Args:
        f (1darr): frequencies at which to calculate expected displacement

    Returns:
        1darr: expected displacement/V_ampl in microns/V
    """
    return (k * f0**2) / np.sqrt((f0**2 - f**2)**2 + f0**2 * f**2 / Q**2)


def phase(f):
    """Calculates the phase delay between the speaker voltage waveform and the photodiode response
        at a given frequency 'f'.

    Args:
        f (1darr): frequencies at which to calculate expected displacement

    Returns:
        1darr: phase in radians
    """
    return np.arctan2(f0 / Q * f, f**2 - f0**2) + c


def displacement_waveform(speaker_data, sample_rate, use_freq, max_freq):
    """Calculates the corresponding displacement waveform based on the given voltage waveform
        using calibration.

    Args:
        speaker_data (1darr): voltage waveform for speaker
        sample_rate (float): sample rate used to generate voltage waveform

    Returns:
        [1darr, 1darr, 1darr]: converted displacement waveform (microns) in time domain,
                                converted displacement waveform in frequency domain,
                                frequency array (Hz)
    """
    speaker_spectrum = fft(speaker_data, norm="ortho")
    n = speaker_data.size
    sample_spacing = 1 / sample_rate 
    freq = fftfreq(n, d=sample_spacing) # units: cycles/s = Hz
    
    # Multiply signal by transfer func in freq domain, then return to time domain
    converted_signal = speaker_spectrum * A(freq) * np.where(freq < 0, 
                                                             np.exp(-1j*phase(-freq)), np.exp(1j*phase(freq)))
    y = np.real(ifft(converted_signal, norm="ortho"))

    return y, converted_signal, freq

def velocity_waveform(speaker_data, sample_rate):
    """Calculates the corresponding velocity waveform based on the given voltage waveform
        using calibration.

    Args:
        speaker_data (1darr): voltage waveform for speaker
        sample_rate (float): sample rate used to generate voltage waveform

    Returns:
        [1darr, 1darr, 1darr]: converted velocity waveform (microns/s) in time domain,
                                converted velocity waveform in frequency domain,
                                frequency array (Hz)
    """
    speaker_spectrum = fft(speaker_data, norm="ortho")
    n = speaker_data.size
    sample_spacing = 1 / sample_rate 
    freq = fftfreq(n, d=sample_spacing) # units: cycles/s = Hz
    
    # Multiply signal by transfer func in freq domain, then return to time domain
    converted_signal = 1j * freq * speaker_spectrum * \
        A(freq) * np.where(freq < 0, np.exp(-1j*phase(-freq)), np.exp(1j*phase(freq)))
    v = np.real(ifft(converted_signal, norm="ortho"))

    return v, converted_signal, freq
