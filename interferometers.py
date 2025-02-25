#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import util
from tqdm import tqdm
import h5py


class MichelsonInterferometer:
    def __init__(self, wavelength, displacement_amplitude, phase):
        self.wavelength = wavelength # in microns
        self.displacement_amplitude = displacement_amplitude # in microns
        self.phase = phase # in radians, stands for random position offset
        self.displacement = None # in microns
        self.velocity = None # in microns/s
        self.time = None # in seconds

    def get_displacement(self, start_frequency, end_frequency, length, sample_rate):
        # Get a random displacement in time, resets each time it is called
        time, displacement = util.bounded_frequency_waveform(start_frequency,
                                                end_frequency, length, sample_rate, False)
        
        # scale max displacement to the desired amplitude
        displacement = displacement/np.max(np.abs(displacement)) * self.displacement_amplitude
        
        return time, displacement
    
    def set_displacement(self, displacement, time):
        # TODO: add checks and tests here, that the displacement is in right
        #  range, sampling, etc.
        self.displacement = displacement
        self.time = time
    
    def get_interferometer_output(self, start_frequency, end_frequency,
                                  measurement_noise_level, length,
                                  sample_rate, displacement, time):
        E0 = 1 + measurement_noise_level * util.bounded_frequency_waveform(1e3,
                                                                          1e6,
                                                                  length, sample_rate, False)[1]
        ER = 0.1 + measurement_noise_level * util.bounded_frequency_waveform(1e3,
                                                                          1e6,
                                                                    length, sample_rate, False)[1]
        
        if displacement is None:
            self.time, self.displacement = self.get_displacement(start_frequency,
                                                 end_frequency, length, sample_rate)
        else:
            self.set_displacement(displacement, time)
    
        interference = np.cos(2 * np.pi / self.wavelength * self.displacement
                              + self.phase)
        
        signal = E0**2 + ER**2 + 2 * E0 * ER * interference
        
        return signal
    
    def get_buffer(self, start_frequency=0, end_frequency=1e3,
                   displacement=None, time=None):
        self.signal = self.get_interferometer_output(start_frequency,
                                                     end_frequency, 0.3,
                                                     8192, 1e6, displacement,
                                                     time)
        
        self.velocity = np.diff(self.displacement)
        self.velocity = np.insert(self.velocity, 0, self.velocity[0])
        self.velocity /= (self.time[1] - self.time[0])
        
        # Remove DC offset
        self.signal = self.signal - np.mean(self.signal)
        
        return self.time, self.signal, self.displacement, self.velocity
        
    def plot_buffer(self):
        time, signal, displacement, velocity = self.get_buffer(0, 1e3)
        
        # time = time[0:256]
        # signal = signal[0:256]
        # displacement = displacement[0:256]
        # velocity = velocity[0:256]
        
        fig, ax1 = plt.subplots(figsize=(18, 6))
        ax1.plot(time, signal, color='b')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal (V)', color='b')
        ax1.tick_params('y', colors='b')
        
        ax2 = ax1.twinx()
        ax2.plot(time, displacement, color='r')
        ax2.plot(time, velocity, color='g')
        ax2.set_ylabel('Displacement (microns)', color='r')
        ax2.tick_params('y', colors='r')
        
        plt.tight_layout()
        plt.show()


def write_pretraining_data(num_shots, num_channels, file_path):
    if num_channels == 1:
        interferometer = MichelsonInterferometer(0.5, 5, np.pi / 4)
        for _ in tqdm(range(num_shots)):
            _, signal, _, velocity = interferometer.get_buffer()
            signal = np.expand_dims(signal, axis=-1)
            velocity = np.expand_dims(velocity, axis=-1)
            # Want to end up with these shapes in h5 file:
            # signal: (num_shots, buffer_size, 1)
            # velocity: (num_shots, buffer_size, 1)
            entries = {"signal": signal, "velocity": velocity}
            util.write_data(file_path, entries)
    elif num_channels == 2:
        interferometer1 = MichelsonInterferometer(1.064, 5, np.pi / 4)
        interferometer2 = MichelsonInterferometer(0.532, 5, 3 * np.pi / 4)
        for _ in tqdm(range(num_shots)):
            time, signal1, displacement, velocity = interferometer1.get_buffer()
            _, signal2, _, _ = interferometer2.get_buffer(displacement=displacement, time=time)
            signal = np.stack((signal1, signal2), axis=-1)
            velocity = np.expand_dims(velocity, axis=-1)
            # Want to end up with these shapes in h5 file:
            # signal: (num_shots, buffer_size, num_channels)
            # velocity: (num_shots, buffer_size, 1)
            entries = {"signal": signal, "velocity": velocity}
            # print(signal.shape)
            # print(velocity.shape)
            util.write_data(file_path, entries)
            
def plot_pretraining_data(file_path):
    with h5py.File(file_path, 'r') as f:
        signal = np.array(f['signal'])
        velocity = np.array(f['velocity'])
        print(signal.shape)
        print(velocity.shape)
        for shot_idx in range(signal.shape[0]):
            fig, ax1 = plt.subplots(figsize=(18, 6))
            ax1.plot(signal[shot_idx, :, 0], color='b', label='Signal 1')
            ax1.plot(signal[shot_idx, :, 1], color='g', label='Signal 2')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Signal (V)', color='b')
            ax1.tick_params('y', colors='b')
            
            ax2 = ax1.twinx()
            ax2.plot(velocity[shot_idx, :, 0], color='r')
            ax2.set_ylabel('Velocity (microns/second)', color='r')
            
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    np.random.seed(0x5EED + 4)
    write_pretraining_data(1000, 2,
                           "/Users/aj14/Desktop/SMI/data/train.h5py")
    # plot_pretraining_data("/Users/nolanpeard/Desktop/test.h5")
    