#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import util

class MichelsonInterferometer:
    def __init__(self, wavelength, displacement_amplitude, phase):
        self.wavelength = wavelength # in microns
        self.displacement_amplitude = displacement_amplitude # in microns
        self.phase = phase
        self.displacement = None
        self.velocity = None

    def get_displacement(self, start_frequency, end_frequency, length, sample_rate):
        # Get a random displacement in time, resets each time it is called
        time, displacement = util.bounded_frequency_waveform(start_frequency,
                                                end_frequency, length, sample_rate)
        
        # scale max displacement to the desired amplitude
        displacement = displacement/np.max(np.abs(displacement)) * self.displacement_amplitude
        
        return time, displacement
    
    def interferometer_output(self, start_frequency, end_frequency,
                              measurement_noise_level, length, sample_rate):
        E0 = 1 + measurement_noise_level * util.bounded_frequency_waveform(1e3,
                                                                          1e6,
                                                                  length, sample_rate)[1]
        ER = 0.1 + measurement_noise_level * util.bounded_frequency_waveform(1e3,
                                                                          1e6,
                                                                    length, sample_rate)[1]
        
        self.time, self.displacement = self.get_displacement(start_frequency,
                                                 end_frequency, length, sample_rate)
    
        interference = np.cos(2 * np.pi / self.wavelength * self.displacement)
        
        signal = E0**2 + ER**2 + 2 * E0 * ER * interference
        
        return signal
    
    def get_pretraining_data(self, start_frequency, end_frequency):
        self.signal = self.interferometer_output(start_frequency,
                                                 end_frequency, 0.1, 8192, 1e6)
        
        self.velocity = np.diff(self.displacement)
        self.velocity = np.insert(self.velocity, 0, self.velocity[0])
        self.velocity /= (self.time[1] - self.time[0])
        
        return self.time, self.signal, self.displacement, self.velocity
    
    def plot_pretraining_data(self):
        time, signal, displacement, velocity = self.get_pretraining_data(0, 1e3)
        
        plt.plot(time, signal)
        plt.plot(time, displacement)
        plt.plot(time, velocity)
        plt.tight_layout()
        plt.show()
        
if __name__ == '__main__':
    interferometer = MichelsonInterferometer(0.5, 5, 0)
    interferometer.plot_pretraining_data()
    