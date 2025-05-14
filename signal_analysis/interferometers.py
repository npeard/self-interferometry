#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class MichelsonInterferometer:
    def __init__(self, wavelength, phase):
        self.wavelength = wavelength # in meters
        self.phase = phase # in radians, stands for random position offset
    
    def get_interferometer_output(self, displacement, time):
        E0 = 1
        ER = 0.1
        
        interference = np.cos(2 * np.pi / self.wavelength * displacement
                              + self.phase)
        
        signal = E0**2 + ER**2 + 2 * E0 * ER * interference
        
        return signal
    
    def get_simulated_buffer(self, displacement=None, time=None):
        signal = self.get_interferometer_output(displacement, time)
        
        velocity = np.diff(displacement)
        velocity = np.insert(velocity, 0, velocity[0])
        velocity /= (time[1] - time[0])
        
        # Remove DC offset
        signal = signal - np.mean(signal)
        
        return time, signal, displacement, velocity
        
    def plot_buffer(self):
        time, signal, displacement, velocity = self.get_simulated_buffer()
        
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
    