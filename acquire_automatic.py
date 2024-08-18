#!/usr/bin/env python3

import os
import sys
import time
import matplotlib.pyplot as plt
import redpitaya_scpi as scpi
import numpy as np
from scipy.fftpack import fft
import math
import util

IP = 'rp-f0c04a.local'
rp_s = scpi.scpi(IP)
print('Connected to ' + IP)

def run_one_shot(use_freq, max_freq, start_freq=1, end_freq=1000, ampl=0.1, decimation=8192, 
                    store_data=False, plot_data=False, filename='data.h5py'):
    """Runs one shot of driving the speaker with a waveform and collecting the relevant data. 

    Args:
        start_freq (int, optional): the lower bound of the valid frequency range. Defaults to 1.
        end_freq (int, optional): the upper bound of the valid frequency range. Defaults to 1000.
        ampl (float, optional): the amplitude of the generated wave. Defaults to 0.1. 
        decimation (int, optional): Decimation that determines sample rate, should be power of 2. Defaults to 8192.
        store_data (bool, optional): Whether to store data in h5py file. Defaults to False.
        plot_data (bool, optional): Whether to plot data after acquisition. Defaults to False.
    """
    ##### Create Waveform #####

    N = 16384 # Number of samples in buffer
    SMPL_RATE_DEC1 = 125e6 # sample rate for decimation=1 in Samples/s (Hz)
    smpl_rate = SMPL_RATE_DEC1//decimation
    burst_time = N / smpl_rate

    wave_form = 'ARBITRARY'
    freq = 1 / burst_time

    t, y = util.bounded_frequency_waveform(start_freq, end_freq, length=N, sample_rate=smpl_rate, 
                                           use_freq=use_freq, max_freq=max_freq)
    y = util.linear_convert(y) # convert range of waveform to [-1, 1] to properly set ampl
    if plot_data:
        plt.plot(t, y)
        plt.show()

    ##### Reset Generation and Acquisition ######
    rp_s.tx_txt('GEN:RST')
    rp_s.tx_txt('ACQ:RST')

    ##### Generation #####
    # Function for configuring Source
    rp_s.sour_set(1, wave_form, ampl, freq, data=y)

    # Enable output
    rp_s.tx_txt('OUTPUT1:STATE ON')
    rp_s.tx_txt('SOUR1:TRig:INT')

    ##### Acqusition #####
    # Function for configuring Acquisition
    rp_s.acq_set(dec=decimation, trig_delay=0)
    rp_s.tx_txt('ACQ:START')
    time.sleep(1)
    rp_s.tx_txt('ACQ:TRig CH2_PE')

    # Wait for trigger
    while 1:
        rp_s.tx_txt('ACQ:TRig:STAT?') # Get Trigger Status
        if rp_s.rx_txt() == 'TD': # Triggered?
            break
    ## ! OS 2.00 or higher only ! ##
    while 1:
        rp_s.tx_txt('ACQ:TRig:FILL?')
        if rp_s.rx_txt() == '1':
            break

    ##### Analaysis #####
    # Read data and plot function for Data Acquisition
    pd_data = np.array(rp_s.acq_data(chan=1, convert=True)) # Volts
    speaker_data = np.array(rp_s.acq_data(chan=2, convert=True)) # Volts
    velocity_data, converted_signal, freq = util.velocity_waveform(speaker_data, smpl_rate, use_freq, max_freq)
    displ_data, _, _ = util.displacement_waveform(speaker_data, smpl_rate, use_freq, max_freq)
    y_vel, y_converted, _ = util.velocity_waveform(ampl*y, smpl_rate, use_freq, max_freq)
    time_data = np.linspace(0, N-1, num=N) / smpl_rate

    if plot_data:
        fig, ax = plt.subplots(nrows=3)

        ax[0].plot(time_data, pd_data, color='blue', label='Observed PD')
        ax[0].plot(time_data, speaker_data, color='black', label='Observed Drive')
        ax[0].plot(time_data, ampl*y, label='Drive Output', alpha=0.5)
        ax[0].legend()
        ax[0].set_ylabel('Amplitude (V)')
        ax[0].set_xlabel('Time (s)')

        ax[1].plot(freq, np.abs(fft(speaker_data)), color='black', label='Observed Drive', marker='.')
        ax[1].plot(freq, np.abs(converted_signal), color='green', label='Expected Observed Vel', marker='.')
        ax[1].plot(freq, np.abs(fft(ampl*y)), color='blue', label='Expected Drive', marker='.')
        ax[1].plot(freq, np.abs(y_converted), color='orange', label='Expected Ideal Vel', marker='.')
        ax[1].loglog()
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('$|\^{V}|$')
        ax[1].legend()

        ax[2].plot(time_data, velocity_data, color='black', label='Observed Drive')
        ax[2].plot(time_data, y_vel, label='Drive Output')
        ax[2].set_ylabel('Expected Vel (Microns/s)')
        ax[2].set_xlabel('Time (s)')
        ax[2].legend()
        plt.tight_layout()
        plt.show()

    if store_data:
        # Store data in h5py file
        path = "/Users/angelajia/Code/College/SMI/data/"
        # filename = "training_data.h5py"
        file_path = os.path.join(path, filename)

        entries = {
                'Time (s)': time_data, 
                'Speaker (V)': speaker_data,
                'Speaker (Microns/s)': velocity_data,
                'PD (V)': pd_data,
                'Speaker (Microns)': displ_data
                }
        
        util.write_data(file_path, entries)
                
    ##### Reset when closing program #####
    rp_s.tx_txt('GEN:RST')
    rp_s.tx_txt('ACQ:RST')

num_shots = 2000
amplitude = 0
end_freq = 1000
max_freq = 10*end_freq
for i in range(num_shots):
    amplitude = np.random.uniform(0.1, 0.6)
    if i % 400 == 0:
        print(f"{i}: ampl = {amplitude}")
    run_one_shot(True, max_freq, 30, end_freq, ampl=amplitude, decimation=256, store_data=True, plot_data=False, 
                 filename='test_max10kHz_30to1kHz_2kshots_dec=256_randampl.h5py')
    # print(i)