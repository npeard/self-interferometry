#!/usr/bin/env python3

import os
import sys
import time
import matplotlib.pyplot as plt
import redpitaya_scpi as scpi
import numpy as np
from numpy.fft import fft
# from scipy.fft import fft # use numpy 
import math
import util
from timeit import timeit
from datetime import datetime

IP = 'rp-f0c04a.local'
rp_s = scpi.scpi(IP)
print('Connected to ' + IP)

def run_one_shot(start_freq=1, end_freq=1000, ampl=0.1, gen_dec=8192, acq_dec=256, num_acq=1, 
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
    gen_smpl_rate = SMPL_RATE_DEC1//gen_dec
    acq_smpl_rate = SMPL_RATE_DEC1//acq_dec
    burst_time = N / gen_smpl_rate

    wave_form = 'ARBITRARY'
    freq = 1 / burst_time

    t, y = util.bounded_frequency_waveform(start_freq, end_freq, length=N, sample_rate=gen_smpl_rate, invert=True)
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
    # print("output enabled", datetime.now().time())

    ##### Acqusition #####
    pd_data = []
    speaker_data = []
    vel_data = []
    # Function for configuring Acquisition
    rp_s.tx_txt('ACQ:RST')
    rp_s.acq_set(dec=acq_dec, trig_delay=8192)

    for i in range(num_acq): 
        rp_s.tx_txt('ACQ:START')
        # ! OS 2.00 or higher only ! ##
        time.sleep(0.1)
        rp_s.tx_txt('ACQ:TRig NOW') # CH2_PE
        # Wait for trigger
        while 1:
            rp_s.tx_txt('ACQ:TRig:STAT?') # Get Trigger Status
            if rp_s.rx_txt() == 'TD': # Triggered?
                # print("td", datetime.now().time())
                break
        while 1:
            rp_s.tx_txt('ACQ:TRig:FILL?')
            if rp_s.rx_txt() == '1':
                # print("filled", datetime.now().time())
                break
        ##### Analysis #####
        # Read data and plot function for Data Acquisition
        pds = np.array(rp_s.acq_data(chan=1, convert=True))
        speaker = np.array(rp_s.acq_data(chan=2, convert=True))
        vels, _, _= util.velocity_waveform(speaker, acq_smpl_rate)
        acq_time_data = np.linspace(0, N-1, num=N) / acq_smpl_rate
        
        if plot_data:
            pd_data.append(pds) # Volts
            speaker_data.append(speaker) # Volts
            vel_data.append(vels)
        
        if store_data:
            # Store data in h5py file
            path = "/Users/angelajia/Code/College/SMI/data/"
            # filename = "training_data.h5py"
            file_path = os.path.join(path, filename)

            entries = {
                    'Speaker (V)': speaker,
                    'Speaker (Microns/s)': vels,
                    'PD (V)': pds
                    }
            util.write_data(file_path, entries)
            
    if plot_data:
        gen_time_data = np.linspace(0, N-1, num=N) / gen_smpl_rate
        pd_data = np.concatenate(pd_data)
        speaker_data = np.concatenate(speaker_data)
        vel_data = np.concatenate(vel_data)
        y_vel, y_converted, _ = util.velocity_waveform(ampl*y, gen_smpl_rate)

        fig, ax = plt.subplots(nrows=4)

        ax[0].plot(pd_data, color='blue', label='Observed PD')
        ax[0].plot(speaker_data, color='black', label='Observed Drive')
        # ax[0].plot(time_data, ampl*y, label='Drive Output', alpha=0.5)
        # ax[0].legend()
        ax[0].set_ylabel('Amplitude (V)')
        ax[0].set_xlabel('Samples')
        
        ax[1].plot(vel_data)
        ax[1].set_xlabel('Samples')
        ax[1].set_ylabel('Expected Vel (Microns/s)')

        ax[2].plot(gen_time_data, y_vel, label='Drive Output', alpha=0.7)
        ax[2].set_ylabel('Expected Vel (Microns/s)')
        ax[2].set_xlabel('Time (s)')
        # ax[1].legend()
        
        ax[3].plot(gen_time_data, ampl*y)
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('Amplitude (V)')
        plt.tight_layout()
        plt.show()
                
    ##### Reset when closing program #####
    rp_s.tx_txt('GEN:RST')
    rp_s.tx_txt('ACQ:RST')


num_shots = 2000
amplitude = 0
acq_num = 1
print("start", datetime.now().time())
for i in range(num_shots):
    amplitude = np.random.uniform(0.1, 0.6)
    if i % 400 == 0:
        print("\t", datetime.now().time())
        print(f"\t{i*acq_num}: ampl = {amplitude}")
    run_one_shot(1, 1000, ampl=amplitude, gen_dec=8192, acq_dec=256, num_acq=acq_num, store_data=True, plot_data=False, 
                filename='test_1to1kHz_invertspectra_trigdelay8192_sleep100ms_2kx1shots_dec=256_8192_randampl.h5py')
print("end", datetime.now().time())