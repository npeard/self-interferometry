#!/usr/bin/env python3

import os
import sys
import time
import matplotlib.pyplot as plt
import redpitaya_scpi as scpi
import numpy as np
from numpy.fft import fft, fftfreq
# from scipy.fft import fft # use numpy
import math
import util
from timeit import timeit
from datetime import datetime
from scipy.signal import butter, filtfilt

IP = 'rp-f0c04a.local'
rp_s = scpi.scpi(IP)
print('Connected to ' + IP)

# plt.rcParams.update({
#     "text.usetex": True
# })

def run_one_shot(start_freq=1, end_freq=1000, ampl=0.1, gen_dec=8192, acq_dec=256,
                 store_data=False, plot_data=False, filename='data.h5py'):
    """Runs one shot of driving the speaker with a waveform and collecting the relevant data.
    Samples only frequencies accessible at acquisition sample rate.

    Args:
        start_freq (int, optional): the lower bound of the valid frequency range. Defaults to 1.
        end_freq (int, optional): the upper bound of the valid frequency range. Defaults to 1000.
        ampl (float, optional): the amplitude of the generated wave. Defaults to 0.1.
        gen_dec (int, optional): Decimation that determines generation sample rate, should be
        power of 2. Defaults to 8192.
        acq_dec (int, optional): Decimation that determines acquisition sample rate, should be
        power of 2. Defaults to 256.
        store_data (bool, optional): Whether to store data in h5py file. Defaults to False.
        plot_data (bool, optional): Whether to plot data after acquisition. Defaults to False.
        filename (string, optional): Name of file to store data.
    """
    ##### Create Waveform #####

    N = 16384  # Number of samples in buffer
    SMPL_RATE_DEC1 = 125e6  # sample rate for decimation=1 in Samples/s (Hz)
    gen_smpl_rate = SMPL_RATE_DEC1 // gen_dec
    acq_smpl_rate = SMPL_RATE_DEC1 // acq_dec
    burst_time = N / gen_smpl_rate

    wave_form = 'ARBITRARY'
    freq = 1 / burst_time

    valid_freqs = fftfreq(N, d=1 / acq_smpl_rate)

    t, y = util.bounded_specific_frequencies(start_freq, end_freq, N, gen_smpl_rate, valid_freqs, True)
    y = util.linear_convert(y)  # convert range of waveform to [-1, 1] to properly set ampl

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
    # Function for configuring Acquisition
    rp_s.tx_txt('ACQ:RST')
    # trig_delay=8192 sets trigger event at beginning of recorded data
    rp_s.acq_set(dec=acq_dec, trig_delay=8192)

    rp_s.tx_txt('ACQ:START')
    # ! OS 2.00 or higher only ! ##
    min_sleep_time = 0.4  # tested that transients removed in this time
    time.sleep(min_sleep_time)
    rp_s.tx_txt('ACQ:TRig NOW')  # CH2_PE
    time.sleep(min_sleep_time)
    # Wait for trigger
    while 1:
        rp_s.tx_txt('ACQ:TRig:STAT?')  # Get Trigger Status
        if rp_s.rx_txt() == 'TD':  # Triggered?
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
    # Expand dims to make single interferometer compatible with multiple
    pds = np.expand_dims(pds, axis=1) 
    speaker = np.expand_dims(speaker, axis=1)

    if store_data:
        # Store data in h5py file
        path = "/Users/angelajia/Code/College/SMI/data/"
        file_path = os.path.join(path, filename)
        entries = {
            'drive': speaker,
            'signal': pds
        }
        util.write_data(file_path, entries)

    if plot_data:
        vel_data, vel_converted, freqs = util.velocity_waveform(speaker, acq_smpl_rate)
        # acq_time_data = np.linspace(0, N - 1, num=N) / acq_smpl_rate

        fig, ax = plt.subplots(nrows=4)

        ax[0].plot(pds, color='blue', label='Observed PD')  # , marker='.')
        ax[0].plot(speaker, color='black', label='Observed Drive')  # , marker='.')
        ax[0].legend()
        ax[0].set_ylabel('Amplitude (V)')
        ax[0].set_xlabel('Samples')

        ax[1].set_title("Vel for acq_dec=256")
        ax[1].plot(vel_data)
        ax[1].set_ylabel('Expected Vel (Microns/s)')
        ax[1].set_xlabel('Samples')

        ax[2].plot(fftfreq(speaker.shape[0], d=1 / acq_smpl_rate), np.abs(fft(speaker, norm='ortho')), marker='.')
        ax[3].plot(freqs, np.abs(vel_converted), marker='.')
        plt.tight_layout()
        plt.show()

    ##### Reset when closing program #####
    rp_s.tx_txt('GEN:RST')
    rp_s.tx_txt('ACQ:RST')


filenames = ['test_03_31_2025.h5py']
shots = [2]
# Store acq_smpl_rate for velocity conversion in Dataloader
store_data = False
acq_dec = 256
SMPL_RATE_DEC1 = 125e6  # sample rate for decimation=1 in Samples/s (Hz)
acq_smpl_rate = SMPL_RATE_DEC1 // acq_dec
acq_spacing = 29.81  # for acq_dec=256

print("start", datetime.now().time())
for (file, num_shots) in zip(filenames, shots):
    print(file)
    print(num_shots)
    if store_data:
        path = "/Users/angelajia/Code/College/SMI/data/"
        file_path = os.path.join(path, file)
        entries = {
            'acq_smpl_rate': np.array([[acq_smpl_rate]])
        }
        util.write_data(file_path, entries)
    for i in range(num_shots):
        amplitude = np.random.uniform(0.1, 0.6)  # 0.1
        print(amplitude)
        if i % 100 == 0:
            print("\t", datetime.now().time())
        # lower = np.random.uniform(0, 970)
        # upper = lower + acq_spacing
        run_one_shot(1, 1000, ampl=amplitude, gen_dec=8192, acq_dec=acq_dec, 
                        store_data=store_data, plot_data=True, filename=file)
    print("end", datetime.now().time())
