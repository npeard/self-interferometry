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
from datetime import datetime
import csv

IP = 'rp-f0c04a.local'
rp_s = scpi.scpi(IP)
print('Connected to ' + IP)

store_data = False # whether or not to save data to CSV

##### Create Waveform #####

N = 16384 # Number of samples in buffer
SMPL_RATE_DEC1 = 125e6 # sample rate for decimation=1 in Samples/s (Hz)
decimation = 8192
smpl_rate = SMPL_RATE_DEC1//decimation
burst_time = N / smpl_rate

wave_form = 'ARBITRARY'
freq =  1 / burst_time
ampl = 0.3 # good range 0-0.6V

# t, y from exampled RP arbitrary wavegen:
# t = np.linspace(0, 1, N)*2*math.pi
# y = np.sin(t) + 1/3*np.sin(3*t) # same overall period as regular sin wave
t, y = util.bounded_frequency_waveform(50, 55, length=N, sample_rate=smpl_rate)
y = util.linear_convert(y) # convert range of waveform to [-1, 1] to properly set ampl
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
rp_s.tx_txt('ACQ:TRig NOW')
time.sleep(1)

# Wait for trigger
while 1:
    rp_s.tx_txt('ACQ:TRig:STAT?') # Get Trigger Status
    print('got status')
    if rp_s.rx_txt() == 'TD': # Triggered?
        break
print('triggered')
## ! OS 2.00 or higher only ! ##
while 1:
    rp_s.tx_txt('ACQ:TRig:FILL?')
    if rp_s.rx_txt() == '1':
        break

##### Analaysis #####
# Read data and plot function for Data Acquisition
pd_data = rp_s.acq_data(chan=1, convert=True) # Volts
speaker_data = rp_s.acq_data(chan=2, convert=True) # Volts
displacement_data, converted_signal, freq = util.displacement_waveform(np.array(speaker_data), smpl_rate, ampl)
y_displ, _, _ = util.displacement_waveform(ampl*y, smpl_rate, ampl)

fig, ax = plt.subplots(nrows=3)

time_data = np.linspace(0, N-1, num=N) / smpl_rate
ax[0].plot(time_data, pd_data, color='blue', label='PD')
ax[0].plot(time_data, speaker_data, color='black', label='Speaker')
ax[0].plot(time_data, ampl*y, label='Original signal')
ax[0].legend()
ax[0].set_ylabel('Amplitude (V)')
ax[0].set_xlabel('Time (s)')

ax[1].plot(freq, np.abs(fft(speaker_data)), color='black', label='Speaker')
ax[1].plot(freq, np.abs(converted_signal), color='green', label='Displacement')
ax[1].loglog()
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('$|\^{V}|$')
ax[1].legend()

ax[2].plot(time_data, displacement_data, color='black', label='Speaker')
ax[2].plot(time_data, y_displ, label='Original signal')
ax[2].set_ylabel('Expected Displacement (Microns)')
ax[2].set_xlabel('Time (s)')
ax[2].legend()
plt.tight_layout()
plt.show()

if store_data:
    # Store data in csv file
    path = "/Users/angelajia/Code/College/SMI/data/"
    filename = f"{datetime.now()}.csv"
    file_path = os.path.join(path, filename)

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time (s)', 'Speaker (V)', 'Speaker (Microns)', 'PD (V)'])
        all_data = np.vstack((time_data, speaker_data, displacement_data, pd_data)).T
        for row in all_data:
            writer.writerow(row)

##### Reset when closing program #####
rp_s.tx_txt('GEN:RST')
rp_s.tx_txt('ACQ:RST')