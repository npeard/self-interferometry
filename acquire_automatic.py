#!/usr/bin/env python3

import os
import sys
import time
import matplotlib.pyplot as plt
import redpitaya_scpi as scpi
import numpy as np
import math
import util
from datetime import datetime

IP = 'rp-f0c04a.local'
rp_s = scpi.scpi(IP)
print('Connected to ' + IP)

# Set up waveform
wave_form = "ARBITRARY"
freq = 100 # good range 10-200Hz
ampl = 0.3 # good range 0-0.6V

N = 16384 # Number of samples in buffer
decimation = 85
smpl_rate = 125e6//decimation
# t, y from exampled RP arbitrary wavegen:
# t = np.linspace(0, 1, N)*2*math.pi
# y = np.sin(t) + 1/3*np.sin(3*t) # same overall period as regular sin wave
t, y = util.bounded_frequency_waveform(20, 1000, length=N, sample_rate=smpl_rate)
y = util.linear_convert(y) # convert range of waveform to [-1, 1] to properly set ampl
plt.plot(t, y)
plt.show()

# Reset Generation and Acquisition
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
# print(rp_s.get_settings())
rp_s.tx_txt('ACQ:START')
time.sleep(1)
rp_s.tx_txt('ACQ:TRig NOW')
# print(rp_s.get_settings())
time.sleep(1)

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

# Read data and plot
# function for Data Acquisition
print(rp_s.get_settings())
pd_data = rp_s.acq_data(chan=1, convert=True) # Volts
speaker_data = rp_s.acq_data(chan=2, convert=True) # Volts
print("data shape:", np.array(pd_data).shape)
print(rp_s.get_settings())

plt.plot(speaker_data, color="black", label="Speaker")        
plt.plot(pd_data, color="blue", label="PD")
plt.legend()
plt.ylabel('Amplitude [V]')
plt.xlabel('Samples')
plt.show()

# Store data in txt file
path = "/Users/angelajia/Code/College/SMI/data/"
filename = f"{datetime.now()}.txt"
file_path = os.path.join(path, filename)

with open(file_path, 'x') as f:
    # f.write(np.array2string(t, threshold=N+1))
    # f.write("\n" + np.array2string(y, threshold=N+1))
    f.write("\n")
    for x in pd_data:
        f.write(str(x))
        f.write("\n")
    f.write("STARTING SPEAKER DATA\n")
    for y in speaker_data:
        f.write(str(y))
        f.write("\n")