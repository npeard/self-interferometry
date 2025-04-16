#!/usr/bin/env python3

import sys
import time
import matplotlib.pyplot as plt
import redpitaya_scpi as scpi
import numpy as np
import matplotlib.ticker as ticker

IP = 'rp-f0c04a.local'
rp_s = scpi.scpi(IP)
print('Connected to ' + IP)

wave_form = "SINE"
freq = 120
ampl = 0.1

N = 16384 # Number of samples in buffer
SMPL_RATE_DEC1 = 125e6 # sample rate for decimation=1 in Samples/s (Hz)
decimation = 256# 32
smpl_rate = SMPL_RATE_DEC1//decimation

# Reset Generation and Acquisition
rp_s.tx_txt('GEN:RST')
rp_s.tx_txt('ACQ:RST')

##### Generation #####
# Function for configuring Source
rp_s.sour_set(1, wave_form, ampl, freq)

# Enable output
rp_s.tx_txt('OUTPUT1:STATE ON')
rp_s.tx_txt('SOUR1:TRig:INT')

##### Acqusition #####
# Function for configuring Acquisition
rp_s.acq_set(dec=decimation, trig_delay=0)
rp_s.tx_txt('ACQ:START')
time.sleep(1)
rp_s.tx_txt('ACQ:TRig CH2_PE')
time.sleep(1)

# Wait for trigger
while 1:
    rp_s.tx_txt('ACQ:TRig:STAT?') # Get Trigger Status
    print('got status')
    if rp_s.rx_txt() == 'TD': # Triggerd?
        break
print('triggered')
## ! OS 2.00 or higher only ! ##
while 1:
    rp_s.tx_txt('ACQ:TRig:FILL?')
    if rp_s.rx_txt() == '1':
        break

# Read data and plot
# function for Data Acquisition
time_data = np.linspace(-(N-1)/2, (N-1)/2, num=N) / smpl_rate
pd_data = rp_s.acq_data(chan=1, convert=True)
speaker_data = rp_s.acq_data(chan=2, convert=True)

print(f"vpp: {np.max(speaker_data) - np.min(speaker_data)}")

fig, ax = plt.subplots(nrows=1)
ax.plot(time_data, pd_data, color='blue', label='Observed PD')
ax.plot(time_data, speaker_data, color='black', label='Observed Drive')
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.001))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.0001))

plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')
plt.show()

##### Reset when closing program #####
rp_s.tx_txt('GEN:RST')
rp_s.tx_txt('ACQ:RST')