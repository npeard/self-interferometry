#!/usr/bin/env python3

import sys
import time
import matplotlib.pyplot as plt
import redpitaya_scpi as scpi

IP = 'rp-f0c04a.local'
rp_s = scpi.scpi(IP)
print('Connected to ' + IP)

wave_form = 'sine'
freq = 1000000
ampl = 1

# Reset Generation and Acquisition
rp_s.tx_txt('GEN:RST')
rp_s.tx_txt('ACQ:RST')

##### Generation #####
# Function for configuring Source
rp_s.sour_set(1, wave_form, ampl, freq, burst=True, ncyc=3)

##### Acqusition #####
# Function for configuring Acquisition
rp_s.acq_set(dec=1, trig_lvl=0, trig_delay=0)

rp_s.tx_txt('ACQ:START')
time.sleep(1)
rp_s.tx_txt('ACQ:TRig AWG_PE')
rp_s.tx_txt('OUTPUT1:STATE ON')
time.sleep(1)

rp_s.tx_txt('SOUR1:TRig:INT')

# Wait for trigger
while 1:
    rp_s.tx_txt('ACQ:TRig:STAT?') # Get Trigger Status
    if rp_s.rx_txt() == 'TD': # Triggerd?
        break

## ! OS 2.00 or higher only ! ##
while 1:
    rp_s.tx_txt('ACQ:TRig:FILL?')
    if rp_s.rx_txt() == '1':
        break

# Read data and plot
# function for Data Acquisition
data = rp_s.acq_data(1, convert= True)

plt.plot(data)
plt.show()