#!/usr/bin/env python3
"""Daisy chain example for Red Pitaya"""

import time
import redpitaya_scpi as scpi
import matplotlib.pyplot as plt
import numpy as np

# Connect OUT1 primary with IN1 primary and IN1 secondary


IP_PRIM = 'rp-f0c04a.local'   # IP Test OS Red Pitaya
IP_SEC = 'rp-f0c026.local'

rp_prim = scpi.scpi(IP_PRIM)
rp_sec = scpi.scpi(IP_SEC)

print("Program Start")

rp_prim.tx_txt('GEN:RST')
rp_prim.tx_txt('ACQ:RST')

rp_sec.tx_txt('GEN:RST')
rp_sec.tx_txt('ACQ:RST')

###### ENABLING THE DAISY CHAIN PRIMARY UNIT ######

rp_prim.tx_txt('DAISY:SYNC:TRig ON')    #! OFF (without sync)
rp_prim.tx_txt('DAISY:SYNC:CLK ON')
rp_prim.tx_txt('DAISY:TRIG_O:SOUR ADC')

rp_prim.tx_txt('DIG:PIN LED5,1')            # LED Indicator

time.sleep(0.2)

print(f"Trig: {rp_prim.txrx_txt('DAISY:SYNC:TRig?')}")
print(f"CLK: {rp_prim.txrx_txt('DAISY:SYNC:CLK?')}")
print(f"Sour: {rp_prim.txrx_txt('DAISY:TRIG_O:SOUR?')}\n")

###### ENABLING THE DAISY CHAIN SECONDARY UNIT ######

rp_sec.tx_txt('DAISY:SYNC:TRig ON')  #! OFF (without sync)
rp_sec.tx_txt('DAISY:SYNC:CLK ON')
rp_sec.tx_txt('DAISY:TRIG_O:SOUR ADC')     # Ext trigger will trigger the ADC

rp_sec.tx_txt('DIG:PIN LED5,1')             # LED Indicator

print("Start generator\n")


### Generation ### - Primary unit
rp_prim.sour_set(1, "sine", 1, 100000)
rp_prim.tx_txt('OUTPUT1:STATE ON')

### Aquisition ###

# Primary unit
rp_prim.acq_set(dec = 2,
                trig_lvl = 0.5,
                trig_delay = 7000)


# Secondary unit
rp_sec.acq_set(dec = 2,
               trig_lvl = 0.5,
               trig_delay = 7000)


rp_sec.tx_txt('ACQ:START')
time.sleep(0.2)                           # Not necessary
rp_sec.tx_txt('ACQ:TRig EXT_NE')          #! CH1_PE (without sync trig) EXT_NE (with sync trig)
                                          # If not synchronised make sure no signal arrives before both units are set up

rp_prim.tx_txt('ACQ:START')
time.sleep(0.2)
rp_prim.tx_txt('ACQ:TRig CH1_PE')

time.sleep(1)                             # Symulating a trigger after one second
rp_prim.tx_txt('SOUR1:TRig:INT')

print("ACQ start")

while 1:
    # Get Trigger Status
    if rp_prim.txrx_txt('ACQ:TRig:STAT?') == 'TD':               # Triggerd?
        break
print("Trigger primary condition met.")

## ! OS 2.00 or higher only ! ##
while 1:
    if rp_prim.txrx_txt('ACQ:TRig:FILL?') == '1':
        break
print("Buffer primary filled.")

while 1:
    # Get Trigger Status
    if rp_sec.txrx_txt('ACQ:TRig:STAT?') == 'TD':               # Triggerd?
        break
print("Trigger secondary condition met.")

## ! OS 2.00 or higher only ! ##
while 1:
    if rp_sec.txrx_txt('ACQ:TRig:FILL?') == '1':
        break
print("Buffer secondary filled.")


# Read data and plot
rp_prim.tx_txt('ACQ:SOUR1:DATA?')               # Read full buffer primary (source 1)
data_string1 = rp_prim.rx_txt()                 # data into a string

rp_sec.tx_txt('ACQ:SOUR1:DATA?')                # Read full buffer secondary (source 1)
data_string2 = rp_sec.rx_txt()

# Display both buffers at once
n = 2
buff = np.zeros((n,16384))

# Remove brackets and empty spaces + string => float
data_string1 = data_string1.strip('{}\n\r').replace("  ", "").split(',')
data_string2 = data_string2.strip('{}\n\r').replace("  ", "").split(',')
# Transform data into data series
buff[0, :] = list(map(float, data_string1))
buff[1, :] = list(map(float, data_string2))


######## PLOTTING THE DATA #########
fig, axs = plt.subplots(n, sharex = True)               # plot the data (n subplots)
fig.suptitle("Measurements P1 S2")

for i in range(0,n,1):                                  # plotting the acquired buffers
    axs[i].plot(buff[i])

plt.show()

rp_prim.close()
rp_sec.close()