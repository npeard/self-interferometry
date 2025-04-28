#!/usr/bin/env python3
"""
Red Pitaya Manager - A comprehensive class for managing Red Pitaya devices
This module provides a class that encapsulates functionality for data acquisition,
waveform generation, and device management for Red Pitaya devices.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from numpy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import scpi
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import random  # Add import for random module

class RedPitayaManager:
    """
    A comprehensive manager for Red Pitaya devices that encapsulates functionality
    for data acquisition, waveform generation, and device management.
    
    This class provides methods for:
    - Connecting to one or more Red Pitaya devices
    - Configuring and triggering scope acquisitions
    - Generating arbitrary waveforms
    - Saving acquired data
    - Plotting acquired data
    - Running multiple acquisitions with configurable delays
    - Daisy-chaining multiple devices
    """
    
    # Constants
    BUFFER_SIZE = 16384  # Number of samples in buffer
    SAMPLE_RATE_DEC1 = 125e6  # Sample rate for decimation=1 in Samples/s (Hz)
    
    def __init__(self, 
                 ip_addresses: Union[str, List[str]],
                 data_save_path: str = None,
                 blink_on_connect: bool = False):
        """
        Initialize the Red Pitaya Manager.
        
        Args:
            ip_addresses: IP address(es) of the Red Pitaya device(s)
            data_save_path: Path to save acquired data (default: None)
            blink_on_connect: Whether to blink LED 0 on successful connection
        """
        self.devices = []
        self.device_names = []
        
        # Connect to devices
        if isinstance(ip_addresses, str):
            ip_addresses = [ip_addresses]
            
        for i, ip in enumerate(ip_addresses):
            try:
                device = scpi.scpi(ip)
                self.devices.append(device)
                self.device_names.append(f"RP{i+1}")
                print(f"Connected to {ip} as {self.device_names[-1]}")
                
                # Blink LED 0 if requested for connectivity confirmation
                if blink_on_connect:
                    self.blink_led(device_idx=i, led_num=0, num_blinks=3, period=0.5)
                    
            except Exception as e:
                print(f"Failed to connect to {ip}: {e}")
        
        # Set data save path
        if data_save_path:
            self.data_save_path = Path(data_save_path)
        else:
            # Default to a data directory in the project
            self.data_save_path = Path(__file__).parent.parent / "signal_analysis" / "data"
            
        # Ensure the data directory exists
        self.data_save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize plotting
        self.fig = None
        self.axes = None
        self.plot_enabled = True
        
        # Initialize device settings with defaults
        self.settings = {
            "gen_dec": 8192,
            "acq_dec": 256,
            "amplitude": 0.1,
            "wave_form": "ARBITRARY",
            "start_freq": 1,
            "end_freq": 1000,
            "trigger_source": "NOW",
            "trigger_delay": 8192,
            "channels_to_acquire": [1, 2],  # Each device only has channels 1 and 2 by default
            "burst_mode": False,
            "burst_count": 1,
            "burst_period": None,
            "input4": False,  # Set to True for 4-input devices
        }
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.close_all()
    
    def close_all(self):
        """Close all device connections."""
        for device in self.devices:
            try:
                device.close()
            except:
                pass
        self.devices = []
        self.device_names = []
    
    def reset_all(self):
        """Reset generation and acquisition on all devices."""
        for device in self.devices:
            device.tx_txt('GEN:RST')
            device.tx_txt('ACQ:RST')
            
    def configure_daisy_chain(self, primary_idx: int = 0, secondary_indices: List[int] = None):
        """
        Configure devices for daisy chain operation.
        
        Args:
            primary_idx: Index of the primary device
            secondary_indices: Indices of secondary devices
        """
        if not secondary_indices:
            secondary_indices = [i for i in range(len(self.devices)) if i != primary_idx]
            
        if primary_idx >= len(self.devices) or any(idx >= len(self.devices) for idx in secondary_indices):
            raise ValueError("Device index out of range")
            
        # Configure primary unit
        primary = self.devices[primary_idx]
        primary.tx_txt('DAISY:SYNC:TRig ON')
        primary.tx_txt('DAISY:SYNC:CLK ON')
        primary.tx_txt('DAISY:TRIG_O:SOUR ADC')
        primary.tx_txt('DIG:PIN LED5,1')  # LED Indicator
        
        print(f"Primary device ({self.device_names[primary_idx]}) daisy chain configured:")
        print(f"  Trig: {primary.txrx_txt('DAISY:SYNC:TRig?')}")
        print(f"  CLK: {primary.txrx_txt('DAISY:SYNC:CLK?')}")
        print(f"  Sour: {primary.txrx_txt('DAISY:TRIG_O:SOUR?')}")
        
        # Configure secondary units
        for idx in secondary_indices:
            secondary = self.devices[idx]
            secondary.tx_txt('DAISY:SYNC:TRig ON')
            secondary.tx_txt('DAISY:SYNC:CLK ON')
            secondary.tx_txt('DAISY:TRIG_O:SOUR ADC')
            secondary.tx_txt('DIG:PIN LED5,1')  # LED Indicator
            
            print(f"Secondary device ({self.device_names[idx]}) daisy chain configured")
    
    def configure_generation(self, 
                           device_idx: int = 0,
                           channel: int = 1, 
                           wave_form: str = None,
                           amplitude: float = None, 
                           start_freq: int = None,
                           end_freq: int = None,
                           gen_dec: int = None,
                           burst_mode: bool = None,
                           burst_count: int = None,
                           burst_period: int = None):
        """
        Configure signal generation on a device.
        
        Args:
            device_idx: Index of the device to configure
            channel: Output channel (1 or 2)
            wave_form: Waveform type (SINE, SQUARE, TRIANGLE, SAWU, SAWD, PWM, ARBITRARY, DC, DC_NEG)
            amplitude: Signal amplitude
            start_freq: Start frequency for frequency sweep
            end_freq: End frequency for frequency sweep
            gen_dec: Decimation for generation
            burst_mode: Enable/disable burst mode
            burst_count: Number of periods in one burst
            burst_period: Total time of one burst in µs
        """
        # Update settings with provided values
        if wave_form is not None:
            self.settings["wave_form"] = wave_form
        if amplitude is not None:
            self.settings["amplitude"] = amplitude
        if start_freq is not None:
            self.settings["start_freq"] = start_freq
        if end_freq is not None:
            self.settings["end_freq"] = end_freq
        if gen_dec is not None:
            self.settings["gen_dec"] = gen_dec
        if burst_mode is not None:
            self.settings["burst_mode"] = burst_mode
        if burst_count is not None:
            self.settings["burst_count"] = burst_count
        if burst_period is not None:
            self.settings["burst_period"] = burst_period
            
        # Check if device index is valid
        if device_idx >= len(self.devices):
            raise ValueError(f"Device index {device_idx} out of range")
            
        device = self.devices[device_idx]
        
        # Reset generation
        device.tx_txt('GEN:RST')
        
        # Calculate parameters
        gen_smpl_rate = self.SAMPLE_RATE_DEC1 // self.settings["gen_dec"]
        burst_time = self.BUFFER_SIZE / gen_smpl_rate
        freq = 1 / burst_time
        
        # If using arbitrary waveform, create it
        if self.settings["wave_form"] == "ARBITRARY":
            # Calculate valid frequencies based on acquisition sample rate
            acq_smpl_rate = self.SAMPLE_RATE_DEC1 // self.settings["acq_dec"]
            valid_freqs = fftfreq(self.BUFFER_SIZE, d=1/acq_smpl_rate)
            
            # Generate waveform
            t = np.linspace(0, burst_time, self.BUFFER_SIZE, endpoint=False)
            
            # Simple implementation - can be replaced with your bounded_specific_frequencies function
            freqs_in_range = []
            for f in valid_freqs:
                if self.settings["start_freq"] <= abs(f) <= self.settings["end_freq"]:
                    freqs_in_range.append(abs(f))
            
            # Create a sum of sine waves at valid frequencies
            y = np.zeros(self.BUFFER_SIZE)
            for f in freqs_in_range:
                if f > 0:  # Skip DC component
                    y += np.sin(2 * np.pi * f * t)
            
            # Normalize to [-1, 1]
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
                
            # Configure source with arbitrary waveform
            device.sour_set(
                channel, 
                self.settings["wave_form"], 
                self.settings["amplitude"], 
                freq, 
                data=y,
                burst=self.settings["burst_mode"],
                ncyc=self.settings["burst_count"],
                period=self.settings["burst_period"]
            )
        else:
            # Configure source with standard waveform
            device.sour_set(
                channel, 
                self.settings["wave_form"], 
                self.settings["amplitude"], 
                freq,
                burst=self.settings["burst_mode"],
                ncyc=self.settings["burst_count"],
                period=self.settings["burst_period"]
            )
            
        print(f"Generation configured on {self.device_names[device_idx]}, channel {channel}")
        
    def enable_output(self, device_idx: int = 0, channel: int = 1, enable: bool = True):
        """
        Enable or disable output on a device channel.
        
        Args:
            device_idx: Index of the device
            channel: Channel number (1 or 2)
            enable: True to enable, False to disable
        """
        if device_idx >= len(self.devices):
            raise ValueError(f"Device index {device_idx} out of range")
            
        device = self.devices[device_idx]
        
        if enable:
            device.tx_txt(f'OUTPUT{channel}:STATE ON')
            print(f"Output enabled on {self.device_names[device_idx]}, channel {channel}")
        else:
            device.tx_txt(f'OUTPUT{channel}:STATE OFF')
            print(f"Output disabled on {self.device_names[device_idx]}, channel {channel}")
    
    def trigger_generation(self, device_idx: int = 0, channel: int = 1):
        """
        Trigger generation on a device channel.
        
        Args:
            device_idx: Index of the device
            channel: Channel number (1 or 2)
        """
        if device_idx >= len(self.devices):
            raise ValueError(f"Device index {device_idx} out of range")
            
        device = self.devices[device_idx]
        device.tx_txt(f'SOUR{channel}:TRig:INT')
        print(f"Generation triggered on {self.device_names[device_idx]}, channel {channel}")
    
    def configure_acquisition(self, 
                            device_idx: int = None,
                            acq_dec: int = None,
                            trigger_delay: int = None,
                            trigger_source: str = None,
                            channels_to_acquire: List[int] = None):
        """
        Configure acquisition on one or more devices.
        
        Args:
            device_idx: Index of the device to configure. If None, configure all devices.
            acq_dec: Decimation for acquisition
            trigger_delay: Trigger delay in samples
            trigger_source: Trigger source (NOW, CH1_PE, CH1_NE, CH2_PE, CH2_NE, EXT_PE, EXT_NE)
            channels_to_acquire: List of channels to acquire data from
        """
        # Update settings with provided values
        if acq_dec is not None:
            self.settings["acq_dec"] = acq_dec
        if trigger_delay is not None:
            self.settings["trigger_delay"] = trigger_delay
        if trigger_source is not None:
            self.settings["trigger_source"] = trigger_source
        if channels_to_acquire is not None:
            self.settings["channels_to_acquire"] = channels_to_acquire
            
        # Determine which devices to configure
        if device_idx is None:
            devices_to_configure = range(len(self.devices))
        else:
            # Check if device index is valid
            if device_idx >= len(self.devices):
                raise ValueError(f"Device index {device_idx} out of range")
            devices_to_configure = [device_idx]
            
        # Configure each device
        for idx in devices_to_configure:
            device = self.devices[idx]
            device_name = self.device_names[idx]
            
            # Reset acquisition
            device.tx_txt('ACQ:RST')
            
            # Configure acquisition
            device.acq_set(
                dec=self.settings["acq_dec"],
                trig_delay=self.settings["trigger_delay"]
            )
            
            print(f"Acquisition configured on {device_name}")
            
        # Print settings once
        print(f"  Decimation: {self.settings['acq_dec']}")
        print(f"  Trigger delay: {self.settings['trigger_delay']}")
        print(f"  Trigger source: {self.settings['trigger_source']}")
        print(f"  Channels to acquire: {self.settings['channels_to_acquire']}")
    
    def start_acquisition(self, device_idx: int = None, wait_for_trigger: bool = True):
        """
        Start acquisition on one or more devices.
        
        Args:
            device_idx: Index of the device to start. If None, start all devices.
            wait_for_trigger: Whether to wait for trigger and buffer fill
        """
        # Determine which devices to start
        if device_idx is None:
            devices_to_start = range(len(self.devices))
        else:
            # Check if device index is valid
            if device_idx >= len(self.devices):
                raise ValueError(f"Device index {device_idx} out of range")
            devices_to_start = [device_idx]
            
        # Start acquisition on each device
        for idx in devices_to_start:
            device = self.devices[idx]
            device_name = self.device_names[idx]
            
            # Start acquisition
            device.tx_txt('ACQ:START')
            time.sleep(0.4)  # Give some time for the device to start acquisition
            
            # Set trigger
            # For daisy-chained devices, secondary devices should use EXT_NE trigger
            if idx == 0:  # Primary device
                device.tx_txt(f'ACQ:TRig {self.settings["trigger_source"]}')
                print(f"Acquisition started on {device_name} with trigger {self.settings['trigger_source']}")
            else:  # Secondary devices
                device.tx_txt('ACQ:TRig EXT_NE')
                print(f"Acquisition started on {device_name} with trigger EXT_NE (external)")
        
        if wait_for_trigger:
            # Wait for trigger on all devices
            print("Waiting for trigger...")
            
            # Wait for primary device trigger first
            if len(devices_to_start) > 0:
                primary_idx = devices_to_start[0]
                primary_device = self.devices[primary_idx]
                
                while True:
                    primary_device.tx_txt('ACQ:TRig:STAT?')
                    if primary_device.rx_txt() == 'TD':  # Triggered?
                        break
                print(f"Trigger detected on {self.device_names[primary_idx]}")
                
                # Wait for primary device buffer fill
                while True:
                    primary_device.tx_txt('ACQ:TRig:FILL?')
                    if primary_device.rx_txt() == '1':
                        break
                print(f"Buffer filled on {self.device_names[primary_idx]}")
            
            # Wait for secondary devices if any
            for idx in devices_to_start[1:]:
                device = self.devices[idx]
                device_name = self.device_names[idx]
                
                while True:
                    device.tx_txt('ACQ:TRig:STAT?')
                    if device.rx_txt() == 'TD':  # Triggered?
                        break
                print(f"Trigger detected on {device_name}")
                
                # Wait for buffer fill
                while True:
                    device.tx_txt('ACQ:TRig:FILL?')
                    if device.rx_txt() == '1':
                        break
                print(f"Buffer filled on {device_name}")
    
    def get_acquisition_data(self, device_idx: int = None) -> Dict[str, np.ndarray]:
        """
        Get acquisition data from one or more devices.
        
        Args:
            device_idx: Index of the device to acquire from. If None, acquire from all devices.
            
        Returns:
            Dictionary mapping channel names to data arrays
        """
        data = {}
        
        # If device_idx is None, get data from all devices
        if device_idx is None:
            devices_to_read = range(len(self.devices))
        else:
            # Check if device index is valid
            if device_idx >= len(self.devices):
                raise ValueError(f"Device index {device_idx} out of range")
            devices_to_read = [device_idx]
            
        # Get data from each device
        for idx in devices_to_read:
            device = self.devices[idx]
            device_name = self.device_names[idx]
            
            # Each device can only have channels 1 and 2
            for chan in [1, 2]:
                if chan in self.settings["channels_to_acquire"]:
                    try:
                        channel_data = np.array(device.acq_data(chan=chan, convert=True))
                        # Use device name in the channel name for clarity
                        data[f"{device_name} Channel {chan}"] = channel_data
                    except Exception as e:
                        print(f"Error acquiring data from {device_name}, channel {chan}: {e}")
            
            # If device is a 4-input device, get data from channels 3 and 4
            if self.settings["input4"]:
                for chan in [3, 4]:
                    if chan in self.settings["channels_to_acquire"]:
                        try:
                            channel_data = np.array(device.acq_data(chan=chan, convert=True))
                            # Use device name in the channel name for clarity
                            data[f"{device_name} Channel {chan}"] = channel_data
                        except Exception as e:
                            print(f"Error acquiring data from {device_name}, channel {chan}: {e}")
            
        return data
    
    def process_velocity_data(self, speaker_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process speaker data to calculate velocity.
        
        Args:
            speaker_data: Speaker voltage data
            
        Returns:
            Tuple of (velocity data, velocity FFT, frequencies)
        """
        acq_smpl_rate = self.SAMPLE_RATE_DEC1 // self.settings["acq_dec"]
        
        # Make a copy of the speaker data to avoid modifying the original
        vel_data = speaker_data.copy() * 1000  # Convert to microns/s (placeholder)
        
        # Calculate FFT
        vel_converted = fft(vel_data, norm='ortho')
        freqs = fftfreq(speaker_data.shape[0], d=1/acq_smpl_rate)
        
        return vel_data, vel_converted, freqs
    
    def save_data(self, data: Dict[str, np.ndarray], filename: str = None) -> str:
        """
        Save acquisition data to a file.
        
        Args:
            data: Dictionary of data to save
            filename: Filename to save to (default: auto-generated)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            # Generate a filename based on current time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.h5py"
            
        file_path = self.data_save_path / filename
        
        # Import util here to avoid circular imports
        try:
            import signal_analysis.util as util
            util.write_data(file_path, data)
            print(f"Data saved to {file_path}")
        except ImportError:
            print("Warning: signal_analysis.util module not found. Data not saved.")
            print(f"Would have saved to {file_path}")
            
        return str(file_path)
    
    def setup_plot(self, num_rows=4):
        """
        Set up the plot with a 2-column layout.
        
        Args:
            num_rows: Number of rows in the plot (should be 4 for the 4 channels)
        """
        if not self.plot_enabled:
            return
            
        # Close any existing plot
        if self.fig is not None:
            plt.close(self.fig)
            
        # Create a new figure with 2 columns: raw signals and FFTs
        self.fig, self.axes = plt.subplots(num_rows, 2, figsize=(12, 8), sharex='col')
        
        # Set up the figure
        plt.tight_layout()
        
    def update_plot(self, data: Dict[str, np.ndarray], vel_data: np.ndarray = None, 
                   vel_fft: np.ndarray = None, freqs: np.ndarray = None):
        """
        Update the plot with new data using a 2-column layout.
        
        Args:
            data: Dictionary of channel data
            vel_data: Velocity data
            vel_fft: Velocity FFT
            freqs: Frequencies for FFT plot
        """
        if not self.plot_enabled or self.fig is None:
            return
            
        # First, remove any existing twin axes to start fresh
        for ax in self.fig.axes[:]:
            if ax not in self.axes.flatten():
                self.fig.delaxes(ax)
                
        # Clear all main axes
        for ax in self.axes.flatten():
            ax.clear()
            
        # Calculate time data
        acq_smpl_rate = self.SAMPLE_RATE_DEC1 // self.settings["acq_dec"]
        time_data = np.linspace(0, (self.BUFFER_SIZE-1)/acq_smpl_rate, self.BUFFER_SIZE)
        
        # Filter to only include raw channel data (not processed data)
        channel_data = {k: v for k, v in data.items() if "Channel" in k}
        
        # Define channel order
        channel_order = [
            "RP1 Channel 1",  # Speaker Drive Voltage
            "RP1 Channel 2",
            "RP2 Channel 1",
            "RP2 Channel 2"
        ]
        
        # Colors for each channel
        colors = ['blue', 'red', 'green', 'purple']
        
        # Calculate FFT frequencies once
        if freqs is None and len(channel_data) > 0:
            # Get the length of any channel data
            any_data = next(iter(channel_data.values()))
            n = len(any_data)
            freqs = np.fft.fftfreq(n, 1/acq_smpl_rate)
            
        # Only plot positive frequencies for FFT
        pos_idx = None
        pos_freqs = None
        if freqs is not None:
            pos_idx = freqs >= 0
            pos_freqs = freqs[pos_idx]
        
        # Plot each channel and its FFT
        for i, channel_name in enumerate(channel_order):
            if channel_name in channel_data and i < len(self.axes):
                # Get the channel data
                channel_values = channel_data[channel_name]
                
                # Create appropriate label
                if channel_name == "RP1 Channel 1":
                    label = "Speaker Drive Voltage"
                else:
                    label = channel_name
                
                # Plot the raw data in the left column
                self.axes[i, 0].plot(time_data, channel_values, color=colors[i % len(colors)])
                self.axes[i, 0].set_ylabel('Amplitude (V)', color=colors[i % len(colors)])
                self.axes[i, 0].tick_params(axis='y', labelcolor=colors[i % len(colors)])
                self.axes[i, 0].set_title(label)
                
                # Add velocity twin axis for the speaker drive voltage
                if channel_name == "RP1 Channel 1" and vel_data is not None:
                    ax_vel = self.axes[i, 0].twinx()
                    ax_vel.plot(time_data, vel_data, 'k-', alpha=0.7)
                    ax_vel.set_ylabel('Velocity (Microns/s)', color='black')
                    ax_vel.tick_params(axis='y', labelcolor='black')
                
                # Plot the FFT in the right column
                if freqs is not None and pos_idx is not None:
                    # Calculate FFT
                    channel_fft_complex = fft(channel_values, norm='ortho')
                    channel_fft_mag = np.abs(channel_fft_complex)
                    channel_fft_phase = np.angle(channel_fft_complex, deg=True)
                    
                    # Create twin axis for phase
                    ax_phase = self.axes[i, 1].twinx()
                    
                    # Plot magnitude on primary y-axis (log scale)
                    self.axes[i, 1].semilogy(pos_freqs, channel_fft_mag[pos_idx], 
                                           color=colors[i % len(colors)], label='Magnitude')
                    self.axes[i, 1].set_ylabel('Magnitude (log)', color=colors[i % len(colors)])
                    self.axes[i, 1].tick_params(axis='y', labelcolor=colors[i % len(colors)])
                    
                    # Plot phase on secondary y-axis with alpha=0.3
                    ax_phase.plot(pos_freqs, channel_fft_phase[pos_idx], 'r-', 
                                label='Phase', alpha=0.3)
                    ax_phase.set_ylabel('Phase (°)', color='red')
                    ax_phase.tick_params(axis='y', labelcolor='red')
                    ax_phase.set_ylim(-180, 180)
                    
                    # Set title for FFT plot
                    self.axes[i, 1].set_title(f"{label} FFT")
                    
                    # Add grid to FFT plot
                    self.axes[i, 1].grid(True, which="both", ls="-", alpha=0.5)
        
        # Set common x-axis labels
        for i in range(len(self.axes)):
            # Only add x-axis label to bottom plots
            if i == len(self.axes) - 1:
                self.axes[i, 0].set_xlabel('Time (s)')
                self.axes[i, 1].set_xlabel('Frequency (Hz)')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to update the plot
    
    def show_plot(self, block=False):
        """
        Show the plot.
        
        Args:
            block: Whether to block execution until the plot window is closed
        """
        if not self.plot_enabled or self.fig is None:
            return
            
        plt.tight_layout()
        plt.draw()
        if block:
            plt.show(block=True)
        else:
            plt.pause(0.001)  # Small pause to update the plot
    
    def run_one_shot(self, 
                    device_idx: int = 0,
                    start_freq: int = None, 
                    end_freq: int = None, 
                    amplitude: float = None,
                    gen_dec: int = None, 
                    acq_dec: int = None,
                    store_data: bool = False, 
                    plot_data: bool = True,
                    filename: str = None,
                    timeout: int = 5,
                    block_plot: bool = True) -> Dict[str, np.ndarray]:
        """
        Run one complete acquisition cycle.
        
        Args:
            device_idx: Index of the device to use as primary (for generation)
            start_freq: Start frequency for frequency sweep
            end_freq: End frequency for frequency sweep
            amplitude: Signal amplitude
            gen_dec: Decimation for generation
            acq_dec: Decimation for acquisition
            store_data: Whether to store data
            plot_data: Whether to plot data
            filename: Filename to save data to
            timeout: Timeout in seconds for waiting for triggers
            block_plot: Whether to block the plot, default True
            
        Returns:
            Dictionary of acquired data
        """
        # Update settings
        if start_freq is not None:
            self.settings["start_freq"] = start_freq
        if end_freq is not None:
            self.settings["end_freq"] = end_freq
        if amplitude is not None:
            self.settings["amplitude"] = amplitude
        if gen_dec is not None:
            self.settings["gen_dec"] = gen_dec
        if acq_dec is not None:
            self.settings["acq_dec"] = acq_dec
        self.plot_enabled = plot_data
        
        # Configure and run acquisition
        self.reset_all()
        
        # Check if we have devices
        if len(self.devices) == 0:
            print("No devices connected")
            return {}
            
        # Get the primary device
        primary_device = self.devices[device_idx]
        primary_name = self.device_names[device_idx]
        
        # Identify secondary devices
        secondary_devices = []
        secondary_names = []
        for i, device in enumerate(self.devices):
            if i != device_idx:
                secondary_devices.append(device)
                secondary_names.append(self.device_names[i])
        
        # Reset and configure all devices
        print(f"Configuring {len(self.devices)} device(s)...")
        
        # Configure generation on primary device
        self.configure_generation(device_idx=device_idx)
        
        # Configure acquisition on all devices
        for i, device in enumerate(self.devices):
            device_name = self.device_names[i]
            device.tx_txt('ACQ:RST')
            device.acq_set(
                dec=self.settings["acq_dec"],
                trig_delay=self.settings["trigger_delay"]
            )
            print(f"Acquisition configured on {device_name}")
        
        # Start acquisition on secondary devices first
        for device, name in zip(secondary_devices, secondary_names):
            device.tx_txt('ACQ:START')
            time.sleep(0.2)
            device.tx_txt('ACQ:TRig EXT_NE')
            print(f"Acquisition started on {name} with trigger EXT_NE")
        
        # Start acquisition on primary device
        primary_device.tx_txt('ACQ:START')
        time.sleep(0.2)
        primary_device.tx_txt('ACQ:TRig CH1_PE')
        print(f"Acquisition started on {primary_name} with trigger CH2_PE")
        
        # Enable output on primary device
        self.enable_output(device_idx=device_idx)
        
        # Trigger generation on primary device
        self.trigger_generation(device_idx=device_idx)
        
        print("ACQ start")
        
        # Wait for primary device trigger with timeout
        print(f"Waiting for {primary_name} trigger (timeout: {timeout}s)...")
        start_time = time.time()
        triggered = False
        
        while time.time() - start_time < timeout:
            primary_device.tx_txt('ACQ:TRig:STAT?')
            status = primary_device.rx_txt()
            print(f"Trigger status: {status}")
            
            if status == 'TD':  # Triggered
                triggered = True
                break
            time.sleep(0.5)  # Check every half second
            
        if not triggered:
            print(f"Timeout waiting for {primary_name} trigger")
            # Try to recover by forcing a trigger
            primary_device.tx_txt('ACQ:TRig NOW')
            time.sleep(1)
        else:
            print(f"Trigger detected on {primary_name}")
        
        # Wait for primary device buffer fill with timeout
        print(f"Waiting for {primary_name} buffer to fill (timeout: {timeout}s)...")
        start_time = time.time()
        filled = False
        
        while time.time() - start_time < timeout:
            primary_device.tx_txt('ACQ:TRig:FILL?')
            fill_status = primary_device.rx_txt()
            print(f"Fill status: {fill_status}")
            
            if fill_status == '1':  # Buffer filled
                filled = True
                break
            time.sleep(0.5)  # Check every half second
            
        if not filled:
            print(f"Timeout waiting for {primary_name} buffer to fill")
        else:
            print(f"Buffer filled on {primary_name}")
        
        # Wait for secondary devices with timeout
        for device, name in zip(secondary_devices, secondary_names):
            # Wait for trigger
            print(f"Waiting for {name} trigger (timeout: {timeout}s)...")
            start_time = time.time()
            triggered = False
            
            while time.time() - start_time < timeout:
                device.tx_txt('ACQ:TRig:STAT?')
                status = device.rx_txt()
                print(f"Trigger status: {status}")
                
                if status == 'TD':  # Triggered
                    triggered = True
                    break
                time.sleep(0.5)  # Check every half second
                
            if not triggered:
                print(f"Timeout waiting for {name} trigger")
                # Try to recover by forcing a trigger
                device.tx_txt('ACQ:TRig NOW')
                time.sleep(1)
            else:
                print(f"Trigger detected on {name}")
            
            # Wait for buffer fill
            print(f"Waiting for {name} buffer to fill (timeout: {timeout}s)...")
            start_time = time.time()
            filled = False
            
            while time.time() - start_time < timeout:
                device.tx_txt('ACQ:TRig:FILL?')
                fill_status = device.rx_txt()
                print(f"Fill status: {fill_status}")
                
                if fill_status == '1':  # Buffer filled
                    filled = True
                    break
                time.sleep(0.5)  # Check every half second
                
            if not filled:
                print(f"Timeout waiting for {name} buffer to fill")
            else:
                print(f"Buffer filled on {name}")
        
        # Get data from all devices
        data = {}
        
        # Get data from primary device
        try:
            for chan in [1, 2]:
                if chan in self.settings["channels_to_acquire"]:
                    try:
                        channel_data = np.array(primary_device.acq_data(chan=chan, convert=True))
                        data[f"{primary_name} Channel {chan}"] = channel_data
                        print(f"Successfully acquired data from {primary_name} Channel {chan}")
                    except Exception as e:
                        print(f"Error acquiring data from {primary_name}, channel {chan}: {e}")
        except Exception as e:
            print(f"Error during data acquisition from primary device: {e}")
        
        # Get data from secondary devices
        for device, name in zip(secondary_devices, secondary_names):
            try:
                for chan in [1, 2]:
                    if chan in self.settings["channels_to_acquire"]:
                        try:
                            channel_data = np.array(device.acq_data(chan=chan, convert=True))
                            data[f"{name} Channel {chan}"] = channel_data
                            print(f"Successfully acquired data from {name} Channel {chan}")
                        except Exception as e:
                            print(f"Error acquiring data from {name}, channel {chan}: {e}")
            except Exception as e:
                print(f"Error during data acquisition from {name}: {e}")
        
        # Process velocity data if speaker data is available
        vel_data_dict = {}
        vel_fft_dict = {}
        freqs = None
        
        # Create a complete data dictionary for saving
        save_data = data.copy()
        
        # Process velocity data for primary device's Channel 1 (speaker drive)
        speaker_key = f"{primary_name} Channel 1"
        if speaker_key in data:  # Channel 1 is now the speaker drive
            vel_data, vel_fft, freqs = self.process_velocity_data(data[speaker_key])
            
            # Store velocity data
            vel_data_dict[primary_name] = vel_data
            vel_fft_dict[primary_name] = vel_fft
            
            # Add to save data
            save_data[f"{primary_name} Velocity (Microns/s)"] = vel_data
        
        # Save data if requested
        if store_data:
            self.save_data(save_data, filename)
        
        # Plot data if requested
        if plot_data and data:  # Only plot if we have data
            # Use 4 rows for the 4 channels
            num_rows = 4
                
            # Only set up the plot if it doesn't exist yet
            if self.fig is None or self.axes.shape != (num_rows, 2):
                self.setup_plot(num_rows=num_rows)
            
            # Use the primary device's velocity data for plotting if available
            primary_vel_data = vel_data_dict.get(primary_name)
            primary_vel_fft = vel_fft_dict.get(primary_name)
            
            # Update the plot with new data
            self.update_plot(data, primary_vel_data, primary_vel_fft, freqs)
            
            # Show the plot without blocking
            self.show_plot(block=block_plot)
        
        return save_data
    
    def run_multiple_shots(self, 
                          num_shots: int,
                          device_idx: int = 0,
                          delay_between_shots: float = 0.5,
                          store_data: bool = True,
                          plot_data: bool = False,
                          random_amplitude: bool = False,
                          filename_prefix: str = None) -> List[Dict[str, np.ndarray]]:
        """
        Run multiple acquisition cycles.
        
        Args:
            num_shots: Number of shots to run
            device_idx: Index of the device to use as primary
            delay_between_shots: Delay between shots in seconds
            store_data: Whether to store data
            plot_data: Whether to plot data
            random_amplitude: Whether to use random amplitude
            filename_prefix: Prefix for filenames
            
        Returns:
            List of dictionaries of acquired data
        """
        print(f"Starting {num_shots} acquisition cycles")
        start_time = datetime.now()
        print(f"Start time: {start_time.strftime('%H:%M:%S.%f')}")
        
        all_data = []
        
        # Set up the plot once if plotting is enabled
        if plot_data:
            # Determine number of rows based on data available
            # We'll assume 4 rows for now (raw, velocity, and 2 FFT plots)
            num_rows = 4
            self.setup_plot(num_rows=num_rows)
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)  # Show the plot without blocking
        
        try:
            for i in range(num_shots):
                print(f"Shot {i}/{num_shots} at {datetime.now().strftime('%H:%M:%S.%f')}")
                
                # Randomize amplitude if requested
                amplitude = None
                if random_amplitude:
                    amplitude = random.uniform(0.05, 0.5)
                
                # Generate filename if storing data
                filename = None
                if store_data and filename_prefix:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{filename_prefix}_{timestamp}_{i}.npz"
                
                # Run one shot
                shot_data = self.run_one_shot(
                    device_idx=device_idx,
                    amplitude=amplitude,
                    store_data=store_data,
                    plot_data=plot_data,
                    filename=filename,
                    block_plot=False
                )
                
                all_data.append(shot_data)
                
                # Wait between shots
                if i < num_shots - 1:  # Don't wait after the last shot
                    time.sleep(delay_between_shots)
        
        except KeyboardInterrupt:
            print("Acquisition interrupted by user")
        
        finally:
            # Make sure to reset devices
            self.reset_all()
            
            # Turn off interactive mode if it was enabled
            if plot_data:
                plt.ioff()
                plt.close(self.fig)  # Close the figure to clean up resources
                self.fig = None
                self.axes = []
            
            end_time = datetime.now()
            print(f"Completed {len(all_data)} acquisition cycles")
            print(f"End time: {end_time.strftime('%H:%M:%S.%f')}")
            
            return all_data
    
    def blink_led(self, device_idx: int = 0, led_num: int = 0, num_blinks: int = 3, period: float = 1.0):
        """
        Blink an LED on the Red Pitaya device for connectivity troubleshooting.
        
        Args:
            device_idx: Index of the device to blink LED on
            led_num: LED number to blink (0-7)
            num_blinks: Number of times to blink
            period: Period of each blink in seconds
        """
        if len(self.devices) <= device_idx:
            print(f"No device at index {device_idx}")
            return
            
        device = self.devices[device_idx]
        device_name = self.device_names[device_idx]
        
        print(f"Blinking LED[{led_num}] on {device_name} {num_blinks} times")
        
        try:
            for i in range(num_blinks):
                # Turn LED on
                device.tx_txt(f'DIG:PIN LED{led_num},1')
                time.sleep(period/2.0)
                
                # Turn LED off
                device.tx_txt(f'DIG:PIN LED{led_num},0')
                time.sleep(period/2.0)
                
            print(f"Finished blinking LED on {device_name}")
        except Exception as e:
            print(f"Error blinking LED on {device_name}: {e}")

# Example usage
if __name__ == "__main__":
    # # Connect to a single Red Pitaya device
    # rp_manager = RedPitayaManager("rp-f0c04a.local")
    
    # # Run a single acquisition
    # rp_manager.run_one_shot(
    #     start_freq=1,
    #     end_freq=1000,
    #     amplitude=0.1,
    #     gen_dec=8192,
    #     acq_dec=256,
    #     store_data=False,
    #     plot_data=True
    # )
    
    # Connect to multiple Red Pitaya devices
    rp_manager = RedPitayaManager(["rp-f0c04a.local", "rp-f0c026.local"], blink_on_connect=True)
    
    # Configure daisy chain
    rp_manager.configure_daisy_chain()

    # # Run a single acquisition
    # rp_manager.run_one_shot(
    #     start_freq=1,
    #     end_freq=1000,
    #     amplitude=0.1,
    #     gen_dec=8192,
    #     acq_dec=256,
    #     store_data=False,
    #     plot_data=True
    # )
    
    # Run multiple acquisitions
    rp_manager.run_multiple_shots(
        num_shots=10,
        delay_between_shots=1.0,
        store_data=False,
        plot_data=True,
        random_amplitude=True
    )
