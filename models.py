#!/usr/bin/env python3

from torch import nn
import torch

class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=7), # Lout = 250, given L = 256
            nn.MaxPool1d(2), # Lout = 125, given L = 250
            nn.Conv1d(16, 32, kernel_size=7), # Lout = 119, given L = 125
            nn.MaxPool1d(2), # Lout = 59, given L = 119
            nn.Conv1d(32, 64, kernel_size=7) # Lout = 53, given L = 59
            nn.MaxPool1d(2), # Lout = 26, given L = 53
            nn.Dropout(0.1), 
            nn.Conv1d(64, 64, kernel_size=7), # Lout = 20, given L = 26
            nn.MaxPool1d(2) # Lout = 10, given L = 20
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(640, 16),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = self.view(640)
        out = self.fc_layers(out)
        return out
        