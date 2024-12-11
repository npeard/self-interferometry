#!/usr/bin/env python3

from torch import nn
import torch

act_fn_by_name = {'LeakyReLU': nn.LeakyReLU(), 'ReLU': nn.ReLU()}

class CNN(nn.Module):
    def __init__(self, input_size, output_size, ch_in=1, activation='LeakyReLU'):
        super(CNN, self).__init__()
        self.ch_in = ch_in
        self.conv_layers = nn.Sequential(
            nn.Conv1d(ch_in, 16, kernel_size=7),  # Lout = 250, given L = 256
            act_fn_by_name[activation], 
            nn.MaxPool1d(2),  # Lout = 125, given L = 250
            nn.Conv1d(16, 32, kernel_size=7),  # Lout = 119, given L = 125
            act_fn_by_name[activation], 
            nn.MaxPool1d(2),  # Lout = 59, given L = 119
            nn.Conv1d(32, 64, kernel_size=7),  # Lout = 53, given L = 59
            act_fn_by_name[activation], 
            nn.MaxPool1d(2),  # Lout = 26, given L = 53
            nn.Dropout(0.1), 
            nn.Conv1d(64, 64, kernel_size=7),  # Lout = 20, given L = 26
            act_fn_by_name[activation], 
            nn.MaxPool1d(2)  # Lout = 10, given L = 20
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(640, 16),
            act_fn_by_name[activation],
            nn.Linear(16, output_size)
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        # print(f"post conv out size: {out.size()}")  # [128, 64, 10]
        out = out.view(out.size(0), self.ch_in, -1)
        # print(f"post conv out reshaped size: {out.size()}")  # confirmed [128, 1, 640]
        out = self.fc_layers(out)  # expect out [128, 1, 1]
        # print(f"post fc out size: {out.size()}") # confirmed: [128, 1, 1]
        return out
        