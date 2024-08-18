import numpy as np
import sys
import torch

import train
import decoder
import models

import datetime

if __name__ == '__main__':
    np.random.seed(0x5EED+3)
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """
        valid_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\valid_max10kHz_30to1kHz_2kshots_dec=256_randampl.h5py'
        train_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\training_max10kHz_30to1kHz_10kshots_dec=256_randampl.h5py'
        test_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\test_max10kHz_30to1kHz_2kshots_dec=256_randampl.h5py'

        print('begin main', datetime.datetime.now())
        step_list = [256, 128, 64, 32] # step sizes for rolling input
        for step in step_list:
            runner = train.TrainingRunner(train_file, valid_file, test_file, step)
            runner.scan_hyperparams()

    else:
        print("Error: Unsupported number of command-line arguments")