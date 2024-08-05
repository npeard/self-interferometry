import numpy as np
import sys
import torch

import train
import decoder
import models

if __name__ == '__main__':
    np.random.seed(0x5EED+3)
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """
        valid_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\valid_data.h5py'
        train_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\training_data.h5py'
        test_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\test_data.h5py'

        runner = train.TrainingRunner(train_file, valid_file, test_file)
        runner.scan_hyperparams()

    else:
        print("Error: Unsupported number of command-line arguments")