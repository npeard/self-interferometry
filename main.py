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
        valid_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\valid.h5py'
        test_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\test.h5py'
        train_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\train.h5py'
        
        # train_file = "/Users/nolanpeard/Desktop/SMI_sim/train_double.h5"
        # valid_file = "/Users/nolanpeard/Desktop/SMI_sim/valid_double.h5"
        # test_file = "/Users/nolanpeard/Desktop/SMI_sim/test_double.h5"

        # print('begin main', datetime.datetime.now())
        step_list = [256]  # [256, 128] #, 64, 32]  # step sizes for rolling input
        for step in step_list:
            runner = train.TrainingRunner(train_file, valid_file, test_file, step)
            runner.scan_hyperparams()

        # runner = train.TrainingRunner(train_file, valid_file, test_file,
        #                               step=256)
        #runner.plot_predictions(model_name="CNN", model_id="tdwhpu2l")
        # runner.plot_predictions(model_name="CNN", model_id="e8vpuie1")

    else:
        print("Error: Unsupported number of command-line arguments")