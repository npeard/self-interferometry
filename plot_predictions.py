import os, sys, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from scipy.stats import pearsonr
import lightning as L
from torch.utils.data import Dataset, DataLoader

import train
import decoder

if __name__ == '__main__':
    # np.random.seed(0x5EED+3)
    if len(sys.argv) == 1:
        """Run functions in this scratch area.
        """
        N = 16384
        SMPL_RATE_DEC1 = 125e6
        decimation = 256
        smpl_rate = SMPL_RATE_DEC1 / decimation
        time_data = np.linspace(0, N - 1, num=N) / smpl_rate

        valid_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\021725_valid_acqfreqs_mixedtones_2kshots_randampl.h5py'
        test_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\acqfreqs_singletone_2kshots_randampl.h5py'
        train_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\021725_train_acqfreqs_mixedtones_10kshots_randampl.h5py'

        model_tag = "fggpzqgv"
        step = 256
        batch_size = 128

        runner = train.TrainingRunner(train_file, valid_file, test_file, step=step)
        model, result = runner.load_model(model_tag)

        modes = ['train', 'valid', 'test']

        for mode in ['test']:
            if mode == 'train':
                iter_loader = iter(runner.train_loader)
            elif mode == 'valid':
                iter_loader = iter(runner.valid_loader_testmode)
            else:  # test mode
                iter_loader = iter(runner.test_loader)
            for i in range(2):
                next(iter_loader)
            batch_val = next(iter_loader)  # batch 3
            batches = [batch_val]
            next(iter_loader)
            batches.append(next(iter_loader))  # batch 5
            next(iter_loader)
            batches.append(next(iter_loader))  # batch 7

            for batch in batches:
                ## Plot results
                inputs = batch[0]
                targets = batch[1]

                print(inputs.shape)
                print(targets.shape)
                outputs_val, _ = model.predict_step(batch, 1, test_mode=True)

                outputs_val = torch.squeeze(outputs_val).cpu().detach().numpy()
                inputs_val = torch.squeeze(inputs).cpu().detach().numpy()
                targets_val = torch.squeeze(targets).cpu().detach().numpy()

                targets_squeezed_val = np.squeeze(targets_val)
                print("targets shape", targets_squeezed_val.shape)
                outputs_squeezed_val = np.squeeze(outputs_val)
                print("preds shape", outputs_squeezed_val.shape)

                fig, ax = plt.subplots(2)
                fig.set_size_inches(8, 6)

                if mode == 'train':
                    inputs_flattened = inputs_val.flatten()
                    ax[0].plot(inputs_flattened)
                    ax[0].set_title('Training PD Trace (scrambled buffer)', fontsize=15)
                    ax[0].set_ylabel('V', fontsize=15)
                    ax[0].set_xticks([])
                    ax[0].tick_params(axis='y', which='major', labelsize=13)
                    # ax[0].set_xlabel('Time (s)')
                    ax[1].plot(targets_squeezed_val, marker='.', label='Target')
                    ax[1].set_title('Velocity', fontsize=15)
                    ax[1].set_ylabel('um/s', fontsize=15)
                    ax[1].set_xlabel('Batch Idx', fontsize=15)

                    ax[1].plot(outputs_squeezed_val, marker='.', label='Pred')
                    ax[1].legend(prop={'size': 12})
                    ax[1].tick_params(axis='both', which='major', labelsize=13)
                else:
                    idx = 0  # use first in batch of 128 shots
                    ax[0].plot(time_data, inputs_val[idx])
                    if mode == 'valid':
                        ax[0].set_title('Validation PD Trace (contiguous buffer)', fontsize=15)
                    elif mode == 'test':
                        ax[0].set_title('Test Shot (contiguous buffer)', fontsize=15)
                    ax[0].set_ylabel('Photodiode signal (V)', fontsize=15)
                    ax[0].set_xticks([])
                    ax[0].tick_params(axis='y', which='major', labelsize=13)
                    # ax[0].set_xlabel('Time (s)')
                    num_groups = outputs_squeezed_val.shape[1]  # 127
                    start_idxs = torch.arange(num_groups) * step

                    ax[1].plot(time_data[start_idxs], targets_squeezed_val[idx], marker='.', label='Target')
                    # ax[1].set_title('', fontsize=15)
                    ax[1].set_ylabel(r'Velocity ($\mu$m/s)', fontsize=15)
                    ax[1].set_xlabel('Time (s)', fontsize=15)

                    ax[1].plot(time_data[start_idxs], outputs_squeezed_val[idx], marker='.', label='Pred')
                    ax[1].legend(prop={'size': 12})
                    ax[1].tick_params(axis='both', which='major', labelsize=13)
                    print(pearsonr(targets_squeezed_val[idx], outputs_squeezed_val[idx]))

                fig.tight_layout()
                plt.show()
    else:
         print("Error: Unsupported number of command-line arguments")