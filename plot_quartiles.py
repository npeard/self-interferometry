import os, sys, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch import optim, nn
import lightning as L
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import LogNorm

import train
import decoder

if __name__ == '__main__':
    if len(sys.argv) == 1:
        N = 16384
        SMPL_RATE_DEC1 = 125e6
        decimation = 256
        smpl_rate = SMPL_RATE_DEC1 / decimation
        time_data = np.linspace(0, N - 1, num=N) / smpl_rate

        valid_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\val_1to1kHz_invertspectra_trigdelay8192_sleep100ms_2kx1shots_dec=256_8192_randampl.h5py'
        test_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\test_1to1kHz_invertspectra_trigdelay8192_sleep100ms_2kx1shots_dec=256_8192_randampl.h5py'
        train_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\train_1to1kHz_invertspectra_trigdelay8192_sleep100ms_10kx1shots_dec=256_8192_randampl.h5py'

        model_tag = "z4dscrhq" #"qn4eo8zu"#"z4dscrhq"
        step = 64
        group_size = 256

        runner = train.TrainingRunner(train_file, valid_file, test_file, step=step)
        model, result = runner.load_model(model_tag)

        modes = ['train', 'valid', 'test']

        # only works for datasets using test_mode=True
        iter_loader = iter(runner.test_loader)

        preds = torch.empty(0)
        truths = torch.empty(0)
        inputs = torch.empty(0)

        num_batches = 10  # ceil[2k/128]
        for i in range(num_batches):
            print(i)
            batch = next(iter_loader)
            input_buffers = torch.squeeze(batch[0])
            targets = torch.squeeze(batch[1])
            outputs = torch.squeeze(model.predict_step(batch, 1, test_mode=True)[0])  # store only y_hat
            # print("Input shape", inputs.shape)  # [batch_size, 16384]
            # print("Target shape", targets.shape) # [batch_size, num_groups]
            # print("Pred shape", outputs.shape)  # [batch_size, num_groups]

            preds = torch.cat((preds, outputs), dim=0)
            truths = torch.cat((truths, targets), dim=0)
            inputs = torch.cat((inputs, input_buffers), dim=0)

        preds = preds.cpu().detach().numpy()  # [num_batches * batch_size, num_groups]
        truths = truths.cpu().detach().numpy()  # [num_batches * batch_size, num_groups]
        inputs = inputs.cpu().detach().numpy()  # [num_batches * batch_size, 16384]

        print("preds shape", preds.shape)
        print("truths shape", truths.shape)
        print("inputs shape", inputs.shape)

        losses = np.mean((preds - truths)**2, axis=1)  # avg_mse_loss per buffer, [num_batches * batch_size]
        print("losses shape", losses.shape)

        sorted_idxs = np.argsort(losses)
        # all sorted arrays should have same shape as their unsorted originals
        sorted_losses = losses[sorted_idxs]
        print("sorted losses shape", sorted_losses.shape)
        sorted_truths = truths[sorted_idxs]
        print("sorted truths shape", sorted_truths.shape)
        sorted_preds = preds[sorted_idxs]
        print("sorted preds shape", sorted_preds.shape)
        sorted_inputs = inputs[sorted_idxs]
        print("sorted inputs shape", sorted_inputs.shape)

        num_buffers = sorted_losses.shape[0]
        q2_start_idx = num_buffers // 4  # End of first quartile
        q3_start_idx = num_buffers // 2  # End of second quartile
        q4_start_idx = 3 * num_buffers // 4  # End of third quartile

        first_quartile = {
            "losses": sorted_losses[:q2_start_idx],
            "truths": sorted_truths[:q2_start_idx],
            "preds": sorted_preds[:q2_start_idx],
            "inputs": sorted_inputs[:q2_start_idx]
        }

        fourth_quartile = {
            "losses": sorted_losses[q4_start_idx:],
            "truths": sorted_truths[q4_start_idx:],
            "preds": sorted_preds[q4_start_idx:],
            "inputs": sorted_inputs[q4_start_idx:]
        }

        print("First quartile buffer count:", first_quartile["inputs"].shape)
        print("Fourth quartile buffer count:", fourth_quartile["inputs"].shape)

        def plot_quartile_examples(quartiles, num_exs=4):
            fig, axes = plt.subplots(num_exs * 2, 2, figsize=(15, num_exs * 5),
                                     gridspec_kw={'height_ratios': [1, 2] * num_exs})

            time_data = np.linspace(0, N - 1, num=N) / smpl_rate
            num_groups = truths.shape[1]
            start_idxs = torch.arange(num_groups) * step
            for i in range(num_exs):
                for j, quartile in enumerate(quartiles):
                    # Plot PD trace
                    axes[i*2, j].plot(time_data, quartile['inputs'][i], color='blue')
                    axes[i*2, j].set_ylabel('V')
                    axes[i*2, j].set_xticks([])

                    # Plot velocities
                    axes[i*2 + 1, j].plot(time_data[start_idxs], quartile['truths'][i],
                                            label='Target', color='blue',
                                            marker='o', markersize=2)
                    axes[i*2 + 1, j].plot(time_data[start_idxs], quartile['preds'][i],
                                            label=f'Pred (Avg MSE: {quartile["losses"][i]:.2f})', color='orange', marker='o',
                                            markersize=2)
                    axes[i*2 + 1, j].set_ylabel('µm/s')
                    axes[i*2 + 1, j].legend()

                    if i == num_exs - 1:  # a bottom plot
                        axes[i*2 + 1, j].set_xlabel('Time (s)')
                    else:
                        axes[i*2 + 1, j].set_xticks([])

            fig.text(0.25, 0.98, '1st Quartile (Best Performance)', ha='center', fontsize=15)
            fig.text(0.75, 0.98, '4th Quartile (Worst Performance)', ha='center', fontsize=15)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for suptitles
            plt.show()

        # Call the function with the first and fourth quartile data
        plot_quartile_examples(
            quartiles=[
                first_quartile,  # Best quartile data
                fourth_quartile  # Worst quartile data
            ]
        )
    else:
        print("Error: Unsupported number of command-line arguments")