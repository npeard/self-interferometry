import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import train
from matplotlib.colors import LogNorm

if __name__ == '__main__':
    if len(sys.argv) == 1:
        N = 16384
        SMPL_RATE_DEC1 = 125e6
        decimation = 256
        smpl_rate = SMPL_RATE_DEC1 / decimation
        time_data = np.linspace(0, N - 1, num=N) / smpl_rate

        valid_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\021725_valid_acqfreqs_mixedtones_2kshots_randampl.h5py'
        test_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\021725_test_acqfreqs_mixedtones_2kshots_randampl.h5py'
        train_file = 'C:\\Users\\aj14\\Desktop\\SMI\\data\\021725_train_acqfreqs_mixedtones_10kshots_randampl.h5py'

        model_tag = '849tnvok'  # "qn4eo8zu"#"z4dscrhq"
        step = 128
        group_size = 256

        runner = train.TrainingRunner(train_file, valid_file, test_file, step=step)
        model, result = runner.load_model(model_tag)

        modes = ['train', 'valid', 'test']

        # only works for datasets using test_mode=True
        for mode in ['test']:
            if mode == 'valid':
                iter_loader = iter(runner.valid_loader_testmode)
            elif mode == 'test':  # test mode
                iter_loader = iter(runner.test_loader)
            else:
                raise ValueError('Invalid mode')

            preds = torch.empty(0)
            truths = torch.empty(0)
            input_segments = torch.empty(0)

            num_batches = 16  # ceil[2k/128]
            for i in range(num_batches):
                print(i)
                batch = next(iter_loader)
                inputs = torch.squeeze(batch[0])
                targets = torch.squeeze(batch[1])
                outputs = torch.squeeze(
                    model.predict_step(batch, 1, test_mode=True)[0]
                )  # store only y_hat
                # print("Input shape", inputs.shape)  # [batch_size, 16384]
                # print("Target shape", targets.shape) # [batch_size, num_groups]
                # print("Pred shape", outputs.shape)  # [batch_size, num_groups]

                preds = torch.cat((preds, torch.flatten(outputs)), dim=0)
                truths = torch.cat((truths, torch.flatten(targets)), dim=0)

            preds = preds.cpu().detach().numpy()
            truths = truths.cpu().detach().numpy()

            num_samples = preds.shape[0]
            print('samples', num_samples)
            losses = (preds - truths) ** 2
            print('losses', losses.shape)

            fig, axs = plt.subplots(
                1, 3, figsize=(18, 6), sharex=True, sharey=True, tight_layout=True
            )  # , gridspec_kw={'height_ratios': [1, 1, 1]})
            num_bins = 51  # odd to see if 0-velocity over/under-predicted

            range_min = min(np.min(truths), np.min(preds))
            range_max = max(np.max(truths), np.max(preds))
            range_max = max(
                np.abs(range_min), range_max
            )  # find max magnitude in all data
            hist_range = [[-range_max, range_max]] * 2
            # 2D Histogram for Counts
            hist_counts, xedges_counts, yedges_counts = np.histogram2d(
                truths, preds, bins=num_bins, range=hist_range
            )
            masked_hist_counts = np.ma.masked_where(hist_counts == 0, hist_counts)
            im_counts_log = axs[0].imshow(
                masked_hist_counts.T,
                extent=[
                    xedges_counts[0],
                    xedges_counts[-1],
                    yedges_counts[0],
                    yedges_counts[-1],
                ],
                origin='lower',
                aspect='equal',
                cmap='coolwarm',
                norm=LogNorm(),
            )
            axs[0].set_xlabel('Expected Velocity (um/s)')
            axs[0].set_ylabel('Predicted Velocity (um/s)')
            axs[0].set_title('Counts for Predicted vs Expected Velocity')
            cbar_0 = fig.colorbar(
                im_counts_log, ax=axs[0], orientation='horizontal', pad=0.1
            )
            cbar_0.set_label('Counts Per Bin')
            # avg_mse = hist_mse_sum / hist_counts # avg mse per bin

            # 2D Histogram for Average MSELoss
            hist_mse_sum, xedges, yedges = np.histogram2d(
                truths, preds, bins=num_bins, weights=losses, range=hist_range
            )
            # masked_hist = np.ma.masked_where(hist_mse_sum == 0, hist_mse_sum)
            cax = axs[1].imshow(
                (hist_mse_sum / masked_hist_counts).T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower',
                aspect='equal',
                cmap='coolwarm',
                norm=LogNorm(),
            )
            # with np.errstate(divide='ignore', invalid='ignore'):
            #     cax = axs[1].imshow((hist_mse_sum / hist_counts).T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto')
            cbar_1 = fig.colorbar(cax, ax=axs[1], orientation='horizontal', pad=0.1)
            cbar_1.set_label('Average MSE Loss Per Bin')
            axs[1].set_xlabel('Expected Velocity (um/s)')
            # axs[1].set_ylabel('Predicted Velocity')
            axs[1].set_title('Avg MSELoss for Predicted vs Expected Velocity')

            # 2D Histogram for Summed MSELoss
            sum_ax = axs[2].imshow(
                hist_mse_sum.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower',
                aspect='equal',
                cmap='coolwarm',
                norm=LogNorm(),
            )
            cbar_sum = fig.colorbar(
                sum_ax, ax=axs[2], orientation='horizontal', pad=0.1
            )
            cbar_sum.set_label('Summed MSE Loss Per Bin')
            axs[2].set_xlabel('Expected Velocity (um/s)')
            # axs[2].set_ylabel('Predicted Velocity')
            axs[2].set_title('Summed MSELoss for Predicted vs Expected Velocity')

            for ax in axs:
                ax.axline((0, 0), slope=1, color='black')
                ax.axline((0, 0), slope=-1, color='black')
                ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
            plt.show()
    else:
        print('Error: Unsupported number of command-line arguments')
