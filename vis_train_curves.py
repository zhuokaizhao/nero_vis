# the script visualizes the training loss curve from input npy files
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # purpose of generating dataset (shift-equivariance, rotation-equivariance, scale-equivariance)
    parser.add_argument('-p', '--purpose', action='store', nargs=1, dest='purpose')
    parser.add_argument('-i', '--input_dir', action='store', type=str, dest='input_dir')
    opt = parser.parse_args()

    # read parameters
    purpose = opt.purpose[0]
    input_dir = opt.input_dir
    eval_interval = 5
    backbone_name = 'resnet50'

    if purpose == 'shift-equivariance':

        jittering_levels = ['0', '33', '66', '100']
        train_epochs = ['50', '100', '100', '100']

        for i, jitter in enumerate(jittering_levels):
            train_loss_path = os.path.join(input_dir, f'object_{jitter}-jittered/faster_rcnn_{backbone_name}_{jitter}-jittered_train_loss_1_{train_epochs[i]}.npy')
            val_loss_path = os.path.join(input_dir, f'object_{jitter}-jittered/faster_rcnn_{backbone_name}_{jitter}-jittered_val_loss_1_{train_epochs[i]}.npy')
            val_mAP_path = os.path.join(input_dir, f'object_{jitter}-jittered/faster_rcnn_{backbone_name}_{jitter}-jittered_val_mAP_1_{train_epochs[i]}.npy')

            train_losses = np.load(train_loss_path)

            # x = list(range(0, num_epoch+eval_interval, eval_interval))
            val_losses = np.load(val_loss_path)
            val_mAP = np.load(val_mAP_path)
            val_mAP = np.insert(val_mAP, 0, 0)

            plt.figure()
            plt.plot(train_losses, label='train loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'{jitter}-jittering train loss curve')
            plt.show()

            plt.figure()
            plt.plot(val_losses, label='val loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'{jitter}-jittering val loss curve')
            plt.show()

            plt.figure()
            plt.plot(val_mAP, label='val mAP')
            plt.xlabel('epoch')
            plt.ylabel('mAP')
            plt.title(f'{jitter}-jittering val mAP curve')
            plt.show()

    elif purpose == 'rotation-equivariance':

        rotation_levels = ['33', '66']
        train_epochs = ['100', '100']

        for i, rot in enumerate(rotation_levels):
            train_loss_path = os.path.join(input_dir, f'object_{rot}-rotated/faster_rcnn_{backbone_name}_{rot}-rotated_train_loss_1_{train_epochs[i]}.npy')
            val_loss_path = os.path.join(input_dir, f'object_{rot}-rotated/faster_rcnn_{backbone_name}_{rot}-rotated_val_loss_1_{train_epochs[i]}.npy')
            val_mAP_path = os.path.join(input_dir, f'object_{rot}-rotated/faster_rcnn_{backbone_name}_{rot}-rotated_val_mAP_1_{train_epochs[i]}.npy')

            train_losses = np.load(train_loss_path)

            # x = list(range(0, num_epoch+eval_interval, eval_interval))
            val_losses = np.load(val_loss_path)
            val_mAP = np.load(val_mAP_path)
            val_mAP = np.insert(val_mAP, 0, 0)

            plt.figure()
            plt.plot(train_losses, label='train loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'{rot}-rotated train loss curve')
            plt.show()

            plt.figure()
            plt.plot(val_losses, label='val loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'{rot}-rotated val loss curve')
            plt.show()

            plt.figure()
            plt.plot(val_mAP, label='val mAP')
            plt.xlabel('epoch')
            plt.ylabel('mAP')
            plt.title(f'{rot}-rotated val mAP curve')
            plt.show()
