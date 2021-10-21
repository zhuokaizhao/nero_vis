# the script visualizes the training loss curve from input npy files
import numpy as np
from matplotlib import pyplot as plt


num_epoch = 100
eval_interval = 5
jittering_levels = ['0', '20', '40', '60', '80', '100']
train_epochs = ['50', '100', '100', '100', '100', '220']
jittering_levels = ['20', '33', '66', '60']
train_epochs = ['100', '100', '100', '100']
backbone_name = 'resnet50'

for i, jitter in enumerate(jittering_levels):
    train_loss_path = f'logs/object_{jitter}-jittered/faster_rcnn_{backbone_name}_{jitter}-jittered_train_loss_1_{train_epochs[i]}.npy'
    val_loss_path = f'logs/object_{jitter}-jittered/faster_rcnn_{backbone_name}_{jitter}-jittered_val_loss_1_{train_epochs[i]}.npy'
    val_mAP_path = f'logs/object_{jitter}-jittered/faster_rcnn_{backbone_name}_{jitter}-jittered_val_mAP_1_{train_epochs[i]}.npy'

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
