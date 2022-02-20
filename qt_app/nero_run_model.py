# the script gets called by nero_app when running the model
import os
import sys
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='0'

# add parent directories for importing model scripts
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from nero_mnist import models as mnist_models

def run_mnist_once(network_model, model_dir, test_image, num_workers=8, shuffle=True, batch_size=1000):
    # basic settings for pytorch
    if torch.cuda.is_available():

        cuda_kwargs = {'num_workers': num_workers,
                        'pin_memory': True,
                        'shuffle': shuffle,
                        'batch_size': batch_size}

        # device set up
        if torch.cuda.device_count() > 1:
            print('\n', torch.cuda.device_count(), 'GPUs available')
            device = torch.device('cuda')
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # model
    if network_model == 'non-eqv' or network_model == 'aug-eqv':
        model = mnist_models.Non_Eqv_Net_MNIST('rotation').to(device)

    elif network_model == 'rot-eqv':
        # number of groups for e2cnn
        num_rotation = 8
        model = mnist_models.Rot_Eqv_Net_MNIST(image_size=None, num_rotation=num_rotation).to(device)

    elif network_model == 'shift-eqv':
        model = mnist_models.Shift_Eqv_Net_MNIST().to(device)

    elif network_model == 'scale-eqv':
        # Scale-Equivariant Steerable Networks
        method = 'SESN'
        # Deep Scale Spaces: Equivariant Over Scale
        # method = 'DSS'
        model = mnist_models.Scale_Eqv_Net_MNIST(method=method).to(device)

    # load previously trained model
    trained_model = torch.load(model_dir)
    model.load_state_dict(trained_model['state_dict'])
    trained_epoch = trained_model['epoch']
    print(trained_epoch)


def run_mnist_all():
    exit()