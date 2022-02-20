# the script gets called by nero_app when running the model

import torch
import numpy as np

def run_mnist_once(model_type, model_path, num_workers=8, shuffle=True, batch_size=1000):
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
        model = models.Non_Eqv_Net_MNIST(type).to(device)

    elif network_model == 'rot-eqv':
        # number of groups for e2cnn
        num_rotation = 8
        model = models.Rot_Eqv_Net_MNIST(image_size=image_size, num_rotation=num_rotation).to(device)

    elif network_model == 'shift-eqv':
        model = models.Shift_Eqv_Net_MNIST().to(device)

    elif network_model == 'scale-eqv':
        # Scale-Equivariant Steerable Networks
        method = 'SESN'
        # Deep Scale Spaces: Equivariant Over Scale
        # method = 'DSS'
        model = models.Scale_Eqv_Net_MNIST(method=method).to(device)

    # load previously trained model
    trained_model = torch.load(model_dir)
    model.load_state_dict(trained_model['state_dict'])
    trained_epoch = trained_model['epoch']


def run_mnist_all():
    exit()