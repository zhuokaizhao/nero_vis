# the script gets called by nero_app when running the model
import os
import sys
import time
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='0'

# add parent directories for importing model scripts
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from nero_mnist import models as mnist_models

def run_mnist_once(network_model, model_dir, test_image):
    print('Running inference on the single image')
    # reformat input image from (height, width, channel) to (batch size, channel, height, width)
    test_image = test_image.permute((2, 0, 1))[None, :, :, :].float()

    # basic settings for pytorch
    if torch.cuda.is_available():
        # device set up
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
    # set model in evaluation mode
    model.eval()

    with torch.no_grad():
        batch_time_start = time.time()
        # prepare current batch's testing data
        test_image = test_image.to(device)

        # inference
        output = model(test_image).cpu().detach().numpy()[0]
        pred = output.argmax()

        batch_time_end = time.time()
        batch_time_cost = batch_time_end - batch_time_start
        print(f'Inference time: {batch_time_cost} seconds')

    return output, pred


# run mnist model on all rotations and return all the results
def run_mnist_all():
    exit()