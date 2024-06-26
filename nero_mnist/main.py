from __future__ import print_function
import os
import sys
import time
import torch
import e2cnn
import argparse
import numpy as np
import torchvision
import torch.optim as optim
from subprocess import call
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from pytorch_model_summary import summary
from torch.optim.lr_scheduler import StepLR

from PIL import Image


import vis
import models
import datasets
import data_transform

os.environ['CUDA_VISIBLE_DEVICES']='0'

# print cuda information
print('\n\nPython VERSION:', sys.version)
print('PyTorch VERSION:', torch.__version__)
print('CUDNN VERSION:', torch.backends.cudnn.version())
print('Number CUDA Devices:', torch.cuda.device_count())
print('Devices')
call(['nvidia-smi', '--format=csv', '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free'])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

# Print iterations progress
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# train each epoch
def train(model, device, train_loader, optimizer):
    all_batch_losses = []
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_time_start = time.time()
        # move data to GPU
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # negative log likelihood loss.
        loss = loss_function(output, target)
        loss.backward()
        all_batch_losses.append(loss.item())
        optimizer.step()
        batch_time_end = time.time()
        batch_time_cost = batch_time_end - batch_time_start

        print_progress_bar(iteration=batch_idx+1,
                            total=len(train_loader),
                            prefix=f'Train batch {batch_idx+1}/{len(train_loader)},',
                            suffix='%s: %.3f, time: %.2f' % ('CE loss', all_batch_losses[-1], batch_time_cost),
                            length=50)

    # return the averaged batch loss
    return np.mean(all_batch_losses)


# validation
def val(model, device, val_loader):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    all_batch_losses = []
    num_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            batch_time_start = time.time()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            all_batch_losses.append(loss.item())
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            num_correct += pred.eq(target.view_as(pred)).sum().item()
            batch_time_end = time.time()
            batch_time_cost = batch_time_end - batch_time_start

            print_progress_bar(iteration=batch_idx+1,
                                total=len(val_loader),
                                prefix=f'Val batch {batch_idx+1}/{len(val_loader)},',
                                suffix='%s: %.3f, time: %.2f' % ('CE loss', all_batch_losses[-1], batch_time_cost),
                                length=50)

    # return the averaged batch loss
    return np.mean(all_batch_losses), num_correct


# test
def test(model, device, test_loader):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    all_batch_avg_losses = []
    total_num_correct = 0
    # number of correct for each digit
    num_digits_per_digit = np.zeros(10, dtype=int)
    num_correct_per_digit = np.zeros(10, dtype=int)
    # all individua loss saved per digit
    individual_losses_per_digit = []
    for i in range(10):
        individual_losses_per_digit.append([])

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            batch_time_start = time.time()
            # print(target)
            # prepare current batch's testing data
            data, target = data.to(device), target.to(device)

            # inference
            output = model(data)
            loss = loss_function(output, target)
            all_batch_avg_losses.append(torch.mean(loss).item())

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            total_num_correct += pred.eq(target.view_as(pred)).sum().item()

            # calculate per-class performance
            # total number of cases for each class (digit)
            for i in range(10):
                num_digits_per_digit[i] += int(target.cpu().tolist().count(i))

            # number of correct for each case
            for i in range(len(pred)):
                # individual losses formed by digits
                individual_losses_per_digit[target.cpu().tolist()[i]].append(loss[i].cpu().item())
                # correctness
                if pred.eq(target.view_as(pred))[i].item() == True:
                    num_correct_per_digit[target.cpu().tolist()[i]] += 1

            batch_time_end = time.time()
            batch_time_cost = batch_time_end - batch_time_start

            print_progress_bar(iteration=batch_idx+1,
                                total=len(test_loader),
                                prefix=f'Test batch {batch_idx+1}/{len(test_loader)},',
                                suffix='%s: %.3f, time: %.2f' % ('CE loss', all_batch_avg_losses[-1], batch_time_cost),
                                length=50)

    # average (over digits) loss and accuracy
    avg_loss = np.mean(all_batch_avg_losses)
    avg_accuracy = total_num_correct / len(test_loader.dataset)
    # each digits loss and accuracy
    sum_individual_losses_per_digit = np.zeros(10)
    for i in range(10):
        sum_individual_losses_per_digit[i] = sum(individual_losses_per_digit[i])
    avg_loss_per_digit = sum_individual_losses_per_digit / num_digits_per_digit
    avg_accuracy_per_digit = num_correct_per_digit / num_digits_per_digit

    # return the averaged batch loss
    return avg_loss, avg_accuracy, avg_loss_per_digit, avg_accuracy_per_digit, individual_losses_per_digit


# normalize input array input [0, 1]
def normalize_0_1(in_array, in_min=None, in_max=None):
    if in_min == None:
        in_min = np.min(in_array)
    if in_max == None:
        in_max = np.max(in_array)

    out_array = 1 - (in_array - in_min) / (in_max - in_min)

    return out_array


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='NERO plots with MNIST')
    # mode (train, test or analyze)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # type of equivariance (rotation, shift, or scale)
    parser.add_argument('--type', required=True, action='store', nargs=1, dest='type')
    # network method (non-eqv, eqv, aug-eqv, etc)
    parser.add_argument('-n', '--network_model', action='store', nargs=1, dest='network_model')
    # image dimension after padding
    parser.add_argument('-s', '--image_size', action='store', nargs=1, dest='image_size')
    # training batch size
    parser.add_argument('--train_batch_size', type=int, dest='train_batch_size',
                        help='input batch size for training')
    # validation batch size
    parser.add_argument('--val_batch_size', type=int, dest='val_batch_size',
                        help='input batch size for validation')
    # test batch size
    parser.add_argument('--test_batch_size', type=int, dest='test_batch_size',
                        help='input batch size for testing')
    # number of training epochs
    parser.add_argument('-e', '--num_epoch', type=int, dest='num_epoch',
                        help='number of epochs to train')
    # save the checkpoint model to the output model directory
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    # checkpoint path for continuing training
    parser.add_argument('-c', '--checkpoint_dir', action='store', nargs=1, dest='checkpoint_path')
    # input or output model directory
    parser.add_argument('-m', '--model_dir', action='store', nargs=1, dest='model_dir')
    # input directory for non-eqv model output when in analyze mode
    parser.add_argument('--non_eqv_path', action='store', nargs=1, dest='non_eqv_path')
    # input directory for eqv model output when in analyze mode
    parser.add_argument('--eqv_path', action='store', nargs=1, dest='eqv_path')
    # input directory for aug eqv model output when in analyze mode
    parser.add_argument('--aug_eqv_path', action='store', nargs=1, dest='aug_eqv_path')
    # output figs directory
    parser.add_argument('-o', '--output_dir', action='store', nargs=1, dest='output_dir')
    # if visualizing data
    parser.add_argument('--vis', action='store_true', dest='vis', default=False)
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)

    args = parser.parse_args()

    # input variables
    mode = args.mode[0]
    type = args.type[0]
    if args.image_size != None:
        image_size = (int(args.image_size[0].split('x')[0]), int(args.image_size[0].split('x')[1]))
    else:
        image_size = None
    # model type (non-eqv or eqv)
    if mode == 'train' or mode == 'test':
        network_model = args.network_model[0]
    # output graph directory
    figs_dir = args.output_dir[0]
    vis_data = args.vis
    verbose = args.verbose

    # when training/testing for rotation equivariance tests
    if type == 'rotation':
        if image_size == None:
            # padded to 29x29 to use odd-size filters with stride 2 when downsampling a feature map in the model
            image_size = (29, 29)

    # when training/testing for shift equivariance tests
    elif type == 'shift':
        # padded to 69x69 to use odd-size filters with stride 2 when downsampling a feature map in the model
        if image_size == None:
            image_size = (69, 69)

    # when training/testing for scale equivariance tests
    elif type == 'scale':
        all_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        if image_size == None:
            image_size = (49, 49)
            # image_size = (29, 29)

    # training mode
    if mode == 'train':

        # default train_batch_size and val_batch_size if not assigned
        if args.train_batch_size == None:
            train_batch_size = 64
        else:
            train_batch_size = args.train_batch_size
        if args.val_batch_size == None:
            val_batch_size = 1000
        else:
            val_batch_size = args.val_batch_size

        # number of training epoch
        num_epoch = args.num_epoch

        # MNIST models use Adam
        learning_rate = 5e-5
        weight_decay = 1e-5

        # checkpoint path to load the model from
        if args.checkpoint_path != None:
            checkpoint_path = args.checkpoint_path[0]
        else:
            checkpoint_path = None

        # directory to save the model to
        model_dir = args.model_dir[0]

        # kwargs for train and val dataloader
        train_kwargs = {'batch_size': train_batch_size}
        val_kwargs = {'batch_size': val_batch_size}
        if torch.cuda.is_available():
            # additional cuda kwargs
            cuda_kwargs = {'num_workers': 8,
                            'pin_memory': True,
                            'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            val_kwargs.update(cuda_kwargs)

            # device set up
            if torch.cuda.device_count() > 1:
                print('\n', torch.cuda.device_count(), 'GPUs available')
                device = torch.device('cuda')
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        if verbose:
            print(f'\nmode: {mode}')
            print(f'GPU usage: {device}')
            print(f'netowrk model: {network_model}')
            if image_size != None:
                print(f'Padded image size: {image_size[1]}x{image_size[0]}')
            else:
                print(f'Padded image size: {image_size}')
            print(f'training batch size: {train_batch_size}')
            print(f'validation batch size: {val_batch_size}')
            print(f'output model dir: {model_dir}')
            print(f'epoch size: {num_epoch}')
            print(f'initial learning rate: {learning_rate}')
            print(f'optimizer weight decay: {weight_decay}')
            print(f'model input checkpoint path: {checkpoint_path}')

        # model
        starting_epoch = 0
        if type == 'rotation':
            if network_model == 'non-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
                # data transform
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Pad(((image_size[1]-28)//2, (image_size[0]-28)//2, (image_size[1]-28)//2+1, (image_size[0]-28)//2+1), fill=0, padding_mode='constant'),
                ])
            elif network_model == 'rot-eqv':
                # number of groups for e2cnn
                num_rotation = 8
                model = models.Rot_Eqv_Net_MNIST(image_size=image_size, num_rotation=num_rotation).to(device)
                # data transform
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Pad(((image_size[1]-28)//2, (image_size[0]-28)//2, (image_size[1]-28)//2+1, (image_size[0]-28)//2+1), fill=0, padding_mode='constant'),
                ])
            elif network_model == 'aug-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
                # data transform
                transform = torchvision.transforms.Compose([
                    data_transform.RandomRotate(28, image_size[0])
                ])
            else:
                raise Exception(f'Wrong network model type {network_model}')

        elif type == 'shift':
            if network_model == 'non-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Pad(((image_size[1]-28)//2, (image_size[0]-28)//2, (image_size[1]-28)//2+1, (image_size[0]-28)//2+1), fill=0, padding_mode='constant'),
                ])
            elif network_model == 'shift-eqv':
                model = models.Shift_Eqv_Net_MNIST().to(device)
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Pad(((image_size[1]-28)//2, (image_size[0]-28)//2, (image_size[1]-28)//2+1, (image_size[0]-28)//2+1), fill=0, padding_mode='constant'),
                ])
            elif network_model == 'aug-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
                # data transform
                transform = torchvision.transforms.Compose([
                    # random padding
                    data_transform.RandomShift(28, image_size[0])
                ])
            else:
                raise Exception(f'Wrong network model type {network_model}')

        elif type == 'scale':
            if network_model == 'non-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Pad(((image_size[1]-28)//2, (image_size[0]-28)//2, (image_size[1]-28)//2+1, (image_size[0]-28)//2+1), fill=0, padding_mode='constant')
                ])
            elif network_model == 'scale-eqv':
                # Scale-Equivariant Steerable Networks (newer)
                method = 'SESN'
                # Deep Scale Spaces: Equivariant Over Scale
                # method = 'DSS'
                model = models.Scale_Eqv_Net_MNIST(method=method).to(device)
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Pad(((image_size[1]-28)//2, (image_size[0]-28)//2, (image_size[1]-28)//2+1, (image_size[0]-28)//2+1), fill=0, padding_mode='constant')
                ])
            elif network_model == 'aug-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
                transform = torchvision.transforms.Compose([
                    data_transform.RandomScale(all_scales, 28, image_size[0])
                ])
            else:
                raise Exception(f'Wrong network model type {network_model}')

        else:
            raise Exception(f'Wrong type {type}')

        # visualize the model structure
        if network_model == 'non-eqv' or network_model == 'aug-eqv':
            print(summary(model, torch.zeros((1, 1, image_size[1], image_size[0])).to(device)))

        # split into train and validation dataset
        dataset = datasets.MnistDataset(mode='train', transform=transform)
        train_data, val_data = torch.utils.data.random_split(dataset, [50000, 10000])

        # pytorch data loader
        train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_data, **val_kwargs)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # load previously trained model information
        if checkpoint_path != None:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            starting_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

        # train the model
        all_epoch_train_losses = []
        all_epoch_val_losses = []
        for epoch in range(starting_epoch, starting_epoch+num_epoch):
            print(f'\nStarting epoch {epoch+1}/{starting_epoch+num_epoch}')
            epoch_start_time = time.time()
            cur_epoch_train_loss = train(model, device, train_loader, optimizer)
            all_epoch_train_losses.append(cur_epoch_train_loss)
            cur_epoch_val_loss, num_correct = val(model, device, val_loader)
            all_epoch_val_losses.append(cur_epoch_val_loss)
            epoch_end_time = time.time()
            epoch_time_cost = epoch_end_time - epoch_start_time

            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    cur_epoch_val_loss, num_correct, len(val_loader.dataset),
                    100. * num_correct / len(val_loader.dataset)))
            print(f'Epoch time cost: {epoch_time_cost}')

            if (epoch+1) % args.checkpoint_interval == 0:

                # save model as a checkpoint so further training could be resumed
                if network_model == 'non-eqv' or network_model == 'shift-eqv' or network_model == 'aug-eqv':
                    model_path = os.path.join(model_dir, f'{network_model}_mnist_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
                elif network_model == 'rot-eqv':
                    model_path = os.path.join(model_dir, f'{network_model}_rot-group{num_rotation}_mnist_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
                elif network_model == 'scale-eqv':
                    model_path = os.path.join(model_dir, f'{network_model}_{method}_mnist_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
                else:
                    raise Exception(f'Unknown network model {network_model}')

                # if trained on multiple GPU's, store model.module.state_dict()
                if torch.cuda.device_count() > 1:
                    model_checkpoint = {
                                            'epoch': epoch+1,
                                            'state_dict': model.module.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'train_loss': all_epoch_train_losses,
                                            'val_loss': all_epoch_val_losses
                                        }
                else:
                    model_checkpoint = {
                                            'epoch': epoch+1,
                                            'state_dict': model.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'train_loss': all_epoch_train_losses,
                                            'val_loss': all_epoch_val_losses
                                        }

                torch.save(model_checkpoint, model_path)
                print(f'\nTrained model checkpoint has been saved to {model_path}\n')

                # save loss graph and model
                if checkpoint_path != None:
                    checkpoint = torch.load(checkpoint_path)
                    prev_train_losses = checkpoint['train_loss']
                    prev_val_losses = checkpoint['val_loss']
                    all_epoch_train_losses = prev_train_losses + all_epoch_train_losses
                    all_epoch_val_losses = prev_val_losses + all_epoch_val_losses

                plt.plot(all_epoch_train_losses, label='Train')
                plt.plot(all_epoch_val_losses, label='Validation')
                plt.title(f'Training and validation loss on {network_model} model')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend(loc='upper right')
                loss_path = os.path.join(figs_dir, f'{network_model}_mnist_batch{train_batch_size}_epoch{epoch+1}_loss.png')
                plt.savefig(loss_path)
                print(f'\nLoss graph has been saved to {loss_path}')

    elif 'test' in mode:

        if args.test_batch_size == None:
            test_batch_size = 1000
        else:
            test_batch_size = args.test_batch_size

        network_model = args.network_model[0]
        # directory to load the model from
        model_dir = args.model_dir[0]
        test_kwargs = {'batch_size': test_batch_size}

        # basic settings for pytorch
        if torch.cuda.is_available():
            # additional kwargs
            if vis:
                num_workers = 1
                shuffle = False
            else:
                num_workers = 8
                shuffle = True

            cuda_kwargs = {'num_workers': num_workers,
                            'pin_memory': True,
                            'shuffle': shuffle}
            test_kwargs.update(cuda_kwargs)

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

        if verbose:
            print(f'\nmode: {mode}')
            print(f'GPU usage: {device}')
            print(f'netowrk model: {network_model}')
            print(f'testing batch size: {test_batch_size}')
            print(f'input model dir: {model_dir}')

        # test on each rotation degree
        if type == 'rotation':
            max_rot = 360
            increment = 1
            general_loss = np.zeros(max_rot//increment)
            general_accuracy = np.zeros(max_rot//increment)
            categorical_loss = np.zeros((max_rot//increment, 10))
            categorical_accuracy = np.zeros((max_rot//increment, 10))
            individual_categorical_loss = []
            for d in range(10):
                individual_categorical_loss.append([])

            for k in range(0, max_rot, increment):
                print(f'\n Testing rotation angle {k}/{max_rot-increment}')
                # row
                transform = torchvision.transforms.Compose([
                    # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
                    torchvision.transforms.Resize(28*3),
                    torchvision.transforms.RandomRotation((k, k), resample=Image.BILINEAR, expand=False),
                    torchvision.transforms.Resize(28),
                    torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant'),
                ])

                # prepare test dataset
                dataset = datasets.MnistDataset(mode='test', transform=transform, vis=vis_data)
                test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

                # test the model
                cur_general_loss, \
                cur_general_accuracy, \
                cur_categorical_loss, \
                cur_categorical_accuracy, \
                cur_individual_losses_per_digit = test(model, device, test_loader)
                general_loss[k//increment] = cur_general_loss
                general_accuracy[k//increment] = cur_general_accuracy
                categorical_loss[k//increment, :] = cur_categorical_loss
                categorical_accuracy[k//increment, :] = cur_categorical_accuracy

                for d in range(10):
                    individual_categorical_loss[d].append(cur_individual_losses_per_digit[d])
                    # print(np.array(individual_categorical_loss[d]).shape)
                    # exit()

                print(f'Rotation {k} accuracy: {cur_general_accuracy}, loss: {cur_general_loss}')

            # save the accuracy as npz file
            if network_model == 'non-eqv' or network_model == 'aug-eqv':
                loss_path = os.path.join(figs_dir, f'{network_model}_mnist_{image_size[0]}x{image_size[1]}_angle-increment{increment}_epoch{trained_epoch}_result.npz')
            elif network_model == 'rot-eqv':
                loss_path = os.path.join(figs_dir, f'{network_model}_rot-group{num_rotation}_mnist_{image_size[0]}x{image_size[1]}_angle-increment{increment}_epoch{trained_epoch}_result.npz')

            individual_categorical_loss_0 = np.array(individual_categorical_loss[0])
            individual_categorical_loss_1 = np.array(individual_categorical_loss[1])
            individual_categorical_loss_2 = np.array(individual_categorical_loss[2])
            individual_categorical_loss_3 = np.array(individual_categorical_loss[3])
            individual_categorical_loss_4 = np.array(individual_categorical_loss[4])
            individual_categorical_loss_5 = np.array(individual_categorical_loss[5])
            individual_categorical_loss_6 = np.array(individual_categorical_loss[6])
            individual_categorical_loss_7 = np.array(individual_categorical_loss[7])
            individual_categorical_loss_8 = np.array(individual_categorical_loss[8])
            individual_categorical_loss_9 = np.array(individual_categorical_loss[9])

            np.savez(loss_path,
                    general_loss=general_loss,
                    general_accuracy=general_accuracy,
                    categorical_loss=categorical_loss,
                    categorical_accuracy=categorical_accuracy,
                    individual_categorical_loss_digit_0=individual_categorical_loss_0,
                    individual_categorical_loss_digit_1=individual_categorical_loss_1,
                    individual_categorical_loss_digit_2=individual_categorical_loss_2,
                    individual_categorical_loss_digit_3=individual_categorical_loss_3,
                    individual_categorical_loss_digit_4=individual_categorical_loss_4,
                    individual_categorical_loss_digit_5=individual_categorical_loss_5,
                    individual_categorical_loss_digit_6=individual_categorical_loss_6,
                    individual_categorical_loss_digit_7=individual_categorical_loss_7,
                    individual_categorical_loss_digit_8=individual_categorical_loss_8,
                    individual_categorical_loss_digit_9=individual_categorical_loss_9)

            print(f'\nTesting result has been saved to {loss_path}')

        # test on each shift from -20 to 20
        elif type == 'shift':
            # shift use the image coordinate system
            # |-----> x
            # |
            # |
            # v y
            shift_amount = (image_size[0] - 29)//2
            general_loss = np.zeros((shift_amount*2+1, shift_amount*2+1))
            general_accuracy = np.zeros((shift_amount*2+1, shift_amount*2+1))
            categorical_loss = np.zeros((shift_amount*2+1, shift_amount*2+1, 10))
            categorical_accuracy = np.zeros((shift_amount*2+1, shift_amount*2+1, 10))
            individual_categorical_loss = []
            for d in range(10):
                individual_categorical_loss.append([])
            for y_shift in range(-shift_amount, shift_amount+1):
                for x_shift in range(-shift_amount, shift_amount+1):
                    # pad the MNIST image
                    print(f'\n Testing shift amount x = {x_shift}, y = {y_shift}')

                    # padding information based on shift left, top, right and bottom
                    # example 1
                    # if y_shift = -20, x_shift = -20, digit is at top left corner
                    # so left_padding = 0, top padding = 0, right padding = 40, bottom padding = 40
                    # example 2
                    # if y_shift = 20, x_shift = 20, digit is at bottom right corner
                    # so left_padding = 40, top padding = 40, right padding = 0, bottom padding = 0
                    # example 3
                    # if y_shift = 20, x_shift = -20, digit is at bottom left corner
                    # so left_padding = 0, top padding = 40, right padding = 40, bottom padding = 0
                    # example 4
                    # if y_shift = -20, x_shift = 20, digit is at top right corner
                    # so left_padding = 40, top padding = 0, right padding = 0, bottom padding = 40
                    left_padding = np.abs(x_shift + shift_amount)
                    top_padding = np.abs(y_shift + shift_amount)
                    right_padding = np.abs(x_shift - shift_amount)
                    bottom_padding = np.abs(y_shift - shift_amount)

                    # row
                    transform = torchvision.transforms.Compose([
                        # padding according to shift
                        torchvision.transforms.Pad((left_padding, top_padding, right_padding, bottom_padding), fill=0, padding_mode='constant'),
                        # traditional 1 padding for being odd
                        torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant')
                    ])

                    # prepare test dataset
                    dataset = datasets.MnistDataset(mode='test', transform=transform, vis=vis_data)
                    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

                    # test the model
                    cur_general_loss, \
                    cur_general_accuracy, \
                    cur_categorical_loss, \
                    cur_categorical_accuracy, \
                    cur_individual_losses_per_digit = test(model, device, test_loader)
                    general_loss[y_shift+shift_amount, x_shift+shift_amount] = cur_general_loss
                    general_accuracy[y_shift+shift_amount, x_shift+shift_amount] = cur_general_accuracy
                    categorical_loss[y_shift+shift_amount, x_shift+shift_amount, :] = cur_categorical_loss
                    categorical_accuracy[y_shift+shift_amount, x_shift+shift_amount, :] = cur_categorical_accuracy

                    for d in range(10):
                        individual_categorical_loss[d].append(cur_individual_losses_per_digit[d])

                    print(f'shift amount x = {x_shift}, y = {y_shift} accuracy: {cur_general_accuracy}, loss: {cur_general_loss}')

            # save the accuracy as npz file
            loss_path = os.path.join(figs_dir, f'{network_model}_mnist_{image_size[0]}x{image_size[1]}_epoch{trained_epoch}_result.npz')

            individual_categorical_loss_0 = np.array(individual_categorical_loss[0])
            individual_categorical_loss_1 = np.array(individual_categorical_loss[1])
            individual_categorical_loss_2 = np.array(individual_categorical_loss[2])
            individual_categorical_loss_3 = np.array(individual_categorical_loss[3])
            individual_categorical_loss_4 = np.array(individual_categorical_loss[4])
            individual_categorical_loss_5 = np.array(individual_categorical_loss[5])
            individual_categorical_loss_6 = np.array(individual_categorical_loss[6])
            individual_categorical_loss_7 = np.array(individual_categorical_loss[7])
            individual_categorical_loss_8 = np.array(individual_categorical_loss[8])
            individual_categorical_loss_9 = np.array(individual_categorical_loss[9])

            np.savez(loss_path,
                    general_loss=general_loss,
                    general_accuracy=general_accuracy,
                    categorical_loss=categorical_loss,
                    categorical_accuracy=categorical_accuracy,
                    individual_categorical_loss_digit_0=individual_categorical_loss_0,
                    individual_categorical_loss_digit_1=individual_categorical_loss_1,
                    individual_categorical_loss_digit_2=individual_categorical_loss_2,
                    individual_categorical_loss_digit_3=individual_categorical_loss_3,
                    individual_categorical_loss_digit_4=individual_categorical_loss_4,
                    individual_categorical_loss_digit_5=individual_categorical_loss_5,
                    individual_categorical_loss_digit_6=individual_categorical_loss_6,
                    individual_categorical_loss_digit_7=individual_categorical_loss_7,
                    individual_categorical_loss_digit_8=individual_categorical_loss_8,
                    individual_categorical_loss_digit_9=individual_categorical_loss_9)

            print(f'\nTesting result has been saved to {loss_path}')

        # test on each scale from 0.3 to 1.7
        elif type == 'scale':
            general_loss = np.zeros(len(all_scales))
            general_accuracy = np.zeros(len(all_scales))
            categorical_loss = np.zeros((len(all_scales), 10))
            categorical_accuracy = np.zeros((len(all_scales), 10))

            for k, scale in enumerate(all_scales):
                print(f'\n Testing scale {scale}')
                # prepare data transform
                resize_size = int(np.floor(28 * scale))
                # size difference must be odd
                if (image_size[1]-resize_size) % 2 == 0:
                    resize_size += 1

                transform = torchvision.transforms.Compose([
                    # resize and pad the image equally to image_size
                    torchvision.transforms.Resize(resize_size),
                    torchvision.transforms.Pad(((image_size[1]-resize_size)//2, (image_size[0]-resize_size)//2, (image_size[1]-resize_size)//2+1, (image_size[0]-resize_size)//2+1), fill=0, padding_mode='constant'),
                ])


                # prepare test dataset
                dataset = datasets.MnistDataset(mode='test', transform=transform, vis=vis_data)
                test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

                # test the model
                cur_general_loss, cur_general_accuracy, cur_categorical_loss, cur_categorical_accuracy = test(model, device, test_loader)
                general_loss[k] = cur_general_loss
                general_accuracy[k] = cur_general_accuracy
                categorical_loss[k, :] = cur_categorical_loss
                categorical_accuracy[k, :] = cur_categorical_accuracy

                print(f'Scale {scale} accuracy: {cur_general_accuracy}, loss: {cur_general_loss}')

            # save the accuracy as npz file
            loss_path = os.path.join(figs_dir, f'{network_model}_mnist_{image_size[0]}x{image_size[1]}_epoch{trained_epoch}_result.npz')
            np.savez(loss_path,
                    general_loss=general_loss,
                    general_accuracy=general_accuracy,
                    categorical_loss=categorical_loss,
                    categorical_accuracy=categorical_accuracy)

            print(f'\nTesting result has been saved to {loss_path}')

    elif 'analyze' in mode:
        non_eqv_path = args.non_eqv_path[0]
        eqv_path = args.eqv_path[0]
        aug_eqv_path = args.aug_eqv_path[0]
        if verbose:
            print(f'\nmode: {mode}')
            print(f'input non-eqv result dir: {non_eqv_path}')
            print(f'input eqv result dir: {eqv_path}')
            print(f'input aug-eqv result dir: {aug_eqv_path}')
            print(f'output figures dir: {figs_dir}')

        # load the non-eqv and eqv result
        non_eqv_result_file = np.load(non_eqv_path)
        eqv_result_file = np.load(eqv_path)
        aug_eqv_result_file = np.load(aug_eqv_path)

        non_eqv_result = non_eqv_result_file['categorical_loss']
        eqv_result = eqv_result_file['categorical_loss']
        aug_eqv_result = aug_eqv_result_file['categorical_loss']

        # normalize all loss in 0/1 scale and have it one-minused
        min_loss = np.min([np.min(non_eqv_result), np.min(eqv_result), np.min(aug_eqv_result)])
        max_loss = np.max([np.max(non_eqv_result), np.max(eqv_result), np.max(aug_eqv_result)])
        non_eqv_result = normalize_0_1(non_eqv_result, min_loss, max_loss)
        eqv_result = normalize_0_1(eqv_result, min_loss, max_loss)
        aug_eqv_result = normalize_0_1(aug_eqv_result, min_loss, max_loss)

        all_colors = ['peru', 'darkviolet', 'green']
        all_styles = ['solid', 'dot', 'dash']

        if type == 'rotation':
            # print out average error per digit for both models
            for i in range(non_eqv_result.shape[1]):
                cur_non_eqv_avg = np.mean(non_eqv_result[:, i])
                cur_eqv_avg = np.mean(eqv_result[:, i])
                cur_aug_eqv_avg = np.mean(aug_eqv_result[:, i])
                print(f'\nDigit {i} avg loss for Non-eqv model: {cur_non_eqv_avg}')
                print(f'Digit {i} avg loss for Eqv model: {cur_eqv_avg}')
                print(f'Digit {i} avg loss for Aug-Eqv model: {cur_aug_eqv_avg}')

            # make and save the plot
            method_labels = ['CNN', 'Steerable CNN', 'CNN+Augmentation']
            # plot interactive polar line plots
            result_path = os.path.join(figs_dir, f'mnist_{mode}_{type}.html')
            # plot_title = f'Non-equivariant vs Equivariant model on {data_name} Digit {i}'
            plot_title = None
            plotting_digits = list(range(10))
            subplot_names = []
            for digit in plotting_digits:
                subplot_names.append(f'Digit {digit}')
            vis.plot_interactive_line_polar(plotting_digits,
                                            method_labels,
                                            all_colors,
                                            all_styles,
                                            [non_eqv_result, eqv_result, aug_eqv_result],
                                            plot_title,
                                            subplot_names,
                                            result_path)

            # load and reformat all individual results (all 10 digits)
            individual_non_eqv_results = []
            individual_eqv_results = []
            individual_aug_eqv_results = []
            for i in range(10):
                # each non_eqv_result_file[f'individual_categorical_loss_digit_{i}'] is num_degrees * num_digit_samples
                individual_non_eqv_results.append(normalize_0_1(non_eqv_result_file[f'individual_categorical_loss_digit_{i}'], min_loss, max_loss))
                individual_eqv_results.append(normalize_0_1(eqv_result_file[f'individual_categorical_loss_digit_{i}'], min_loss, max_loss))
                individual_aug_eqv_results.append(normalize_0_1(aug_eqv_result_file[f'individual_categorical_loss_digit_{i}'], min_loss, max_loss))

            # plot individual polar line plot
            plot_title = None
            plotting_individual_digits = [6]
            # randomly pick 10 digits
            plotting_sample_indices = np.random.randint(0, 360, size=10)
            # plot index
            for i in plotting_individual_digits:
                subplot_names = []
                for index in plotting_sample_indices:
                    subplot_names.append(f'Sample {index}')

                cur_individual_non_eqv = individual_non_eqv_results[i]
                cur_individual_eqv = individual_eqv_results[i]
                cur_individual_aug_eqv = individual_aug_eqv_results[i]
                cur_result_path = os.path.join(figs_dir, f'mnist_{mode}_{type}_digit{i}_samples.html')

                vis.plot_interactive_line_polar(plotting_sample_indices,
                                                method_labels,
                                                all_colors,
                                                all_styles,
                                                [cur_individual_non_eqv, cur_individual_eqv, cur_individual_aug_eqv],
                                                plot_title,
                                                subplot_names,
                                                cur_result_path)

            # dimension reduction on aggregated nero
            individual_losses = [individual_non_eqv_results,
                                    individual_eqv_results,
                                    individual_aug_eqv_results]

            individual_losses_low_dimensions = [[], [], []]

            # PCA
            pca_dim = 3
            pca = PCA(n_components=pca_dim, svd_solver='full')
            for i in range(len(individual_losses)):
                for j in plotting_individual_digits:
                    # individual_losses[i][j] has shape num_degrees * num_digit_samples, thus transpose before feeding into pca
                    individual_losses_low_dimensions[i].append(pca.fit_transform(individual_losses[i][j].transpose()))

            # plot scatter plot
            all_marker_styles = ['circle', 'square', 'diamond']
            pca_result_path = os.path.join(figs_dir, f'mnist_{mode}_{type}_pca_{pca_dim}d_scatter.html')
            # marker type
            vis.plot_interactive_scatter(pca_dim,
                                        plotting_individual_digits,
                                        method_labels,
                                        all_colors,
                                        all_marker_styles,
                                        individual_losses_low_dimensions,
                                        plot_title,
                                        pca_result_path)

        elif type == 'shift':
            # print out average error per digit for both models
            print(non_eqv_result.shape)
            for i in range(non_eqv_result.shape[2]):
                cur_non_eqv_avg = np.mean(non_eqv_result[:, :, i])
                cur_eqv_avg = np.mean(eqv_result[:, :, i])
                cur_aug_eqv_avg = np.mean(aug_eqv_result[:, i])
                print(f'\nDigit {i} avg loss for Non-eqv model: {cur_non_eqv_avg}')
                print(f'Digit {i} avg loss for Eqv model: {cur_eqv_avg}')
                print(f'Digit {i} avg loss for Aug-Eqv model: {cur_aug_eqv_avg}')

            # make and save the plot
            plotting_digits = [2, 3, 5]
            plot_title = None
            result_path = os.path.join(figs_dir, f'mnist_{mode}_{type}.html')
            vis.plot_interactive_heatmap(plotting_digits,
                                            ['CNN', 'Anti-Aliasing', 'CNN+Augmentation'],
                                            [non_eqv_result, eqv_result, aug_eqv_result],
                                            plot_title,
                                            result_path)

        elif type == 'scale':
            # print out average error per digit for both models
            print(non_eqv_result.shape)
            for i in range(non_eqv_result.shape[1]):
                cur_non_eqv_avg = np.mean(non_eqv_result[:, i])
                cur_eqv_avg = np.mean(eqv_result[:, i])
                cur_aug_eqv_avg = np.mean(aug_eqv_result[:, i])
                print(f'\nDigit {i} avg loss for Non-eqv model: {cur_non_eqv_avg}')
                print(f'Digit {i} avg loss for SESN model: {cur_eqv_avg}')
                print(f'Digit {i} avg loss for Aug-Eqv model: {cur_aug_eqv_avg}')

            # make and save the plot
            all_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
            # plot interactive polar line plots
            result_path = os.path.join(figs_dir, f'mnist_{mode}_{type}.html')
            # plot_title = f'Non-equivariant vs Equivariant model on {data_name} Digit {i}'
            plot_title = None
            plotting_digits = list(range(10))
            vis.plot_interactive_line(plotting_digits, all_scales, ['Non-Eqv', 'SESN', 'Aug-Eqv'], all_colors, [non_eqv_result, eqv_result, aug_eqv_result], plot_title, result_path)


if __name__ == '__main__':
    main()