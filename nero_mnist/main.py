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
from pytorch_model_summary import summary
from torch.optim.lr_scheduler import StepLR

from PIL import Image


import vis
import models
import datasets

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
    general_batch_losses = []
    general_num_correct = 0
    # categorical number of correct
    categorical_num_digits = np.zeros(10, dtype=int)
    categorical_num_correct = np.zeros(10, dtype=int)
    # categorical loss
    categorical_losses = np.zeros(10)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            batch_time_start = time.time()
            # print(target)
            # prepare current batch's testing data
            data, target = data.to(device), target.to(device)

            # inference
            output = model(data)
            loss = loss_function(output, target)
            general_batch_losses.append(torch.mean(loss).item())

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            general_num_correct += pred.eq(target.view_as(pred)).sum().item()

            # calculate per-class performance
            # total number of cases for each class (digit)
            for i in range(10):
                categorical_num_digits[i] += int(target.cpu().tolist().count(i))

            # number of correct for each case
            for i in range(len(pred)):
                # losses
                categorical_losses[target.cpu().tolist()[i]] += loss[i]
                # correctness
                if pred.eq(target.view_as(pred))[i].item() == True:
                    categorical_num_correct[target.cpu().tolist()[i]] += 1

            batch_time_end = time.time()
            batch_time_cost = batch_time_end - batch_time_start

            print_progress_bar(iteration=batch_idx+1,
                                total=len(test_loader),
                                prefix=f'Test batch {batch_idx+1}/{len(test_loader)},',
                                suffix='%s: %.3f, time: %.2f' % ('CE loss', general_batch_losses[-1], batch_time_cost),
                                length=50)

    general_loss = np.mean(general_batch_losses)
    general_accuracy = general_num_correct / len(test_loader.dataset)
    categorical_loss = categorical_losses / categorical_num_digits
    categorical_accuracy = categorical_num_correct / categorical_num_digits

    # return the averaged batch loss
    return general_loss, general_accuracy, categorical_loss, categorical_accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='NERO plots with MNIST')
    # mode (train, test or analyze)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # type of equivariance (rotation, shift, or scale)
    parser.add_argument('--type', required=True, action='store', nargs=1, dest='type')
    # network method (non-eqv, eqv, etc)
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
        if image_size == None:
            image_size = (59, 59)

    # training mode
    if mode == 'train':

        # default train_batch_size and val_batch_size if not assigned
        if args.train_batch_size == None:
            train_batch_size = 64
            val_batch_size = 1000
        else:
            train_batch_size = args.train_batch_size
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

        # when training for scale equivariance tests
        # elif type == 'scale':
        #     if image_size == None:
        #         image_size =

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

        # data transform
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(((image_size[1]-28)//2, (image_size[0]-28)//2, (image_size[1]-28)//2+1, (image_size[0]-28)//2+1), fill=0, padding_mode='constant'),
        ])

        # split into train and validation dataset
        dataset = datasets.MnistDataset(mode='train', transform=transform)
        train_data, val_data = torch.utils.data.random_split(dataset, [50000, 10000])

        # pytorch data loader
        train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_data, **val_kwargs)

        # model
        starting_epoch = 0
        if type == 'rotation':
            if network_model == 'non-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
            elif network_model == 'rot-eqv':
                # number of groups for e2cnn
                num_rotation = 8
                model = models.Rot_Eqv_Net_MNIST(image_size=image_size, num_rotation=num_rotation).to(device)
            else:
                raise Exception(f'Wrong network model type {network_model}')

        elif type == 'shift':
            if network_model == 'non-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
            elif network_model == 'shift-eqv':
                model = models.Shift_Eqv_Net_MNIST().to(device)
            else:
                raise Exception(f'Wrong network model type {network_model}')

        elif type == 'scale':
            if network_model == 'non-eqv':
                model = models.Non_Eqv_Net_MNIST(type).to(device)
            elif network_model == 'scale-eqv':
                model = models.Scale_Eqv_Net_MNIST().to(device)

        # visualize the model structure
        if network_model == 'non-eqv':
            print(summary(model, torch.zeros((1, 1, image_size[1], image_size[0])).to(device)))

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
            print(f'\n Starting epoch {epoch+1}/{starting_epoch+num_epoch}')
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

            if (epoch+1) % args.checkpoint_interval == 0:

                # save model as a checkpoint so further training could be resumed
                if network_model == 'non-eqv' or network_model == 'shift-eqv' or network_model == 'scale-eqv':
                    model_path = os.path.join(model_dir, f'{network_model}_mnist_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
                elif network_model == 'rot-eqv':
                    model_path = os.path.join(model_dir, f'{network_model}_rot-group{num_rotation}_mnist_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
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
        if network_model == 'non-eqv':
            model = models.Non_Eqv_Net_MNIST(type).to(device)

        elif network_model == 'rot-eqv':
            # number of groups for e2cnn
            num_rotation = 8
            model = models.Rot_Eqv_Net_MNIST(image_size=image_size, num_rotation=num_rotation).to(device)

        elif network_model == 'shift-eqv':
            model = models.Shift_Eqv_Net_MNIST().to(device)

        elif network_model == 'scale-eqv':
            model = models.Scale_Eqv_Net_MNIST().to(device)

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
            general_loss_heatmap = np.zeros(max_rot//increment)
            general_accuracy_heatmap = np.zeros(max_rot//increment)
            categorical_loss_heatmap = np.zeros((max_rot//increment, 10))
            categorical_accuracy_heatmap = np.zeros((max_rot//increment, 10))

            for k in range(0, max_rot, increment):
                print(f'\n Testing rotation angle {k}/{max_rot-increment}')
                # row
                transform = torchvision.transforms.Compose([
                    # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
                    torchvision.transforms.Resize(28*3),
                    torchvision.transforms.RandomRotation((k, k), resample=Image.BILINEAR, expand=False),
                    torchvision.transforms.Resize(28),
                    # torchvision.transforms.Pad((i, i, image_size[1]-28-i, image_size[0]-28-i), fill=0, padding_mode='constant'),
                    torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant'),
                ])

                # prepare test dataset
                dataset = datasets.MnistDataset(mode='test', transform=transform, vis=vis_data)
                test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

                # test the model
                general_loss, general_accuracy, categorical_loss, categorical_accuracy = test(model, device, test_loader)
                general_loss_heatmap[k//increment] = general_loss
                general_accuracy_heatmap[k//increment] = general_accuracy
                categorical_loss_heatmap[k//increment, :] = categorical_loss
                categorical_accuracy_heatmap[k//increment, :] = categorical_accuracy

                print(f'Rotation {k} accuracy: {general_accuracy}')

            # save the accuracy as npz file
            if network_model == 'non-eqv':
                loss_path = os.path.join(figs_dir, f'{network_model}_mnist_{image_size[0]}x{image_size[1]}_angle-increment{increment}_epoch{trained_epoch}_result.npz')
            elif network_model == 'rot-eqv':
                loss_path = os.path.join(figs_dir, f'{network_model}_rot-group{num_rotation}_mnist_{image_size[0]}x{image_size[1]}_angle-increment{increment}_epoch{trained_epoch}_result.npz')

            np.savez(loss_path,
                    general_loss_heatmap=general_loss_heatmap,
                    general_accuracy_heatmap=general_accuracy_heatmap,
                    categorical_loss_heatmap=categorical_loss_heatmap,
                    categorical_accuracy_heatmap=categorical_accuracy_heatmap)

            print(f'\nTesting result has been saved to {loss_path}')

        # test on each shift from -20 to 20
        elif type == 'shift':
            # shift use the image coordinate system
            # |-----> x
            # |
            # |
            # v y
            shift_amount = (image_size[0] - 29)//2
            general_loss_heatmap = np.zeros((shift_amount*2+1, shift_amount*2+1))
            general_accuracy_heatmap = np.zeros((shift_amount*2+1, shift_amount*2+1))
            categorical_loss_heatmap = np.zeros((shift_amount*2+1, shift_amount*2+1, 10))
            categorical_accuracy_heatmap = np.zeros((shift_amount*2+1, shift_amount*2+1, 10))
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
                    general_loss, general_accuracy, categorical_loss, categorical_accuracy = test(model, device, test_loader)
                    general_loss_heatmap[y_shift+shift_amount, x_shift+shift_amount] = general_loss
                    general_accuracy_heatmap[y_shift+shift_amount, x_shift+shift_amount] = general_accuracy
                    categorical_loss_heatmap[y_shift+shift_amount, x_shift+shift_amount, :] = categorical_loss
                    categorical_accuracy_heatmap[y_shift+shift_amount, x_shift+shift_amount, :] = categorical_accuracy

                    print(f'shift amount x = {x_shift}, y = {y_shift} accuracy: {general_accuracy}')

            # save the accuracy as npz file
            loss_path = os.path.join(figs_dir, f'{network_model}_mnist_{image_size[0]}x{image_size[1]}_epoch{trained_epoch}_result.npz')
            np.savez(loss_path,
                    general_loss_heatmap=general_loss_heatmap,
                    general_accuracy_heatmap=general_accuracy_heatmap,
                    categorical_loss_heatmap=categorical_loss_heatmap,
                    categorical_accuracy_heatmap=categorical_accuracy_heatmap)

            print(f'\nTesting result has been saved to {loss_path}')

        # test on each scale from 0.5 to 2
        elif type == 'scale':
            all_scales = list(np.arange(0.5, 2.1, 0.1))
            general_loss_heatmap = np.zeros(len(all_scales))
            general_accuracy_heatmap = np.zeros(len(all_scales))
            categorical_loss_heatmap = np.zeros((len(all_scales), 10))
            categorical_accuracy_heatmap = np.zeros((len(all_scales), 10))

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
                general_loss, general_accuracy, categorical_loss, categorical_accuracy = test(model, device, test_loader)
                general_loss_heatmap[k] = general_loss
                general_accuracy_heatmap[k] = general_accuracy
                categorical_loss_heatmap[k, :] = categorical_loss
                categorical_accuracy_heatmap[k, :] = categorical_accuracy

                print(f'Scale {scale} accuracy: {general_accuracy}')

            # save the accuracy as npz file
            loss_path = os.path.join(figs_dir, f'{network_model}_mnist_{image_size[0]}x{image_size[1]}_epoch{trained_epoch}_result.npz')
            np.savez(loss_path,
                    general_loss_heatmap=general_loss_heatmap,
                    general_accuracy_heatmap=general_accuracy_heatmap,
                    categorical_loss_heatmap=categorical_loss_heatmap,
                    categorical_accuracy_heatmap=categorical_accuracy_heatmap)

            print(f'\nTesting result has been saved to {loss_path}')

    elif 'analyze' in mode:
        non_eqv_path = args.non_eqv_path[0]
        eqv_path = args.eqv_path[0]
        if verbose:
            print(f'\nmode: {mode}')
            print(f'input non-eqv result dir: {non_eqv_path}')
            print(f'input eqv result dir: {eqv_path}')
            print(f'output figures dir: {figs_dir}')

        # load the non-eqv and eqv result
        non_eqv_result = np.load(non_eqv_path)
        eqv_result = np.load(eqv_path)

        if type == 'rotation':
            # print out average error per digit for both models
            print(non_eqv_result['categorical_accuracy_heatmap'].shape)
            for i in range(non_eqv_result['categorical_accuracy_heatmap'].shape[1]):
                cur_non_eqv_avg = np.mean(non_eqv_result['categorical_accuracy_heatmap'][:, i])
                cur_eqv_avg = np.mean(eqv_result['categorical_accuracy_heatmap'][:, i])
                print(f'\nDigit {i} avg accuracy for Non-eqv model: {cur_non_eqv_avg}')
                print(f'Digit {i} avg accuracy for Eqv model: {cur_eqv_avg}')

            # make and save the plot
            # plot interactive polar line plots
            result_path = os.path.join(figs_dir, f'mnist_{mode}_{type}.html')
            # plot_title = f'Non-equivariant vs Equivariant model on {data_name} Digit {i}'
            plot_title = None
            plotting_digits = list(range(10))
            vis.plot_interactive_line_polar(plotting_digits, non_eqv_result, eqv_result, plot_title, result_path)

        elif type == 'shift':
            # print out average error per digit for both models
            print(non_eqv_result['categorical_accuracy_heatmap'].shape)
            for i in range(non_eqv_result['categorical_accuracy_heatmap'].shape[2]):
                cur_non_eqv_avg = np.mean(non_eqv_result['categorical_accuracy_heatmap'][:, :, i])
                cur_eqv_avg = np.mean(eqv_result['categorical_accuracy_heatmap'][:, :, i])
                print(f'\nDigit {i} avg accuracy for Non-eqv model: {cur_non_eqv_avg}')
                print(f'Digit {i} avg accuracy for Eqv model: {cur_eqv_avg}')

            # make and save the plot
            plotting_digits = list(range(10))
            # non-eqv
            non_eqv_result_path = os.path.join(figs_dir, f'mnist_{mode}_{type}_non_eqv.html')
            non_eqv_plot_title = None
            vis.plot_interactive_heatmap('Non-Eqv', plotting_digits, non_eqv_result, non_eqv_plot_title, non_eqv_result_path)
            # eqv
            eqv_result_path = os.path.join(figs_dir, f'mnist_{mode}_{type}_eqv.html')
            eqv_plot_title = None
            vis.plot_interactive_heatmap('Shift-Eqv', plotting_digits, eqv_result, eqv_plot_title, eqv_result_path)


if __name__ == '__main__':
    main()