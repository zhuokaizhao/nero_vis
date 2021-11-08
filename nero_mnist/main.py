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
    parser = argparse.ArgumentParser(description='Visualizing rotation equivariance')
    # mode (data, train, or test mode)
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
    # network method (non-eqv, eqv, etc)
    parser.add_argument('-n', '--network-model', action='store', nargs=1, dest='network_model')
    # dataset
    parser.add_argument('-d', '--data-name', action='store', nargs=1, dest='data_name')
    # image dimension after padding
    parser.add_argument('-s', '--image-size', action='store', nargs=1, dest='image_size')
    # training batch size
    parser.add_argument('--train-batch-size', type=int, dest='train_batch_size',
                        help='input batch size for training')
    # validation batch size
    parser.add_argument('--val-batch-size', type=int, dest='val_batch_size',
                        help='input batch size for validation')
    # test batch size
    parser.add_argument('--test-batch-size', type=int, dest='test_batch_size',
                        help='input batch size for testing')
    # number of training epochs
    parser.add_argument('-e', '--num-epoch', type=int, dest='num_epoch',
                        help='number of epochs to train')
    # checkpoint path for continuing training
    parser.add_argument('-c', '--checkpoint-dir', action='store', nargs=1, dest='checkpoint_path')
    # input or output model directory
    parser.add_argument('-m', '--model-dir', action='store', nargs=1, dest='model_dir')
    # output directory (tfrecord in 'data' mode, figure in 'training' mode)
    parser.add_argument('-o', '--output-dir', action='store', nargs=1, dest='output_dir')
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)

    args = parser.parse_args()

    # input variables
    mode = args.mode[0]
    data_name = args.data_name[0]
    if args.image_size != None:
        image_size = (int(args.image_size[0].split('x')[0]), int(args.image_size[0].split('x')[1]))
    else:
        image_size = None
    # output graph directory
    figs_dir = args.output_dir[0]
    verbose = args.verbose

    # training mode
    if mode == 'train':
        network_model = args.network_model[0]
        # default train_batch_size and val_batch_size if not assigned
        if args.train_batch_size == None:
            if data_name == 'mnist' or data_name == 'mnist-rot':
                train_batch_size = 64
                if 'rot' in network_model:
                    val_batch_size = 100
                else:
                    val_batch_size = 1000
            elif data_name == 'cifar-10':
                train_batch_size = 128
                if 'rot' in network_model:
                    val_batch_size = 200
                else:
                    val_batch_size = 500
        else:
            train_batch_size = args.train_batch_size
            val_batch_size = args.val_batch_size

        # number of training epoch
        num_epoch = args.num_epoch

        # number of groups for e2cnn
        if network_model == 'rot-eqv' or 'trans-rot-eqv':
            num_rotation = 8

        # assign learning rate for different models
        # MNIST models use Adam
        if data_name == 'mnist' or data_name == 'mnist-rot':
            learning_rate = 5e-5
            weight_decay = 1e-5
        # VGG uses SGD
        elif data_name == 'cifar-10':
            learning_rate = 0.001
            momentum = 0.9
            weight_decay = 5e-4

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
            cuda_kwargs = {'num_workers': 1,
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
            print(f'dataset: {data_name}')
            if image_size != None:
                print(f'Padded image size: {image_size[0]}x{image_size[1]}')
            else:
                print(f'Padded image size: {image_size}')
            print(f'training batch size: {train_batch_size}')
            print(f'validation batch size: {val_batch_size}')
            print(f'output model dir: {model_dir}')
            print(f'epoch size: {num_epoch}')
            print(f'initial learning rate: {learning_rate}')
            print(f'optimizer weight decay: {weight_decay}')
            if data_name == 'cifar-10':
                print(f'optimizer momentum: {momentum}')
            print(f'input checkpoint path: {checkpoint_path}')

        if data_name == 'mnist':
            # transform
            if image_size != None:
                transform = torchvision.transforms.Compose([
                    # transform includes padding (right and bottom) to image_size and totensor
                    torchvision.transforms.Pad((0, 0, image_size[1]-28, image_size[0]-28), fill=0, padding_mode='constant'),
                ])
            else:
                transform = torchvision.transforms.Compose([
                    # transform includes padding (right and bottom) to 29x29 (for model purpose) and totensor
                    torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant'),
                ])
                image_size = (29, 29)

            # split into train and validation dataset
            dataset = datasets.MnistDataset(mode='train', transform=transform)
            train_data, val_data = torch.utils.data.random_split(dataset, [50000, 10000])

        elif data_name == 'mnist-rot':
            # transform
            if image_size != None:
                transform = torchvision.transforms.Compose([
                    # transform includes padding (right and bottom) to image_size and totensor
                    torchvision.transforms.Pad((0, 0, image_size[1]-28, image_size[0]-28), fill=0, padding_mode='constant'),
                ])
            else:
                transform = torchvision.transforms.Compose([
                    # transform includes padding (right and bottom) to 29x29 (for model purpose) and totensor
                    torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant'),
                ])

            # split into train and validation dataset
            dataset = datasets.MnistRotDataset(mode='train', transform=transform)
            train_data, val_data = torch.utils.data.random_split(dataset, [10000, 2000])

        elif data_name == 'cifar-10':
            # split into train and validation dataset
            # pad the image from 32x32 to image_size on right and bottom (left, top, right and bottom)
            if image_size != None:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Pad((0, 0, image_size[1]-32, image_size[0]-32), fill=0, padding_mode='constant'),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            else:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            dataset = torchvision.datasets.CIFAR10(root='/home/zhuokai/Desktop/nvme1n1p1/Data/CIFAR-10/',
                                                    train=True,
                                                    download=True,
                                                    transform=transform)

            train_data, val_data = torch.utils.data.random_split(dataset, [45000, 5000])

        # pytorch data loader
        train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_data, **val_kwargs)

        # model
        starting_epoch = 0
        if data_name == 'mnist':
            if network_model == 'non-eqv':
                model = models.Non_Eqv_Net_MNIST().to(device)

            elif network_model == 'trans-eqv':
                model = models.Trans_Eqv_Net_MNIST().to(device)

            elif network_model == 'rot-eqv':
                model = models.Rot_Eqv_Net_MNIST(image_size=image_size, num_rotation=num_rotation).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        elif data_name == 'cifar-10':
            if network_model == 'non-eqv':
                model = models.vgg19_bn(network_model).to(device)

            elif network_model == 'trans-eqv':
                model = models.vgg19_bn(network_model).to(device)

            elif network_model == 'rot-eqv':
                model = models.vgg11_bn(network_model, image_size, num_groups=2).to(device)

            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

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

        # save model as a checkpoint so further training could be resumed
        if network_model == 'non-eqv':
            model_path = os.path.join(model_dir, f'{network_model}_{data_name}_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
        elif network_model == 'trans-eqv':
            model_path = os.path.join(model_dir, f'{network_model}_{data_name}_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
        elif network_model == 'rot-eqv':
            model_path = os.path.join(model_dir, f'{network_model}_rot-group{num_rotation}_{data_name}_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
        elif network_model == 'trans-rot-eqv':
            model_path = os.path.join(model_dir, f'{network_model}_rot-group{num_rotation}_{data_name}_{image_size[0]}x{image_size[1]}_batch{train_batch_size}_epoch{epoch+1}.pt')
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
        loss_path = os.path.join(figs_dir, f'{network_model}_{data_name}_batch{train_batch_size}_epoch{epoch+1}_loss.png')
        plt.savefig(loss_path)
        print(f'\nLoss graph has been saved to {loss_path}')


    elif mode == 'test':
        test_batch_size = args.test_batch_size
        network_model = args.network_model[0]
        # directory to load the model from
        model_dir = args.model_dir[0]
        test_kwargs = {'batch_size': test_batch_size}

        # number of groups for e2cnn
        if network_model == 'rot-eqv' or 'trans-rot-eqv':
            num_rotation = 8

        # basic settings for pytorch
        if torch.cuda.is_available():
            # additional kwargs
            cuda_kwargs = {'num_workers': 1,
                            'pin_memory': True,
                            'shuffle': True}
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
        if data_name == 'mnist':
            if network_model == 'non-eqv':
                model = models.Non_Eqv_Net_MNIST().to(device)

            elif network_model == 'trans-eqv':
                model = models.Trans_Eqv_Net_MNIST().to(device)

            elif network_model == 'rot-eqv':
                model = models.Rot_Eqv_Net_MNIST(image_size=image_size, num_rotation=num_rotation).to(device)

        elif data_name == 'cifar-10':
            if network_model == 'non-eqv':
                model = models.vgg19_bn(network_model, image_size=image_size).to(device)

            elif network_model == 'trans-eqv':
                model = models.vgg19_bn(network_model, image_size=image_size).to(device)

            elif network_model == 'rot-eqv':
                model = models.vgg19_bn(network_model, image_size=image_size, num_groups=8).to(device)

        # load previously trained model
        trained_model = torch.load(model_dir)
        model.load_state_dict(trained_model['state_dict'])
        trained_epoch = trained_model['epoch']

        all_degree_general_losses = []
        all_degree_general_accuracy = []
        all_degree_categorical_losses = []
        all_degree_categorical_accuracy = []

        if verbose:
            print(f'\nmode: {mode}')
            print(f'GPU usage: {device}')
            print(f'netowrk model: {network_model}')
            print(f'dataset: {data_name}')
            print(f'testing batch size: {test_batch_size}')
            print(f'input model dir: {model_dir}')

        # suppose the padding size is 64x64, mnist has 28x28 image, cifar has 32x32 image
        # if data_name == 'mnist' or data_name == 'mnist-rot':
        #     num_move = image_size[0] - 28
        # elif data_name == 'cifar-10':
        #     num_move = image_size[0] - 32
        # elif data_name == 'cifar-100':
        #     raise NotImplementedError('Not ready yet')
        if image_size == None:
            image_size = (29, 29)
        # test on each rotation degree
        # rotation
        max_rot = 360
        increment = 1
        # plot error at each moved position as a heatmap
        # general_loss_heatmap = np.zeros((max_rot//increment, num_move))
        # general_accuracy_heatmap = np.zeros((max_rot//increment, num_move))
        # categorical_loss_heatmap = np.zeros((max_rot//increment, num_move, 10))
        # categorical_accuracy_heatmap = np.zeros((max_rot//increment, num_move, 10))
        general_loss_heatmap = np.zeros(max_rot//increment)
        general_accuracy_heatmap = np.zeros(max_rot//increment)
        categorical_loss_heatmap = np.zeros((max_rot//increment, 10))
        categorical_accuracy_heatmap = np.zeros((max_rot//increment, 10))
        for k in range(0, max_rot, increment):
            print(f'\n Testing rotation angle {k}/{max_rot-increment}')
            # consider we are changing distance as variable, row == col
            # row
            # for i in range(num_move):
            #     print(f'\nRow {i}/{num_move-1}, Column {i}/{num_move-1}')
            if data_name == 'mnist':
                transform = torchvision.transforms.Compose([
                    # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
                    torchvision.transforms.Resize(28*3),
                    torchvision.transforms.RandomRotation((k, k), resample=Image.BILINEAR, expand=False),
                    torchvision.transforms.Resize(28),
                    # torchvision.transforms.Pad((i, i, image_size[1]-28-i, image_size[0]-28-i), fill=0, padding_mode='constant'),
                    torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant'),
                ])

                # split into train and validation dataset
                dataset = datasets.MnistDataset(mode='test', transform=transform)
                test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

            elif data_name == 'mnist-rot':
                transform = torchvision.transforms.Compose([
                    # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
                    torchvision.transforms.Resize(28*3),
                    torchvision.transforms.RandomRotation((k, k), resample=Image.BILINEAR, expand=False),
                    torchvision.transforms.Resize(28),
                    torchvision.transforms.Pad((i, i, image_size[1]-28-i, image_size[0]-28-i), fill=0, padding_mode='constant'),
                ])

                # split into train and validation dataset
                dataset = datasets.MnistRotDataset(mode='test', transform=transform)
                test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

            elif data_name == 'cifar-10':
                # prepare the image in the correct position
                transform = torchvision.transforms.Compose([
                    # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
                    torchvision.transforms.Resize(32*3),
                    torchvision.transforms.RandomRotation((k, k), expand=False, fill=0),
                    torchvision.transforms.Resize(32),
                    # (left, top, right and bottom)
                    torchvision.transforms.Pad((i, i, image_size[1]-32-i, image_size[0]-32-i), fill=0, padding_mode='constant'),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                test_data = torchvision.datasets.CIFAR10(root='/home/zhuokai/Desktop/nvme1n1p1/Data/CIFAR-10/',
                                                        train=False,
                                                        download=True,
                                                        transform=transform)

                test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

            elif data_name == 'cifar-100':
                raise NotImplementedError('Not ready yet')

            # test the model
            general_loss, general_accuracy, categorical_loss, categorical_accuracy = test(model, device, test_loader)
            general_loss_heatmap[k//increment] = general_loss
            general_accuracy_heatmap[k//increment] = general_accuracy
            categorical_loss_heatmap[k//increment, :] = categorical_loss
            categorical_accuracy_heatmap[k//increment, :] = categorical_accuracy

            print(f'Rotation {k} accuracy: {general_accuracy}')

        # save the accuracy as npz file
        if network_model == 'non-eqv':
            loss_path = os.path.join(figs_dir, f'{network_model}_{data_name}_{image_size[0]}x{image_size[1]}_angle-increment{increment}_epoch{trained_epoch}_result.npz')
        elif network_model == 'trans-eqv':
            loss_path = os.path.join(figs_dir, f'{network_model}_{data_name}_{image_size[0]}x{image_size[1]}_angle-increment{increment}_epoch{trained_epoch}_result.npz')
        elif network_model == 'rot-eqv':
            loss_path = os.path.join(figs_dir, f'{network_model}_rot-group{num_rotation}_{data_name}_{image_size[0]}x{image_size[1]}_angle-increment{increment}_epoch{trained_epoch}_result.npz')
        elif network_model == 'trans-rot-eqv':
            loss_path = os.path.join(figs_dir, f'{network_model}_rot-group{num_rotation}_{data_name}_{image_size[0]}x{image_size[1]}_angle-increment{increment}_epoch{trained_epoch}_result.npz')

        np.savez(loss_path,
                general_loss_heatmap=general_loss_heatmap,
                general_accuracy_heatmap=general_accuracy_heatmap,
                categorical_loss_heatmap=categorical_loss_heatmap,
                categorical_accuracy_heatmap=categorical_accuracy_heatmap)

        print(f'\nTesting result has been saved to {loss_path}')


    elif mode == 'analyze':
        if verbose:
            print(f'\nmode: {mode}')
            print(f'dataset: {data_name}')
            print(f'output figures dir: {figs_dir}')

        # load the non-eqv result
        non_eqv_dir = '/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/classification/mnist/figs/non_eqv'
        non_eqv_name = 'non-eqv_mnist_29x29_angle-increment1_epoch50_result.npz'
        non_eqv_path = os.path.join(non_eqv_dir, non_eqv_name)
        non_eqv_result = np.load(non_eqv_path)

        # load the trans-eqv result
        # trans_eqv_dir = '/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/classification/mnist/figs/trans_eqv'
        # trans_eqv_name = 'trans-eqv_mnist_64x64_angle-increment1_epoch50_result.npz'
        # trans_eqv_path = os.path.join(trans_eqv_dir, trans_eqv_name)
        # trans_eqv_result = np.load(trans_eqv_path)

        # load the rot-eqv result
        rot_eqv_dir = '/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/classification/mnist/figs/rot_eqv'
        rot_eqv_name = 'rot-eqv_rot-group8_mnist_29x29_angle-increment1_epoch30_result.npz'
        rot_eqv_path = os.path.join(rot_eqv_dir, rot_eqv_name)
        rot_eqv_result = np.load(rot_eqv_path)


        # test for each individual digit
        for i in range(10):
            plot_title = f'Non-equivariant vs Equivariant model on {data_name} Digit {i}'
            result_path = os.path.join(figs_dir, f'{data_name}_test_digit_{i}.html')

            # vis.plot_interactive_polar_heatmap(data_name, non_eqv_result, trans_eqv_result, rot_eqv_result, plot_title, result_path)
            vis.plot_interactive_line_polar(i, non_eqv_result, rot_eqv_result, plot_title, result_path)



if __name__ == '__main__':
    main()