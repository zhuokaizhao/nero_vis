# the script gets called by nero_app when running the model
from cgi import test
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

import mnist_models

# MNIST dataset for running in aggregate mode
class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels, transform=None, vis=False):

        self.transform = transform
        self.vis = vis
        self.images = images
        self.labels = labels

        self.num_samples = len(self.labels)


    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        return image, label

    def __len__(self):
        return len(self.labels)


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


# digit recognition (mnist)
def load_mnist_model(network_model, model_dir):
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

    loaded_model = torch.load(model_dir, map_location=device)
    model.load_state_dict(loaded_model['state_dict'])
    # set model in evaluation mode
    model.eval()

    print(f'{network_model} loaded')

    return model

def run_mnist_once(model, test_image, test_label=None, batch_size=None):

    # print('Running inference on the single image')
    if len(test_image.shape) == 3:
        # reformat input image from (height, width, channel) to (batch size, channel, height, width)
        test_image = test_image.permute((2, 0, 1))[None, :, :, :].float()
    elif len(test_image.shape) == 4:
        # reformat input image from (batch_size, height, width, channel) to (batch_size, batch size, channel, height, width)
        test_image = test_image.permute((0, 3, 1, 2)).float()
    else:
        raise Exception('Wrong input image shape')

    # basic settings for pytorch
    if torch.cuda.is_available():
        # device set up
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        # single image mode
        if test_image.shape[0] == 1:
            batch_time_start = time.time()
            # prepare current batch's testing data
            test_image = test_image.to(device)

            # inference
            output = model(test_image).cpu().detach().numpy()[0]

            batch_time_end = time.time()
            batch_time_cost = batch_time_end - batch_time_start
            # print(f'Inference time: {batch_time_cost} seconds')

            return output

        elif len(test_image.shape) == 4 and batch_size:
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

            test_kwargs = {'batch_size': batch_size,
                            'num_workers': 8,
                            'pin_memory': True,
                            'shuffle': False}

            # generate dataset and data loader
            dataset = MnistDataset(test_image, test_label)
            test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

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


# object detection (coco)
def load_coco_model(network_model, model_dir):
    print(network_model)
# run mnist model on all rotations and return all the results
# def run_mnist_all_rotations(model, test_image):
#     exit()