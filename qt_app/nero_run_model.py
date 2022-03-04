# the script gets called by nero_app when running the model
import os
import sys
import time
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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

        if self.transform is not None:
            image = self.transform(image)

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

    loaded_model = torch.load(model_dir, map_location=device)
    model.load_state_dict(loaded_model['state_dict'])
    # set model in evaluation mode
    model.eval()

    print(f'{network_model} loaded')

    return model


# run model on either on a single MNIST image or a batch of MNIST images
def run_mnist_once(model, test_image, test_label=None, batch_size=None, rotate_angle=None):

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
            all_output = np.zeros((len(test_image), 10))
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
            if rotate_angle:
                img_size = 29

                # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(img_size*3),
                    torchvision.transforms.RandomRotation(degrees=(rotate_angle, rotate_angle), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, expand=False),
                    torchvision.transforms.Resize(img_size),
                ])
            else:
                transform = None

            # create angle
            dataset = MnistDataset(test_image, test_label, transform=transform)
            test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

            for batch_idx, (data, target) in enumerate(test_loader):
                batch_time_start = time.time()
                # print(target)
                # prepare current batch's testing data
                data, target = data.to(device), target.to(device)

                # inference
                output = model(data)
                all_output[batch_idx*batch_size:(batch_idx+1)*batch_size, :] = output.cpu().detach().numpy()
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

                # print_progress_bar(iteration=batch_idx+1,
                #                     total=len(test_loader),
                #                     prefix=f'Test batch {batch_idx+1}/{len(test_loader)},',
                #                     suffix='%s: %.3f, time: %.2f' % ('CE loss', all_batch_avg_losses[-1], batch_time_cost),
                #                     length=50)

            # average (over digits) loss and accuracy
            avg_accuracy = total_num_correct / len(test_loader.dataset)
            # each digits loss and accuracy
            avg_accuracy_per_digit = num_correct_per_digit / num_digits_per_digit

            # return the averaged batch loss
            return avg_accuracy, avg_accuracy_per_digit, all_output


# object detection (coco)
def load_coco_model(model_dir, num_classes = 5):

    # basic settings for pytorch
    if torch.cuda.is_available():
        # device set up
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # initiate model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                num_classes=num_classes+1,
                                                                pretrained_backbone=True,
                                                                min_size=128)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    # load the weight
    loaded_model = torch.load(model_dir, map_location=device)
    model.load_state_dict(loaded_model['state_dict'])

    # set model in evaluation mode
    model.eval()

    return model


# run model on either on a single COCO image or a batch of COCO images
def run_coco_once(model, test_image, test_label=None, batch_size=None):
    print('aaa')

