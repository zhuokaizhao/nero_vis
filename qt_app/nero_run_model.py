# the script gets called by nero_app when running the model
from dataclasses import dataclass
import os
import sys
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

os.environ['CUDA_VISIBLE_DEVICES']='0'

# add parent directories for importing model scripts
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import models
import datasets
import nero_utilities
import nero_transform

# helper function that resizes the image
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


# digit recognition (mnist)
def load_model(mode, network_model, model_dir):

    # basic settings for pytorch
    if torch.cuda.is_available():
        # device set up
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if mode == 'digit_recognition':
        # model
        if network_model == 'non_eqv' or network_model == 'aug_eqv':
            model = models.Non_Eqv_Net_MNIST('rotation').to(device)

        elif network_model == 'rot_eqv':
            # number of groups for e2cnn
            num_rotation = 8
            model = models.Rot_Eqv_Net_MNIST(image_size=None, num_rotation=num_rotation).to(device)

        loaded_model = torch.load(model_dir, map_location=device)
        model.load_state_dict(loaded_model['state_dict'])

    elif mode == 'object_detection':
        num_classes = 5
        image_size = 128
        if network_model == 'custom_trained':
            # model = models.Custom_Trained_FastRCNN()
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                            num_classes=num_classes+1,
                                                                            pretrained_backbone=True,
                                                                            min_size=image_size)
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
            model.to(device)
            model.load_state_dict(torch.load(model_dir, map_location=device))
            print(f'{network_model} loaded from {model_dir}')

        elif network_model == 'pre_trained':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                        min_size=image_size).to(device)

            print(f'{network_model} loaded from PyTorch')


    # set model in evaluation mode
    model.eval()

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
            # prepare current batch's testing data
            test_image = test_image.to(device)

            # inference
            output = model(test_image).cpu().detach().numpy()[0]

            return output

        # multiple images (aggregated) mode
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
            dataset = datasets.MnistDataset(test_image, test_label, transform=transform)
            test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

            for batch_idx, (data, target) in enumerate(test_loader):
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

            # average (over digits) loss and accuracy
            avg_accuracy = total_num_correct / len(test_loader.dataset)
            # each digits loss and accuracy
            avg_accuracy_per_digit = num_correct_per_digit / num_digits_per_digit

            # return the averaged batch loss
            return avg_accuracy, avg_accuracy_per_digit, all_output


# helper function on processing coco model output
def process_model_outputs(outputs, targets, iou_thres=0.5, conf_thres=0):

    def bbox_iou(box1, box2):
        """
        Returns the IoU of two bounding boxes
        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    # get all the outputs (the model proposes multiple bounding boxes for each object)
    all_qualified_outputs = []
    all_precisions = []
    all_recalls = []
    all_F_measure = []
    # loop through each proposed object
    for i in range(len(outputs)):

        if outputs[i] is None:
            continue

        # output has to pass the conf threshold to filter out noisy predictions
        # faster-rcnn output has format (x1, y1, x2, y2, conf, class_pred)
        # output is ordered by confidence
        output = outputs[i]
        cur_object_outputs = []
        for j in range(len(output)):
            if output[j, 4] >= conf_thres:
                cur_object_outputs.append(output[j].numpy())

        # convert conf-qualified results to tensor
        cur_object_outputs = torch.from_numpy(np.array(cur_object_outputs))

        # all predicted labels and true label for the current object
        pred_labels = cur_object_outputs[:, -1].numpy()
        true_labels = targets[targets[:, 0] == i][:, 1].numpy()

        # all predicted bounding boxes and true bb for the current object
        pred_boxes = cur_object_outputs[:, :4]
        true_boxes = targets[targets[:, 0] == i][:, 2:]
        qualified_output = []
        num_true_positive = 0
        num_false_positive = 0

        # loop through all the proposed bounding boxes
        for j, pred_box in enumerate(pred_boxes):
            # compute iou
            cur_iou = bbox_iou(pred_box.unsqueeze(0), true_boxes)
            # even it might be wrong, we want the output
            qualified_output.append(np.append(cur_object_outputs[j].numpy(), cur_iou))

            # if the label is predicted correctly
            if pred_labels[j] == true_labels[i]:
                # if iou passed threshold
                if cur_iou > iou_thres:
                    num_true_positive += 1
                else:
                    num_false_positive += 1
            # if the label is wrong, mark as false positive
            else:
                num_false_positive += 1

        # make to numpy array
        qualified_output = np.array(qualified_output)
        # Number of TP and FP is computed from all predictions (un-iou thresheld)
        # precision = TP / (TP + FP)
        precision = num_true_positive / (num_true_positive + num_false_positive + 1e-16)
        # Recall = TP / (TP + FN), where (TP + FN) is just the number of ground truths
        recall = num_true_positive / (len(true_boxes) + 1e-16)
        # F-measure = (2 * Precision * Recall) / (Precision + Recall)
        F_measure = (2 * precision * recall) / (precision + recall + 1e-16)

        # rank all outputs based on IOU
        qualified_output = sorted(qualified_output, key=lambda x: x[-1])[::-1]
        all_qualified_outputs.append(qualified_output)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_F_measure.append(F_measure)

    all_qualified_outputs = np.array(all_qualified_outputs)
    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    all_F_measure = np.array(all_F_measure)

    return all_qualified_outputs, all_precisions, all_recalls, all_F_measure


# run model on either a single COCO image or a batch of COCO images
def run_coco_once(mode, model_name, model, test_image, custom_names, pytorch_names, test_label=None, batch_size=1, x_tran=None, y_tran=None):

    # basic settings for pytorch
    if torch.cuda.is_available():
        # device set up
        device = torch.device('cuda')
        Tensor = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        Tensor = torch.FloatTensor

    # prepare input image shapes
    if mode == 'single':
        # reformat input image from (height, width, channel) to (batch size, channel, height, width)
        test_image = test_image.permute((2, 0, 1))[None, :, :, :].float()
        with torch.no_grad():
            # single image mode
            # prepare current batch's testing data
            test_image = test_image.to(device)

            # run model inference
            outputs_dict = model(test_image)
            outputs = []
            # the model proposes multiple bounding boxes for each object
            for i in range(len(outputs_dict)):
                image_pred = outputs_dict[i]
                pred_boxes = image_pred['boxes'].cpu()
                pred_labels = image_pred['labels'].cpu()
                pred_confs = image_pred['scores'].cpu()
                # print(pred_boxes)
                if len(pred_boxes) != 0:
                    # when we are using pretrained model and ground truth label is present
                    if (model_name == 'FasterRCNN (Pre-trained)') and type(test_label) != type(None):
                        valid_indices = []
                        for j in range(len(pred_labels)):
                            if pytorch_names[int(pred_labels[j]-1)] in custom_names:
                                valid_indices.append(j)
                                pred_labels[j] = custom_names.index(pytorch_names[int(pred_labels[j]-1)]) + 1
                            else:
                                continue

                        if len(valid_indices) != 0:
                            # transform output in the format of (x1, y1, x2, y2, conf, class_pred)
                            output = torch.zeros((len(valid_indices), 6))
                            output[:, :4] = pred_boxes[valid_indices]
                            output[:, 4] = pred_confs[valid_indices]
                            output[:, 5] = pred_labels[valid_indices]
                            outputs.append(output)
                    else:
                        # transform output in the format of (x1, y1, x2, y2, conf, class_pred)
                        output = torch.zeros((len(pred_boxes), 6))
                        output[:, :4] = pred_boxes
                        output[:, 4] = pred_confs
                        output[:, 5] = pred_labels

                        outputs.append(output)

            if outputs != []:
                cur_qualified_output, cur_precision, cur_recall, cur_F_measure = process_model_outputs(outputs, torch.from_numpy(test_label))
            else:
                # no qualified output
                cur_qualified_output = np.zeros((1, 1, 7))
                cur_precision = np.zeros(1)
                cur_recall = np.zeros(1)
                cur_F_measure = np.zeros(1)

            return [cur_qualified_output, cur_precision, cur_recall, cur_F_measure]

    # multiple image (batch) mode
    elif mode == 'aggregate':
        img_size = 128
        # different transforms based on if we are evaluating with pre-trained model
        if (model_name == 'FasterRCNN (Pre-trained)'):
            AGGREGATE_TRANSFORMS = transforms.Compose([nero_transform.FixedJittering(img_size, x_tran, y_tran),
                                                        nero_transform.ConvertLabel(pytorch_names, custom_names),
                                                        nero_transform.ToTensor()])
        else:
            AGGREGATE_TRANSFORMS = transforms.Compose([nero_transform.FixedJittering(img_size, x_tran, y_tran),
                                                        nero_transform.ToTensor()])

        # in aggregate mode test_image is a path that includes paths of all the images
        dataset = datasets.COCODataset(list_path=test_image,
                                        img_size=img_size,
                                        transform=AGGREGATE_TRANSFORMS)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=dataset.collate_fn
        )

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # all the class labels ground truth
        labels = []
        # List of tuples (TP, confs, pred)
        sample_metrics = []
        all_losses = []
        for batch_i, (_, imgs, targets) in enumerate(dataloader):

            if targets is None:
                continue








    # create dataloader to take care of classes conversion, etc
    # EVALUATE_TRANSFORMS = transforms.Compose([nero_transform.ConvertLabel(original_names, custom_names),
    #                                         nero_transform.ToTensor()])

    # dataset = datasets.CreateDataset(path, img_size=img_size, transform=EVALUATE_TRANSFORMS)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=1,
    #     collate_fn=dataset.collate_fn
    # )






