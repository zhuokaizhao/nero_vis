# the script gets called by nero_app when running the model
from dataclasses import dataclass
import os
import sys
from sympy import EX
import torch
import torchvision
import numpy as np
from tqdm import tqdm
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

    elif mode == 'piv':
        if network_model == 'PIV-LiteFlowNet-en':
            model = models.PIV_LiteFlowNet_en()
            model.to(device)
            loaded_model = torch.load(model_dir, map_location=device)
            model.load_state_dict(loaded_model['state_dict'])
            print(f'{network_model} loaded from {model_dir}')

    else:
        raise Exception(f'Unrecognized mode {mode}')

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


# filter/modify the model outputs by supported classes
def clean_model_outputs(model_name, outputs_dict, original_names, desired_names):
    outputs = []
    # actually outputs_dict has length 1 because it is single image mode
    for i in range(len(outputs_dict)):
        image_pred = outputs_dict[i]
        pred_boxes = image_pred['boxes'].cpu()
        pred_labels = image_pred['labels'].cpu()
        pred_confs = image_pred['scores'].cpu()
        # print('\npred_labels', pred_labels)
        # the model proposes multiple bounding boxes for each object
        if len(pred_boxes) != 0:
            # when we are using pretrained model and ground truth label is present
            if (model_name == 'FasterRCNN (Pre-trained)'):
                valid_indices = []
                for j in range(len(pred_labels)):
                    # print('\nPred label 1', original_names[int(pred_labels[j]-1)])
                    if original_names[int(pred_labels[j]-1)] in desired_names:
                        valid_indices.append(j)
                        pred_labels[j] = desired_names.index(original_names[int(pred_labels[j]-1)]) + 1
                        # print('After conversion:', pred_labels[j])
                    else:
                        continue

                if len(valid_indices) != 0:
                    # transform output in the format of (x1, y1, x2, y2, conf, class_pred)
                    output = torch.zeros((len(valid_indices), 6))
                    output[:, :4] = pred_boxes[valid_indices]
                    output[:, 4] = pred_confs[valid_indices]
                    output[:, 5] = pred_labels[valid_indices]
                else:
                    output = torch.zeros((1, 6))
            else:
                # transform output in the format of (x1, y1, x2, y2, conf, class_pred)
                output = torch.zeros((len(pred_boxes), 6))
                output[:, :4] = pred_boxes
                output[:, 4] = pred_confs
                output[:, 5] = pred_labels
        else:
            output = torch.zeros((1, 6))

        outputs.append(output)

    return outputs


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
    all_qualified_outputs = np.zeros(len(outputs), dtype=np.ndarray)
    # all_qualified_outputs = []
    all_precisions = np.zeros(len(outputs))
    all_recalls = np.zeros(len(outputs))
    all_F_measure = np.zeros(len(outputs))

    # loop through model's outputs on each image sample
    for i in range(len(outputs)):

        if outputs[i] is None:
            continue

        # output has to pass the conf threshold to filter out noisy predictions
        # faster-rcnn output has format (x1, y1, x2, y2, conf, class_pred)
        # output is ordered by confidence from the model by default
        output = outputs[i]
        cur_object_outputs = []
        for j in range(len(output)):
            if output[j, 4] >= conf_thres:
                cur_object_outputs.append(output[j].numpy())

        # convert conf-qualified results to tensor
        cur_object_outputs = torch.from_numpy(np.array(cur_object_outputs))

        # all predicted labels and true label for the current object
        pred_labels = cur_object_outputs[:, -1].numpy()
        true_labels = targets[targets[:, 0] == i][:, 5].numpy()

        # all predicted bounding boxes and true bb for the current object
        pred_boxes = cur_object_outputs[:, :4]
        true_boxes = targets[targets[:, 0] == i][:, 1:5]
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
            if pred_labels[j] == true_labels[0]:
                # if iou passed threshold
                if cur_iou > iou_thres:
                    num_true_positive += 1
                else:
                    num_false_positive += 1
            # if the label is wrong, mark as false positive
            else:
                num_false_positive += 1

        # Number of TP and FP is computed from all predictions (un-iou thresheld)
        # precision = TP / (TP + FP)
        precision = num_true_positive / (num_true_positive + num_false_positive + 1e-16)
        # Recall = TP / (TP + FN), where (TP + FN) is just the number of ground truths
        recall = min(1, num_true_positive / (len(true_boxes) + 1e-16))
        # F-measure = (2 * Precision * Recall) / (Precision + Recall)
        F_measure = (2 * precision * recall) / (precision + recall + 1e-16)

        # rank all outputs based on IOU and then convert to numpy array
        qualified_output = np.array(sorted(qualified_output, key=lambda x: x[-1])[::-1])
        all_qualified_outputs[i] = qualified_output
        all_precisions[i] = precision
        all_recalls[i] = recall
        all_F_measure[i] = F_measure


    return all_qualified_outputs, all_precisions, all_recalls, all_F_measure


# run model on either a single COCO image or a batch of COCO images
def run_coco_once(mode, model_name, model, test_image, custom_names, pytorch_names, test_label=None, batch_size=1, x_tran=None, y_tran=None, coco_names=None):

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
            outputs = clean_model_outputs(model_name, outputs_dict, pytorch_names, custom_names)

            if outputs != []:
                all_qualified_output, all_precision, all_recall, all_F_measure = process_model_outputs(outputs, torch.from_numpy(test_label))
                # since single mode has 1 image input, it could be directly converted to numpy array
                all_qualified_output = np.array(all_qualified_output)
                all_precision = np.array(all_precision)
                all_recall = np.array(all_recall)
                all_F_measure = np.array(all_F_measure)
            else:
                # no qualified output
                all_qualified_output = np.zeros((1, 1, 7))
                all_precision = np.zeros(1)
                all_recall = np.zeros(1)
                all_F_measure = np.zeros(1)

    # multiple image (batch) mode
    elif mode == 'aggregate':
        if not coco_names:
            raise Exception('Needs coco_names for aggregate mode')

        # in this mode test_label needs to be None
        if test_label:
            raise Exception(f'{mode} mode requires NO test_label input as they are loaded via dataloader')

        img_size = 128
        # transforms that are to apply on loaded images
        # since in this mode images are loaded from original COCO, their labels are based on original coco labels, not custom
        AGGREGATE_TRANSFORMS = transforms.Compose([nero_transform.FixedJittering(img_size, x_tran, y_tran),
                                                    nero_transform.ConvertLabel(coco_names, custom_names),
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

        all_qualified_output = np.zeros(len(dataset), dtype=np.ndarray)
        all_precision = np.zeros(len(dataset))
        all_recall = np.zeros(len(dataset))
        all_F_measure = np.zeros(len(dataset))

        for batch_i, (images_paths, test_images, test_labels) in enumerate(dataloader):

            if test_labels is None:
                continue

            with torch.no_grad():
                test_images = test_images.to(device)
                outputs_dict = model(test_images)
                # print('\n Current image path', images_paths)
                # print('\n Ground truth label', test_labels)
                # outputs = nero_utilities.non_max_suppression(output_dict, conf_thres=0, nms_thres=0.4)
                outputs = clean_model_outputs(model_name, outputs_dict, pytorch_names, custom_names)
                if outputs != []:
                    cur_qualified_output, cur_precision, cur_recall, cur_F_measure = process_model_outputs(outputs, test_labels)
                else:
                    # no qualified output
                    cur_qualified_output = []
                    cur_precision = []
                    cur_recall = []
                    cur_F_measure = []

                all_qualified_output[batch_i*batch_size:batch_i*batch_size+len(test_images)] = cur_qualified_output
                all_precision[batch_i*batch_size:batch_i*batch_size+len(test_images)] = cur_precision
                all_recall[batch_i*batch_size:batch_i*batch_size+len(test_images)] = cur_recall
                all_F_measure[batch_i*batch_size:batch_i*batch_size+len(test_images)] = cur_F_measure

    return [all_qualified_output, all_precision, all_recall, all_F_measure]


# run model on either one pair of PIV images or a batch of pairs
def run_piv_once(mode, model_name, model, image_1, image_2, batch_size=1):

    # basic settings for pytorch
    if torch.cuda.is_available():
        # device set up
        device = torch.device('cuda')
        Tensor = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        Tensor = torch.FloatTensor

    if mode == 'single':
        # permute from [index, height, width, dim] to [index, dim, height, width]
        cur_image_pair = torch.cat((image_1, image_2), 2).float().permute(2, 0, 1).unsqueeze(0).to(device)

        if model_name == 'PIV-LiteFlowNet-en':
            # get prediction from loaded model
            prediction = model(cur_image_pair)

            # put on cpu and permute to channel last
            cur_label_pred_pt = prediction.cpu().data
            # change label shape to (height, width, dim)
            cur_label_pred_pt = cur_label_pred_pt.permute(0, 2, 3, 1)
            cur_label_pred_pt = cur_label_pred_pt[0]

        elif model_name == 'Horn-Schunck':
            img_height, img_width = image_1.shape[:2]
            # prepare images for HS
            first_image = image_1.reshape(img_height, img_width) * 1.0/255.0
            second_image = image_2.reshape(img_height, img_width) * 1.0/255.0
            u, v = models.Horn_Schunck(first_image, second_image)
            cur_label_pred_pt = torch.zeros((img_height, img_width, 2))
            cur_label_pred_pt[:, :, 0] = torch.from_numpy(u)
            cur_label_pred_pt[:, :, 1] = torch.from_numpy(v)


        return cur_label_pred_pt


