from __future__ import division

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from PIL import Image
import pandas as pd
from gluoncv import data, utils
import matplotlib.pyplot as plt

import utilities
import model
import data_transform
import prepare_dataset

os.environ['CUDA_VISIBLE_DEVICES']='0'


def single_test_evaluate(model_type, model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, desired_names=None, original_names=None, pytorch_pt_names=None, lite_data=False):
    '''
    model_type: used for distinguishing pre-trained models (pt), which used different target names
    model: loaded PyTorch model
    path: data path
    iou_thres: IOU threshold for defining "acceptable predictions"
    conf_thres: confidence threshold
    nms_thres: NMS threshold for non-maximum suppresion
    img_size: size of input images
    batch_size: PyTorch batch size
    desired_names: custom class names
    original_names: original COCO class names
    pytorch_pt_names: COCO names for pre-trained (pt) model
    lite_data: lite data is the 1D slice of the 2D dataset
    '''

    model.eval()
    # Get dataloader
    EVALUATE_TRANSFORMS = transforms.Compose([data_transform.ConvertLabel(original_names, desired_names),
                                            data_transform.ToTensor()])

    dataset = prepare_dataset.CreateDataset(path, img_size=img_size, transform=EVALUATE_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn
    )


    if lite_data:
        result_size = int(len(dataloader.dataset))
    else:
        result_size = int(np.sqrt(len(dataloader.dataset)))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # save the top 3 results (if available)
    # targets labels and bounding boxes
    num_keep = 3

    # save all the acceptable predictions
    # lite data is 1D
    if lite_data:
        all_target_label = np.zeros(result_size)
        all_target_bb = np.zeros((result_size, 4))

        # prediction results that have been sorted by conf
        # object labels
        all_pred_label_sorted_by_conf = np.zeros((result_size, num_keep))
        # object bounding boxes
        all_pred_bb_sorted_by_conf = np.zeros((result_size, 4*num_keep))
        all_confidence_sorted_by_conf = np.zeros((result_size, num_keep))
        all_iou_sorted_by_conf = np.zeros((result_size, num_keep))

        # results that have been sorted by iou
        # object labels
        all_pred_label_sorted_by_iou = np.zeros((result_size, num_keep))
        # object bounding boxes
        all_pred_bb_sorted_by_iou = np.zeros((result_size, 4*num_keep))
        all_confidence_sorted_by_iou = np.zeros((result_size, num_keep))
        all_iou_sorted_by_iou = np.zeros((result_size, num_keep))

        # results that have been sorted by conf*iou
        # object labels
        all_pred_label_sorted_by_conf_iou = np.zeros((result_size, num_keep))
        # object bounding boxes
        all_pred_bb_sorted_by_conf_iou = np.zeros((result_size, 4*num_keep))
        all_confidence_sorted_by_conf_iou = np.zeros((result_size, num_keep))
        all_iou_sorted_by_conf_iou = np.zeros((result_size, num_keep))

        # results that have been sorted by conf*iou*precision
        # object labels
        all_pred_label_sorted_by_conf_iou_pr = np.zeros((result_size, num_keep))
        # object bounding boxes
        all_pred_bb_sorted_by_conf_iou_pr = np.zeros((result_size, 4*num_keep))
        all_confidence_sorted_by_conf_iou_pr = np.zeros((result_size, num_keep))
        all_iou_sorted_by_conf_iou_pr = np.zeros((result_size, num_keep))

        # precision result
        all_precision = np.zeros((result_size, result_size))
    # full-size data
    else:
        all_target_label = np.zeros((result_size, result_size))
        all_target_bb = np.zeros((result_size, result_size, 4))

        # prediction results that have been sorted by conf
        # object labels
        all_pred_label_sorted_by_conf = np.zeros((result_size, result_size, num_keep))
        # object bounding boxes
        all_pred_bb_sorted_by_conf = np.zeros((result_size, result_size, 4*num_keep))
        all_confidence_sorted_by_conf = np.zeros((result_size, result_size, num_keep))
        all_iou_sorted_by_conf = np.zeros((result_size, result_size, num_keep))

        # results that have been sorted by iou
        # object labels
        all_pred_label_sorted_by_iou = np.zeros((result_size, result_size, num_keep))
        # object bounding boxes
        all_pred_bb_sorted_by_iou = np.zeros((result_size, result_size, 4*num_keep))
        all_confidence_sorted_by_iou = np.zeros((result_size, result_size, num_keep))
        all_iou_sorted_by_iou = np.zeros((result_size, result_size, num_keep))

        # results that have been sorted by conf*iou
        # object labels
        all_pred_label_sorted_by_conf_iou = np.zeros((result_size, result_size, num_keep))
        # object bounding boxes
        all_pred_bb_sorted_by_conf_iou = np.zeros((result_size, result_size, 4*num_keep))
        all_confidence_sorted_by_conf_iou = np.zeros((result_size, result_size, num_keep))
        all_iou_sorted_by_conf_iou = np.zeros((result_size, result_size, num_keep))

        # results that have been sorted by conf*iou*precision
        # object labels
        all_pred_label_sorted_by_conf_iou_pr = np.zeros((result_size, result_size, num_keep))
        # object bounding boxes
        all_pred_bb_sorted_by_conf_iou_pr = np.zeros((result_size, result_size, 4*num_keep))
        all_confidence_sorted_by_conf_iou_pr = np.zeros((result_size, result_size, num_keep))
        all_iou_sorted_by_conf_iou_pr = np.zeros((result_size, result_size, num_keep))

        # precision result
        all_precision = np.zeros((result_size, result_size))
        all_recall = np.zeros((result_size, result_size))
        all_F_measure = np.zeros((result_size, result_size))


    # all the confidence - used for quick test in lite data
    if lite_data:
        all_acceptable_conf = []
        all_acceptable_iou = []

    for batch_i, (path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # array index (needed when doing normal evaluation)
        if not lite_data:
            row = batch_i // result_size
            col = batch_i % result_size
            # print(f'row: {row}, col: {col}')

        # visualize the input image
        # channel first to channel last
        # show_img = imgs[0].permute(1, 2, 0).numpy()
        # plt.imshow(show_img)
        # plt.show()

        if targets is None:
            continue

        # when we are using pretrained model
        if model_type == 'pt':
            for i in range(len(targets)):
                targets[i, 1] = pytorch_pt_names.index(desired_names[int(targets[i, 1]-1)]) + 1

        # Extract labels to compute IOU
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs_dict = model(imgs)

        # since for single-test, batch size is kept at 1
        outputs = []
        for i in range(len(outputs_dict)):
            image_pred = outputs_dict[i]
            pred_boxes = image_pred['boxes'].cpu()
            pred_labels = image_pred['labels'].cpu()
            pred_confs = image_pred['scores'].cpu()

            # transform output in the format of (x1, y1, x2, y2, conf, class_pred)
            output = torch.zeros((len(pred_boxes), 6))
            output[:, :4] = pred_boxes
            output[:, 4] = pred_confs
            output[:, 5] = pred_labels

            outputs.append(output)

        empty_output = False
        if outputs != []:
            # print(len(outputs[0]))
            # get the predicted bounding boxes and its iou values of each object
            cur_pred_label_sorted_by_conf, cur_pred_bb_sorted_by_conf, cur_confidence_sorted_by_conf, cur_iou_sorted_by_conf, \
            cur_pred_label_sorted_by_iou, cur_pred_bb_sorted_by_iou, cur_confidence_sorted_by_iou, cur_iou_sorted_by_iou, \
            cur_pred_label_sorted_by_conf_iou, cur_pred_bb_sorted_by_conf_iou, cur_confidence_sorted_by_conf_iou, cur_iou_sorted_by_conf_iou, \
            cur_pred_label_sorted_by_conf_iou_pr, cur_pred_bb_sorted_by_conf_iou_pr, cur_confidence_sorted_by_conf_iou_pr, cur_iou_sorted_by_conf_iou_pr, \
                cur_precision, cur_recall, cur_F_measure = utilities.process_model_outputs(outputs, targets, iou_thres=iou_thres, conf_thres=conf_thres)
            # print(cur_confidence_sorted_by_iou)
            if len(cur_pred_label_sorted_by_conf) > 1:
                raise Exception('Multi-object case has not been considered')

            # from gluoncv import data, utils
            # from matplotlib import pyplot as plt
            # class_names = load_classes(data_config["desired_names"])
            # names = ['single-test_-10_-16.png', 'single-test_-11_-17.png']
            # img_path = f'/home/zhuokai/Desktop/nvme1n1p1/Data/processed_coco/small_dataset/shift_equivariance/single_test_temp/cup/images/{names[batch_i]}'
            # orig_img = np.asarray(Image.open(img_path))
            # utils.viz.plot_bbox(orig_img, np.array([cur_pred_bb_sorted_by_conf[0][0]]), scores=None,
            #                         labels=np.array([cur_pred_label_sorted_by_conf[0][0]]), class_names=class_names)
            # plt.show()
            # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/detection/logs_zz/shift_equivariance/pattern_investigation/result_{batch_i}.png')

            # when the output is not empty
            if cur_pred_label_sorted_by_conf != []:
                # target labels and bounding boxes
                if lite_data:
                    all_target_label[batch_i] = targets.numpy()[0, 1]
                    all_target_bb[batch_i, :] = targets.numpy()[0, 2:]
                else:
                    all_target_label[row, col] = targets.numpy()[0, 1]
                    all_target_bb[row, col, :] = targets.numpy()[0, 2:]


                # for results sorted by conf
                # extract label information from the highest-confidence result
                # if len(cur_pred_label_sorted_by_conf[0]) < 3:
                #     print('ahhhhh less than 3')
                for i in range(min(len(cur_pred_label_sorted_by_conf[0]), num_keep)):

                    if lite_data:
                        # for results sorted by conf
                        all_pred_label_sorted_by_conf[batch_i, i] = cur_pred_label_sorted_by_conf[0][i]
                        # extract bb information from the highest-confidence result
                        all_pred_bb_sorted_by_conf[batch_i, i*4:(i+1)*4] = cur_pred_bb_sorted_by_conf[0][i, :]
                        # extract confidence from the highest-confidence result
                        all_confidence_sorted_by_conf[batch_i, i] = cur_confidence_sorted_by_conf[0][i]
                        # extract iou from the highest-confidence result
                        all_iou_sorted_by_conf[batch_i, i] = cur_iou_sorted_by_conf[0][i]

                        # for results sorted by iou
                        # extract label information from the highest-confidence result
                        all_pred_label_sorted_by_iou[batch_i, i] = cur_pred_label_sorted_by_iou[0][i]
                        # extract bb information from the highest-confidence result
                        all_pred_bb_sorted_by_iou[batch_i, i*4:(i+1)*4] = cur_pred_bb_sorted_by_iou[0][i, :]
                        # extract confidence from the highest-confidence result
                        all_confidence_sorted_by_iou[batch_i, i] = cur_confidence_sorted_by_iou[0][i]
                        # extract iou from the highest-confidence result
                        all_iou_sorted_by_iou[batch_i, i] = cur_iou_sorted_by_iou[0][i]

                        # for results sorted by conf*iou
                        # extract label information from the highest-confidence result
                        all_pred_label_sorted_by_conf_iou[batch_i, i] = cur_pred_label_sorted_by_conf_iou[0][i]
                        # extract bb information from the highest-confidence result
                        all_pred_bb_sorted_by_conf_iou[batch_i, i*4:(i+1)*4] = cur_pred_bb_sorted_by_conf_iou[0][i, :]
                        # extract confidence from the highest-confidence result
                        all_confidence_sorted_by_conf_iou[batch_i, i] = cur_confidence_sorted_by_conf_iou[0][i]
                        # extract iou from the highest-confidence result
                        all_iou_sorted_by_conf_iou[batch_i, i] = cur_iou_sorted_by_conf_iou[0][i]

                        # for results sorted by conf*iou*precision
                        # extract label information from the highest-confidence result
                        all_pred_label_sorted_by_conf_iou_pr[batch_i, i] = cur_pred_label_sorted_by_conf_iou_pr[0][i]
                        # extract bb information from the highest-confidence result
                        all_pred_bb_sorted_by_conf_iou_pr[batch_i, i*4:(i+1)*4] = cur_pred_bb_sorted_by_conf_iou_pr[0][i, :]
                        # extract confidence from the highest-confidence result
                        all_confidence_sorted_by_conf_iou_pr[batch_i, i] = cur_confidence_sorted_by_conf_iou_pr[0][i]
                        # extract iou from the highest-confidence result
                        all_iou_sorted_by_conf_iou_pr[batch_i, i] = cur_iou_sorted_by_conf_iou_pr[0][i]

                        # precision, recall and F-measure
                        all_precision[batch_i] = cur_precision[0]
                        all_recall[batch_i] = cur_recall[0]
                        all_F_measure[batch_i] = cur_F_measure[0]

                    else:
                        # for results sorted by conf
                        all_pred_label_sorted_by_conf[row, col, i] = cur_pred_label_sorted_by_conf[0][i]
                        # extract bb information from the highest-confidence result
                        all_pred_bb_sorted_by_conf[row, col, i*4:(i+1)*4] = cur_pred_bb_sorted_by_conf[0][i, :]
                        # extract confidence from the highest-confidence result
                        all_confidence_sorted_by_conf[row, col, i] = cur_confidence_sorted_by_conf[0][i]
                        # extract iou from the highest-confidence result
                        all_iou_sorted_by_conf[row, col, i] = cur_iou_sorted_by_conf[0][i]

                        # for results sorted by iou
                        # extract label information from the highest-confidence result
                        all_pred_label_sorted_by_iou[row, col, i] = cur_pred_label_sorted_by_iou[0][i]
                        # extract bb information from the highest-confidence result
                        all_pred_bb_sorted_by_iou[row, col, i*4:(i+1)*4] = cur_pred_bb_sorted_by_iou[0][i, :]
                        # extract confidence from the highest-confidence result
                        all_confidence_sorted_by_iou[row, col, i] = cur_confidence_sorted_by_iou[0][i]
                        # extract iou from the highest-confidence result
                        all_iou_sorted_by_iou[row, col, i] = cur_iou_sorted_by_iou[0][i]

                        # for results sorted by conf*iou
                        # extract label information from the highest-confidence result
                        all_pred_label_sorted_by_conf_iou[row, col, i] = cur_pred_label_sorted_by_conf_iou[0][i]
                        # extract bb information from the highest-confidence result
                        all_pred_bb_sorted_by_conf_iou[row, col, i*4:(i+1)*4] = cur_pred_bb_sorted_by_conf_iou[0][i, :]
                        # extract confidence from the highest-confidence result
                        all_confidence_sorted_by_conf_iou[row, col, i] = cur_confidence_sorted_by_conf_iou[0][i]
                        # extract iou from the highest-confidence result
                        all_iou_sorted_by_conf_iou[row, col, i] = cur_iou_sorted_by_conf_iou[0][i]

                        # for results sorted by conf*iou*precision
                        # extract label information from the highest-confidence result
                        all_pred_label_sorted_by_conf_iou_pr[row, col, i] = cur_pred_label_sorted_by_conf_iou_pr[0][i]
                        # extract bb information from the highest-confidence result
                        all_pred_bb_sorted_by_conf_iou_pr[row, col, i*4:(i+1)*4] = cur_pred_bb_sorted_by_conf_iou_pr[0][i, :]
                        # extract confidence from the highest-confidence result
                        all_confidence_sorted_by_conf_iou_pr[row, col, i] = cur_confidence_sorted_by_conf_iou_pr[0][i]
                        # extract iou from the highest-confidence result
                        all_iou_sorted_by_conf_iou_pr[row, col, i] = cur_iou_sorted_by_conf_iou_pr[0][i]

                        # precision
                        all_precision[row, col] = cur_precision[0]
                        all_recall[row, col] = cur_recall[0]
                        all_F_measure[row, col] = cur_F_measure[0]
            else:
                empty_output = True
        else:
            empty_output = True

        # if no model outputs
        if empty_output:
            # target labels
            if lite_data:
                all_target_label[batch_i] = targets.numpy()[0, 1]
                all_target_bb[batch_i, :] = targets.numpy()[0, 2:]

                # fill for the prediction values
                # sorted by conf
                # label-related
                all_pred_label_sorted_by_conf[batch_i, :] = -1
                # bounding box related
                all_pred_bb_sorted_by_conf[batch_i, :] = -1
                # prediction confidence
                all_confidence_sorted_by_conf[batch_i, :] = 0
                # iou and custom quality
                all_iou_sorted_by_conf[batch_i, :] = 0

                # sorted by iou
                # label-related
                all_pred_label_sorted_by_iou[batch_i, :] = -1
                # bounding box related
                all_pred_bb_sorted_by_iou[batch_i, :] = -1
                # prediction confidence
                all_confidence_sorted_by_iou[batch_i, :] = 0
                # iou and custom quality
                all_iou_sorted_by_iou[batch_i, :] = 0

                # sorted by conf*iou
                # label-related
                all_pred_label_sorted_by_conf_iou[batch_i, :] = -1
                # bounding box related
                all_pred_bb_sorted_by_conf_iou[batch_i, :] = -1
                # prediction confidence
                all_confidence_sorted_by_conf_iou[batch_i, :] = 0
                # iou and custom quality
                all_iou_sorted_by_conf_iou[batch_i, :] = 0

                # sorted by conf*iou*pr
                # label-related
                all_pred_label_sorted_by_conf_iou_pr[batch_i, :] = -1
                # bounding box related
                all_pred_bb_sorted_by_conf_iou_pr[batch_i, :] = -1
                # prediction confidence
                all_confidence_sorted_by_conf_iou_pr[batch_i, :] = 0
                # iou and custom quality
                all_iou_sorted_by_conf_iou_pr[batch_i, :] = 0

                all_precision[batch_i] = 0

            else:
                all_target_label[row, col] = targets.numpy()[0, 1]
                all_target_bb[row, col, :] = targets.numpy()[0, 2:]

                # fill for the prediction values
                # sorted by conf
                # label-related
                all_pred_label_sorted_by_conf[row, col, :] = -1
                # bounding box related
                all_pred_bb_sorted_by_conf[row, col, :] = -1
                # prediction confidence
                all_confidence_sorted_by_conf[row, col, :] = 0
                # iou and custom quality
                all_iou_sorted_by_conf[row, col, :] = 0

                # sorted by iou
                # label-related
                all_pred_label_sorted_by_iou[row, col, :] = -1
                # bounding box related
                all_pred_bb_sorted_by_iou[row, col, :] = -1
                # prediction confidence
                all_confidence_sorted_by_iou[row, col, :] = 0
                # iou and custom quality
                all_iou_sorted_by_iou[row, col, :] = 0

                # sorted by conf*iou
                # label-related
                all_pred_label_sorted_by_conf_iou[row, col, :] = -1
                # bounding box related
                all_pred_bb_sorted_by_conf_iou[row, col, :] = -1
                # prediction confidence
                all_confidence_sorted_by_conf_iou[row, col, :] = 0
                # iou and custom quality
                all_iou_sorted_by_conf_iou[row, col, :] = 0

                # sorted by conf*iou*pr
                # label-related
                all_pred_label_sorted_by_conf_iou_pr[row, col, :] = -1
                # bounding box related
                all_pred_bb_sorted_by_conf_iou_pr[row, col, :] = -1
                # prediction confidence
                all_confidence_sorted_by_conf_iou_pr[row, col, :] = 0
                # iou and custom quality
                all_iou_sorted_by_conf_iou_pr[row, col, :] = 0

                all_precision[row, col] = 0

    return [all_target_label, all_target_bb,
            all_pred_label_sorted_by_conf, all_pred_bb_sorted_by_conf, all_confidence_sorted_by_conf, all_iou_sorted_by_conf,
            all_pred_label_sorted_by_iou, all_pred_bb_sorted_by_iou, all_confidence_sorted_by_iou, all_iou_sorted_by_iou,
            all_pred_label_sorted_by_conf_iou, all_pred_bb_sorted_by_conf_iou, all_confidence_sorted_by_conf_iou, all_iou_sorted_by_conf_iou,
            all_pred_label_sorted_by_conf_iou_pr, all_pred_bb_sorted_by_conf_iou_pr, all_confidence_sorted_by_conf_iou_pr, all_iou_sorted_by_conf_iou_pr,
            all_precision, all_recall, all_F_measure]


# when used in validation, percentage, desired_names and original_names are needed for building dataloader
# when used in test, those three are not needed
def evaluate(mode,
             purpose,
             model_type,
             model,
             path,
             iou_thres,
             conf_thres,
             nms_thres,
             img_size,
             batch_size,
             desired_names=None,
             original_names=None,
             pytorch_pt_names=None,
             rot_percentage=None,
             shift_percentage=None,
             rot=None,
             x_tran=None,
             y_tran=None):

    model.eval()
    # Get dataloader
    if mode == 'val':
        if purpose == 'shift-equivariance':
            EVALUATE_TRANSFORMS = transforms.Compose([data_transform.RandomRotationShift(img_size=img_size,
                                                                                         fixed_rot=False,
                                                                                         fixed_shift=False,
                                                                                         rot_percentage=100,
                                                                                         shift_percentage=shift_percentage),
                                                        data_transform.ShiftEqvAug(),
                                                        data_transform.ConvertLabel(original_names, desired_names),
                                                        data_transform.ToTensor()])
        elif purpose == 'rotation-equivariance':
            EVALUATE_TRANSFORMS = transforms.Compose([data_transform.RandomRotationShift(img_size=img_size,
                                                                                         fixed_rot=False,
                                                                                         fixed_shift=False,
                                                                                         rot_percentage=rot_percentage,
                                                                                         shift_percentage=100),
                                                        data_transform.RotEqvAug(),
                                                        data_transform.ConvertLabel(original_names, desired_names),
                                                        data_transform.ToTensor()])
    elif mode == 'test':
        if purpose == 'shift-equivariance':
            EVALUATE_TRANSFORMS = transforms.Compose([data_transform.FixedJittering(img_size, x_tran, y_tran),
                                                        data_transform.ShiftEqvAug(),
                                                        data_transform.ConvertLabel(original_names, desired_names),
                                                        data_transform.ToTensor()])
        elif purpose == 'rotation-equivariance':
            EVALUATE_TRANSFORMS = transforms.Compose([data_transform.RandomRotationShift(img_size=img_size,
                                                                                         fixed_rot=True,
                                                                                         fixed_shift=True,
                                                                                         rot=rot,
                                                                                         x_tran=0,
                                                                                         y_tran=0),
                                                        data_transform.ConvertLabel(original_names, desired_names),
                                                        data_transform.ToTensor()])

    dataset = prepare_dataset.CreateDataset(list_path=path,
                                            img_size=img_size,
                                            transform=EVALUATE_TRANSFORMS)

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
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        if targets is None:
            continue

        # visualize input data with ground truth for debugging purposes
        # utils.viz.plot_bbox(imgs[0].permute(1, 2, 0).numpy()*255, targets[:, 2:].numpy(), scores=None,
        #                     labels=targets[:, 1].numpy()-1, class_names=desired_names)
        # plt.show()

        if model_type == 'pt':
            # print(f'Before conversion target = {targets}')
            for i in range(len(targets)):
                targets[i, 1] = pytorch_pt_names.index(desired_names[int(targets[i, 1])-1]) + 1
            # print(f'After conversion target = {targets}')

        # Extract class labels from targets
        labels += targets[:, 1].tolist()

        # put on GPU
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        if mode == 'val':
            targets = Variable(targets.type(Tensor), requires_grad=False)

            # put targets in dictionary format that faster-rcnn takes
            targets_dict = []
            for i in range(len(targets)):
                target = {}
                # target box has to be in shape (N, 4)
                target['boxes'] = targets[i:i+1, 2:]
                target['labels'] = targets[i:i+1, 1].type(torch.int64)
                targets_dict.append(target)

            with torch.no_grad():
                model.train()
                loss_dict = model(imgs, targets_dict)
                # print(loss_dict)
                losses = sum(loss for loss in loss_dict.values())
                all_losses.append(losses.item())
                model.eval()
                output_dict = model(imgs)
                # print(f"Avg val confidence: {torch.mean(output_dict[0]['scores'])}")

            outputs = utilities.non_max_suppression(output_dict, conf_thres=conf_thres, nms_thres=nms_thres)
            sample_metrics += utilities.get_batch_statistics(outputs, targets.cpu(), iou_threshold=iou_thres)

        elif mode == 'test':
            with torch.no_grad():
                output_dict = model(imgs)

            outputs = utilities.non_max_suppression(output_dict, conf_thres=conf_thres, nms_thres=nms_thres)
            sample_metrics += utilities.get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        else:
            raise Exception('Unsupported mode in evaluate')

        # visualize detections for debugging purposes
        # utils.viz.plot_bbox(imgs[0].cpu().permute(1, 2, 0).numpy()*255, np.array([outputs[0][:, :4].numpy()]), scores=None,
        #                     labels=outputs[0][:, 5:6].numpy(), class_names=desired_names)
        # print(outputs)
        # utils.viz.plot_bbox(imgs[0].cpu().permute(1, 2, 0).numpy()*255, outputs[0][:, :4].numpy(), scores=None,
        #                     labels=outputs[0][:, 5:6].numpy(), class_names=desired_names)
        # plt.show()

    if len(sample_metrics) == 0:  # no detections over whole data set.
        # return None
        if mode == 'val':
            empty_result = np.empty(len(desired_names))
            empty_result[:] = -1
            # return all_losses, [np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0], dtype=np.int)]
            return all_losses, [empty_result, empty_result, empty_result, empty_result, np.array(list(range(1, len(desired_names)+1)), dtype=np.int)]
        elif mode == 'test':
            # return np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0], dtype=np.int)
            empty_result = np.empty(len(desired_names))
            empty_result[:] = -1
            return empty_result, empty_result, empty_result, empty_result, np.array(list(range(1, len(desired_names)+1)), dtype=np.int), empty_result, empty_result

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # generate per-class results
    precision, recall, AP, f1, ap_class, max_conf, avg_conf = utilities.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    if mode == 'val':
        return all_losses, [precision, recall, AP, f1, ap_class]
    elif mode == 'test':
        return precision, recall, AP, f1, ap_class, max_conf, avg_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # shift-equivariance, rotation-equivariance or scale equivariance
    parser.add_argument('-p', '--purpose', type=str, dest='purpose')
    parser.add_argument('-m', '--mode', type=str, help='test, single-test, or single-test-all-candidates mode')
    parser.add_argument("--data_name", type=str, help="Defines the name of the dataset (coco, mnist, or coco-lite, mnist-lite)")
    parser.add_argument("--batch_size", type=int, default=100, help="size of each image batch")
    parser.add_argument("--data_config", type=str, help="path to data config file")
    # percentage is only for naming use
    parser.add_argument("--percentage", type=int, help="percentage of jittering or scailing")
    # model type (normal, si, or pt)
    parser.add_argument('--model_type', action='store', type=str)
    # pt model does not need a model path
    parser.add_argument("--model_path", type=str, help="path to load model")
    # nms, n_cpu, img_size could just use the defaults
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    # output result dir
    parser.add_argument('-o', '--output_dir', type=str)
    # previous result path (for continuly inferencing)
    parser.add_argument('--prev_result_path', type=str)

    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the testing more verbose")
    opt = parser.parse_args()

    # read parameters
    purpose = opt.purpose
    mode = opt.mode
    data_name = opt.data_name
    if opt.model_type != None:
        model_type = opt.model_type
    else:
        model_type = 'normal'
    if model_type != 'pt':
        percentage = opt.percentage
    else:
        percentage = 'pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    data_config = prepare_dataset.parse_data_config(opt.data_config)
    num_classes = int(data_config['classes'])
    if mode == 'val':
        valid_path = data_config['valid']
    elif mode == 'test' or mode == 'single-test':
        test_path = data_config['test']
    elif mode == 'single-test-all-candidates':
        test_dir = data_config['test']

    # make model and log directory
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # different dataset configuration files are different
    if data_name == 'mnist':
        # original mnist names and desired mnist names are the same
        original_names = prepare_dataset.load_classes(data_config["desired_names"])
        desired_names = prepare_dataset.load_classes(data_config["desired_names"])
    elif data_name == 'coco':
        original_names = prepare_dataset.load_classes(data_config["original_names"])
        desired_names = prepare_dataset.load_classes(data_config["desired_names"])
        # pretrained pytorch model has empty classes
        pytorch_pt_names = prepare_dataset.load_classes(data_config['pytorch_pt_names'])

    # if we are dealing with lite dataset
    if 'lite' in data_name:
        lite_data = True
    else:
        lite_data = False

    # print out inputs info
    print(f'\nPurpose of testing: {purpose}')
    print(f'Test mode: {mode}')
    print(f'Dataset name: {data_name}')
    print(f'Testing model type: {model_type}\n')
    if model_type != 'pt':
        print(f'Model input path: {opt.model_path}')
    else:
        print(f'Model input path: pretrained model')
    print(f'Output dir: {output_dir}')
    if mode == 'val':
        print(f'Val data path: {valid_path}\n')
    elif mode == 'test' or mode == 'single-test':
        print(f'Test data path: {test_path}\n')
    elif mode == 'single-test-all-candidates':
        print(f'Test data dir: {test_dir}\n')

    # Initiate model
    backbone_name = 'resnet50'
    if model_type == 'pt':
        if 'resnet' in backbone_name:
            # model = model.faster_rcnn(num_classes, img_size=opt.img_size, backbone_name=backbone_name)
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                        min_size=opt.img_size)
            print(f'Pretrained model loaded successfully')
    else:
        if 'resnet' in backbone_name:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                        num_classes=num_classes+1,
                                                                        pretrained_backbone=True,
                                                                        min_size=opt.img_size)
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

        # load the weight
        model.load_state_dict(torch.load(opt.model_path))

    model.to(device)
    print(f'\nTrained model loaded successfully')

    # all the test path
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    mode = opt.mode

    if opt.prev_result_path != None:
        prev_result_path = opt.prev_result_path[0]
    else:
        prev_result_path = None

    # make sure the confidence threshold is low
    conf_thres = 1e-4
    # conf_thres = 0.1
    # iou threshold is to define acceptable predictions
    iou_thres = 0.5
    nms_thres = 0.5

    if purpose == 'shift-equivariance':

        if mode == 'test':
            # for test (sparse sampling between -64 to 63)
            positive_trans = [1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 29, 37, 48, 63]
            negative_trans = list(reversed(positive_trans))
            negative_trans = [-x for x in negative_trans]
            x_translation = negative_trans + [0] + positive_trans
            y_translation = negative_trans + [0] + positive_trans


        elif mode == 'dense-test':
            # for test_2 (dense sampling between -16 to 16)
            x_translation = list(range(-10, 11))
            y_translation = list(range(-10, 11))

        elif mode == 'single-test':
            x_translation = list(range(-opt.img_size//2, opt.img_size//2))
            y_translation = list(range(-opt.img_size//2, opt.img_size//2))
            # enforce the batch size to be 1
            opt.batch_size = 1

        elif mode == 'single-test-all-candidates':
            # for test (sparse sampling between -64 to 63)
            positive_trans = [1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 29, 37, 48, 63]
            negative_trans = list(reversed(positive_trans))
            negative_trans = [-x for x in negative_trans]
            x_translation = negative_trans + [0] + positive_trans
            y_translation = negative_trans + [0] + positive_trans
            # enforce the batch size to be 1
            opt.batch_size = 1

            # list that contains all the image and label indices
            all_candidates_path = data_config['candidates_list']
            all_image_indices = np.load(all_candidates_path)['image_index']
            all_label_indices = np.load(all_candidates_path)['key_label_index']

        if mode != 'val':
            if prev_result_path != None:
                all_mean_precision = np.load(prev_result_path)['all_mean_precision']
                all_mean_recall = np.load(prev_result_path)['all_mean_recall']
                all_mean_AP = np.load(prev_result_path)['all_mean_AP']
                all_mean_f1 = np.load(prev_result_path)['all_mean_f1']
                all_max_conf = np.load(prev_result_path)['all_max_conf']
                all_mean_conf = np.load(prev_result_path)['all_mean_conf']
                print(f'Previously evaluated result has been loaded from {prev_result_path}')
            else:
                # all the class-based results
                all_mean_precision = np.zeros((len(y_translation), len(x_translation), 5))
                all_mean_recall = np.zeros((len(y_translation), len(x_translation), 5))
                all_mean_AP = np.zeros((len(y_translation), len(x_translation), 5))
                all_mean_f1 = np.zeros((len(y_translation), len(x_translation), 5))
                all_max_conf = np.zeros((len(y_translation), len(x_translation), 5))
                all_mean_conf = np.zeros((len(y_translation), len(x_translation), 5))

        # if we are testing with the validation dataset
        if mode == 'val':
            loss, metrics_output = evaluate(
                mode='val',
                purpose=purpose,
                model_type=model_type,
                model=model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.4,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
                desired_names=desired_names,
                original_names=original_names,
                percentage=percentage
            )

        # test or dense test
        elif mode == 'test' or mode == 'dense-test':

            # define path for saving testing results
            if model_type == 'pt':
                loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_{mode}_{model_type}.npz')
            else:
                loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_{mode}_{percentage}-jittered_{model_type}.npz')

            for y, y_trans in enumerate(y_translation):
                for x, x_trans in enumerate(x_translation):

                    # for continusly training purpose
                    # if all_mean_AP[y, x, 0] != 0.0:
                    #     print(f'Skipping evaluation at y_tran={y_trans}, x_tran={x_trans}')
                    #     continue

                    print(f'Evaluating y_tran={y_trans}, x_tran={x_trans}')
                    # test_path = os.path.join(test_dir, f'test_x_{x_trans}_y_{y_trans}', 'test.txt')

                    # class_names = load_classes(test_dir, f'test_x_{x_trans}_y_{y_trans}', data_config["names"])
                    precision, recall, AP, f1, ap_class, max_conf, mean_conf = evaluate(
                        mode='test',
                        purpose=purpose,
                        model_type=model_type,
                        model=model,
                        path=test_path,
                        iou_thres=iou_thres,
                        conf_thres=conf_thres,
                        nms_thres=nms_thres,
                        img_size=opt.img_size,
                        batch_size=opt.batch_size,
                        desired_names=desired_names,
                        original_names=original_names,
                        pytorch_pt_names=pytorch_pt_names,
                        x_tran=x_trans,
                        y_tran=y_trans
                    )

                    # save the results in per-class fashion
                    all_mean_precision[y, x, :] = precision
                    all_mean_recall[y, x, :] = recall
                    all_mean_AP[y, x, :] = AP
                    all_mean_f1[y, x, :] = f1
                    all_max_conf[y, x, :] = max_conf
                    all_mean_conf[y, x, :] = mean_conf
                    print(f"mAP for each class at translation [y_trans, x_trans] = [{y_trans}, {x_trans}]: {AP}")

            np.savez(loss_path,
                    all_mean_precision=all_mean_precision,
                    all_mean_recall=all_mean_recall,
                    all_mean_AP=all_mean_AP,
                    all_mean_f1=all_mean_f1,
                    all_max_conf=all_max_conf,
                    all_mean_conf=all_mean_conf)

            print(f'\nTesting result has been saved to {loss_path}')

        # single test
        elif mode == 'single-test':
            # [all_target_label, all_target_bb,
            # all_pred_label_sorted_by_conf, all_pred_bb_sorted_by_conf, all_confidence_sorted_by_conf, all_iou_sorted_by_conf,
            # all_pred_label_sorted_by_iou, all_pred_bb_sorted_by_iou, all_confidence_sorted_by_iou, all_iou_sorted_by_iou,
            # all_pred_label_sorted_by_conf_iou, all_pred_bb_sorted_by_conf_iou, all_confidence_sorted_by_conf_iou, all_iou_sorted_by_conf_iou,
            # all_pred_label_sorted_by_conf_iou_pr, all_pred_bb_sorted_by_conf_iou_pr, all_confidence_sorted_by_conf_iou_pr, all_iou_sorted_by_conf_iou_pr,
            # all_precision, all_recall, all_F_measure]
            results_list = single_test_evaluate(model_type,
                                                model,
                                                path=test_path,
                                                iou_thres=iou_thres,
                                                conf_thres=conf_thres,
                                                nms_thres=nms_thres,
                                                img_size=opt.img_size,
                                                batch_size=opt.batch_size,
                                                desired_names=desired_names,
                                                original_names=original_names,
                                                pytorch_pt_names=pytorch_pt_names,
                                                lite_data=lite_data)

            # save all the results as npz
            if model_type == 'pt':
                loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_{mode}_{model_type}_{iou_thres}.npz')
            else:
                loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_{mode}_{percentage}-jittered_{model_type}_{iou_thres}.npz')

            np.savez(loss_path,
                    all_target_label=results_list[0],
                    all_target_bb=results_list[1],
                    # results sorted by conf
                    all_pred_label_sorted_by_conf=results_list[2],
                    all_pred_bb_sorted_by_conf=results_list[3],
                    all_confidence_sorted_by_conf=results_list[4],
                    all_iou_sorted_by_conf=results_list[5],
                    # results sorted by iou
                    all_pred_label_sorted_by_iou=results_list[6],
                    all_pred_bb_sorted_by_iou=results_list[7],
                    all_confidence_sorted_by_iou=results_list[8],
                    all_iou_sorted_by_iou=results_list[9],
                    # results sorted by conf*iou
                    all_pred_label_sorted_by_conf_iou=results_list[10],
                    all_pred_bb_sorted_by_conf_iou=results_list[11],
                    all_confidence_sorted_by_conf_iou=results_list[12],
                    all_iou_sorted_by_conf_iou=results_list[13],
                    # results sorted by conf*iou*pr
                    all_pred_label_sorted_by_conf_iou_pr=results_list[14],
                    all_pred_bb_sorted_by_conf_iou_pr=results_list[15],
                    all_confidence_sorted_by_conf_iou_pr=results_list[16],
                    all_iou_sorted_by_conf_iou_pr=results_list[17],
                    # precision, recall and F_measure
                    all_precision=results_list[18],
                    all_recall=results_list[19],
                    all_F_measure=results_list[20])

            print(f'\nTest result has been save to npz file at {loss_path}')


        # single test all candiates
        elif mode == 'single-test-all-candidates':

            # for each folder in test dir
            for i in range(len(all_image_indices)):
                image_index = all_image_indices[i]
                label_index = all_label_indices[i]
                test_path = os.path.join(test_dir, f'image_{image_index}_{label_index}', 'single-test.txt')

                results_list = single_test_evaluate(model_type,
                                                    model,
                                                    path=test_path,
                                                    iou_thres=iou_thres,
                                                    conf_thres=conf_thres,
                                                    nms_thres=nms_thres,
                                                    img_size=opt.img_size,
                                                    batch_size=opt.batch_size,
                                                    desired_names=desired_names,
                                                    original_names=original_names,
                                                    pytorch_pt_names=pytorch_pt_names,
                                                    lite_data=lite_data)

                # save all the results as npz
                if model_type == 'pt':
                    loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_{mode}_{model_type}_{image_index}_{label_index}.npz')
                else:
                    loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_{mode}_{percentage}-jittered_{model_type}_{image_index}_{label_index}.npz')

                np.savez(loss_path,
                        all_target_label=results_list[0],
                        all_target_bb=results_list[1],
                        # results sorted by conf
                        all_pred_label_sorted_by_conf=results_list[2],
                        all_pred_bb_sorted_by_conf=results_list[3],
                        all_confidence_sorted_by_conf=results_list[4],
                        all_iou_sorted_by_conf=results_list[5],
                        # results sorted by iou
                        all_pred_label_sorted_by_iou=results_list[6],
                        all_pred_bb_sorted_by_iou=results_list[7],
                        all_confidence_sorted_by_iou=results_list[8],
                        all_iou_sorted_by_iou=results_list[9],
                        # results sorted by conf*iou
                        all_pred_label_sorted_by_conf_iou=results_list[10],
                        all_pred_bb_sorted_by_conf_iou=results_list[11],
                        all_confidence_sorted_by_conf_iou=results_list[12],
                        all_iou_sorted_by_conf_iou=results_list[13],
                        # results sorted by conf*iou*pr
                        all_pred_label_sorted_by_conf_iou_pr=results_list[14],
                        all_pred_bb_sorted_by_conf_iou_pr=results_list[15],
                        all_confidence_sorted_by_conf_iou_pr=results_list[16],
                        all_iou_sorted_by_conf_iou_pr=results_list[17],
                        # precision
                        all_precision=results_list[18],
                        all_recall=results_list[19],
                        all_F_measure=results_list[20])

                print(f'\nTest result has been save to npz file at {loss_path}')

    # rotation equivariance
    elif purpose == 'rotation-equivariance':

        if model_type == 'pt':
            loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_{mode}_{model_type}.npz')
        else:
            loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_{percentage}-rotated_{mode}_{model_type}.npz')

        # all the rotations
        # all_rotations = list(range(0, 360, 5))
        all_rotations = list(range(90, 90, 1))

        if prev_result_path != None:
            all_single_class_precision = np.load(prev_result_path)['all_single_class_precision']
            all_single_class_recall = np.load(prev_result_path)['all_single_class_recall']
            all_single_class_AP = np.load(prev_result_path)['all_single_class_AP']
            all_single_class_f1 = np.load(prev_result_path)['all_single_class_f1']
            print(f'Previously evaluated result has been loaded from {prev_result_path}')
        else:
            # initialize all the results (5 classes)
            all_single_class_precision = np.zeros((len(all_rotations), 5))
            all_single_class_recall = np.zeros((len(all_rotations), 5))
            all_single_class_AP = np.zeros((len(all_rotations), 5))
            all_single_class_f1 = np.zeros((len(all_rotations), 5))

        for i, rot in enumerate(all_rotations):
            # for continusly testing purpose
            if all_single_class_AP[i, 0] != 0:
                print(f'Skipping evaluation at scale={scale}')
                continue

            print(f'Evaluating rot={rot}')

            precision, recall, AP, f1, ap_class, max_conf, mean_conf = evaluate(
                mode='test',
                purpose=purpose,
                model_type=model_type,
                model=model,
                path=test_path,
                iou_thres=iou_thres,
                conf_thres=conf_thres,
                nms_thres=nms_thres,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
                desired_names=desired_names,
                original_names=original_names,
                pytorch_pt_names=pytorch_pt_names,
                rot=rot
            )

            # save the results
            all_single_class_precision[i] = precision
            all_single_class_recall[i] = recall
            all_single_class_AP[i] = AP
            all_single_class_f1[i] = f1
            print(f"APs at rotation = {rot}: {AP}")

        np.savez(loss_path,
                all_single_class_precision=all_single_class_precision,
                all_single_class_recall=all_single_class_recall,
                all_single_class_AP=all_single_class_AP,
                all_single_class_f1=all_single_class_f1)

        print(f'\nTesting result has been saved to {loss_path}')

    # scale equivariance case
    elif purpose == 'scale-equivariance':
        loss_path = os.path.join(output_dir, f'faster_rcnn_{backbone_name}_test_custom_coco_{mode}_{model_type}.npz')
        if 'single' in opt.data_config:
            raise NotImplementedError('Not yet implemented')
        else:
            all_scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

        if prev_result_path != None:
            all_single_class_precision = np.load(prev_result_path)['all_single_class_precision']
            all_single_class_recall = np.load(prev_result_path)['all_single_class_recall']
            all_single_class_AP = np.load(prev_result_path)['all_single_class_AP']
            all_single_class_f1 = np.load(prev_result_path)['all_single_class_f1']
            print(f'Previously evaluated result has been loaded from {prev_result_path}')
        else:
            # all the results (5 classes)
            all_single_class_precision = np.zeros((len(all_scales), 5))
            all_single_class_recall = np.zeros((len(all_scales), 5))
            all_single_class_AP = np.zeros((len(all_scales), 5))
            all_single_class_f1 = np.zeros((len(all_scales), 5))

        if 'single' in opt.data_config:
            raise NotImplementedError('Not yet implemented')
        else:
            for i, scale in enumerate(all_scales):
                # for continusly testing purpose
                if all_single_class_AP[i, 0] != 0:
                    print(f'Skipping evaluation at scale={scale}')
                    continue

                print(f'Evaluating scale={scale}')
                naming_scale = int(scale*100)
                if 'single' in opt.data_config:
                    raise NotImplementedError('Not yet implemented')
                else:
                    test_path = os.path.join(test_dir, f'test_{naming_scale}_scaled', 'test.txt')

                # class_names = load_classes(test_dir, f'test_{naming_scale}_scaled', data_config["names"])

                precision, recall, AP, f1, ap_class = evaluate(
                    mode='test',
                    purpose=purpose,
                    model=model,
                    path=test_path,
                    iou_thres=opt.iou_thres,
                    conf_thres=opt.conf_thres,
                    nms_thres=opt.nms_thres,
                    img_size=opt.img_size,
                    batch_size=opt.batch_size,
                )

                # save the results
                all_single_class_precision[i] = precision
                all_single_class_recall[i] = recall
                all_single_class_AP[i] = AP
                all_single_class_f1[i] = f1
                print(f"mAP at scale = {scale}: {AP.mean()}")

            np.savez(loss_path,
                    all_single_class_precision=all_single_class_precision,
                    all_single_class_recall=all_single_class_recall,
                    all_single_class_AP=all_single_class_AP,
                    all_single_class_f1=all_single_class_f1)

            print(f'\nTesting result has been saved to {loss_path}')
