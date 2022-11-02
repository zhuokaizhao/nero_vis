# handles functions such as NMS
from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
# import nrrd


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness from large to small (highest confidence to lowest)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique target classes from all label files
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    # highest and average confidence of each object class
    max_conf_per_class = []
    mean_conf_per_class = []
    # for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
    for c in unique_classes:
        # all predictions results (True or False) for this class
        i = pred_cls == c
        # Number of ground truth objects
        n_gt = (target_cls == c).sum()
        # Number of predicted objects that has correct label for current class
        n_p = i.sum()
        # confidence
        if (conf[i] != []):
            max_conf_per_class.append(conf[i][0])
            mean_conf_per_class.append(conf[i].mean())
        else:
            max_conf_per_class.append(0)
            mean_conf_per_class.append(0)

        # empty objects
        if n_p == 0 and n_gt == 0:
            continue
        # precision, recall, and ap are all zero when true positive = 0
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        # compute p, r, ap
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall TP/(TP + FN), where (TP + FN) is just the number of ground truths
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    max_conf_per_class = np.array(max_conf_per_class)
    mean_conf_per_class = np.array(mean_conf_per_class)

    return p, r, ap, f1, unique_classes.astype("int32"), max_conf_per_class, mean_conf_per_class


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []

    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


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


def non_max_suppression(prediction_dict, conf_thres, nms_thres):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        N*(x1, y1, x2, y2, conf_score, class_pred)
    """

    # initialize the output list (len(output) = batch_size)
    output = [None for _ in range(len(prediction_dict))]

    for image_i, prediction in enumerate(prediction_dict):

        # prediction dictionary contains 'boxes', 'labels', 'scores'
        # construct the image_pred array with [boxes, conf_scores, class_label_pred]
        image_pred = torch.zeros((len(prediction['boxes']), 6))
        image_pred[:, :4] = prediction['boxes']
        image_pred[:, 4] = prediction['scores']
        image_pred[:, 5] = prediction['labels']

        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Sort by the confidence score
        detections = image_pred[(-image_pred[:, 4]).argsort()]

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            # compute if the current bounding box has a large overlap over all others
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # check if the current bounding box has the same label over all others
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def process_model_outputs(outputs, targets, iou_thres=0.5, conf_thres=1e-4):

    # get all the outputs (the model proposes multiple bounding boxes for each object)
    all_qualified_outputs_sorted_by_conf = []
    all_qualified_outputs_sorted_by_iou = []
    all_qualified_outputs_sorted_by_conf_iou = []
    all_qualified_outputs_sorted_by_conf_iou_pr = []
    all_precisions = []
    all_recalls = []
    all_F_measure = []

    # loop through each object
    for i in range(len(outputs)):

        if outputs[i] is None:
            continue

        # output has to pass the conf threshold to filter out noisy predictions
        # faster-rcnn output has format (x1, y1, x2, y2, conf, class_pred)
        output = outputs[i]
        cur_object_outputs = []
        for j in range(len(output)):
            if output[j, 4] >= conf_thres:
                cur_object_outputs.append(output[j].numpy())

        cur_object_outputs = np.array(cur_object_outputs)
        cur_object_outputs = torch.from_numpy(cur_object_outputs)

        # cur_object_outputs = outputs[i]

        # all predicted labels and true label for the current object
        pred_labels = cur_object_outputs[:, -1].numpy()
        true_labels = targets[targets[:, 0] == i][:, 1].numpy()

        # all predicted bounding boxes and true bb for the current object
        pred_boxes = cur_object_outputs[:, :4]
        true_boxes = targets[targets[:, 0] == i][:, 2:]
        cur_qualified_output = []
        num_true_positive = 0
        num_false_positive = 0

        # loop through all the proposed bounding boxes
        for j, pred_box in enumerate(pred_boxes):
            # print(f'True label: {true_labels[i]}, Pred label: {pred_labels[j]}')
            # if the label is predicted correctly
            if pred_labels[j] == true_labels[i]:
                # compute iou
                cur_iou = bbox_iou(pred_box.unsqueeze(0), true_boxes)
                # if iou passed threshold
                if cur_iou > iou_thres:
                    # save current output
                    cur_qualified_output.append(np.append(cur_object_outputs[j].numpy(), cur_iou))
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
        recall = num_true_positive / (len(true_boxes) + 1e-16)
        # F-measure = (2 * Precision * Recall) / (Precision + Recall)
        F_measure = (2 * precision * recall) / (precision + recall + 1e-16)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_F_measure.append(F_measure)

        # sort all the qualified output based on conf (class_score) from high to low
        cur_qualified_output_sorted_by_conf = sorted(cur_qualified_output, key=lambda x: x[4])[::-1]

        # sort all the qualified output based on iou from high to low
        cur_qualified_output_sorted_by_iou = sorted(cur_qualified_output, key=lambda x: x[-1])[::-1]

        # sort all the qualified output based on conf*iou from high to low
        cur_qualified_output_sorted_by_conf_iou = sorted(cur_qualified_output, key=lambda x: x[4]*x[-1])[::-1]

        # sort all the qualified output based on conf*iou from high to low
        cur_qualified_output_sorted_by_conf_iou_pr = sorted(cur_qualified_output, key=lambda x: x[4]*x[-1]*precision)[::-1]

        # keep the outputs
        all_qualified_outputs_sorted_by_conf.append(cur_qualified_output_sorted_by_conf)
        all_qualified_outputs_sorted_by_iou.append(cur_qualified_output_sorted_by_iou)
        all_qualified_outputs_sorted_by_conf_iou.append(cur_qualified_output_sorted_by_conf_iou)
        all_qualified_outputs_sorted_by_conf_iou_pr.append(cur_qualified_output_sorted_by_conf_iou_pr)

    # convert list to numpy array
    all_qualified_outputs_sorted_by_conf = np.array(all_qualified_outputs_sorted_by_conf)
    all_qualified_outputs_sorted_by_iou = np.array(all_qualified_outputs_sorted_by_iou)
    all_qualified_outputs_sorted_by_conf_iou = np.array(all_qualified_outputs_sorted_by_conf_iou)
    all_qualified_outputs_sorted_by_conf_iou_pr = np.array(all_qualified_outputs_sorted_by_conf_iou_pr)

    # divide outputs into perspective names
    # sorted by conf
    all_pred_labels_sorted_by_conf = []
    all_pred_bb_sorted_by_conf = []
    all_confidence_sorted_by_conf = []
    all_iou_sorted_by_conf = []

    # sorted by iou
    all_pred_labels_sorted_by_iou = []
    all_pred_bb_sorted_by_iou = []
    all_confidence_sorted_by_iou = []
    all_iou_sorted_by_iou = []

    # sorted by conf*iou
    all_pred_labels_sorted_by_conf_iou = []
    all_pred_bb_sorted_by_conf_iou = []
    all_confidence_sorted_by_conf_iou = []
    all_iou_sorted_by_conf_iou = []

    # sorted by conf*iou*pr
    all_pred_labels_sorted_by_conf_iou_pr = []
    all_pred_bb_sorted_by_conf_iou_pr = []
    all_confidence_sorted_by_conf_iou_pr = []
    all_iou_sorted_by_conf_iou_pr = []

    for i in range(len(all_qualified_outputs_sorted_by_conf)):
        output_sorted_by_conf = all_qualified_outputs_sorted_by_conf[i]
        output_sorted_by_iou = all_qualified_outputs_sorted_by_iou[i]
        output_sorted_by_conf_iou = all_qualified_outputs_sorted_by_conf_iou[i]
        output_sorted_by_conf_iou_pr = all_qualified_outputs_sorted_by_conf_iou_pr[i]

        # if current output is empty
        if len(output_sorted_by_conf) == 0:
            continue

        pred_label_sorted_by_conf = output_sorted_by_conf[:, -2]
        pred_box_sorted_by_conf = output_sorted_by_conf[:, :4]
        pred_confidence_sorted_by_conf = output_sorted_by_conf[:, 4]
        iou_sorted_by_conf = output_sorted_by_conf[:, -1]

        pred_label_sorted_by_iou = output_sorted_by_iou[:, -2]
        pred_box_sorted_by_iou = output_sorted_by_iou[:, :4]
        pred_confidence_sorted_by_iou = output_sorted_by_iou[:, 4]
        iou_sorted_by_iou = output_sorted_by_iou[:, -1]

        pred_label_sorted_by_conf_iou = output_sorted_by_conf_iou[:, -2]
        pred_box_sorted_by_conf_iou = output_sorted_by_conf_iou[:, :4]
        pred_confidence_sorted_by_conf_iou = output_sorted_by_conf_iou[:, 4]
        iou_sorted_by_conf_iou = output_sorted_by_conf_iou[:, -1]

        pred_label_sorted_by_conf_iou_pr = output_sorted_by_conf_iou_pr[:, -2]
        pred_box_sorted_by_conf_iou_pr = output_sorted_by_conf_iou_pr[:, :4]
        pred_confidence_sorted_by_conf_iou_pr = output_sorted_by_conf_iou_pr[:, 4]
        iou_sorted_by_conf_iou_pr = output_sorted_by_conf_iou_pr[:, -1]

        all_pred_labels_sorted_by_conf.append(pred_label_sorted_by_conf)
        all_pred_bb_sorted_by_conf.append(pred_box_sorted_by_conf)
        all_confidence_sorted_by_conf.append(pred_confidence_sorted_by_conf)
        all_iou_sorted_by_conf.append(iou_sorted_by_conf)

        all_pred_labels_sorted_by_iou.append(pred_label_sorted_by_iou)
        all_pred_bb_sorted_by_iou.append(pred_box_sorted_by_iou)
        all_confidence_sorted_by_iou.append(pred_confidence_sorted_by_iou)
        all_iou_sorted_by_iou.append(iou_sorted_by_iou)

        all_pred_labels_sorted_by_conf_iou.append(pred_label_sorted_by_conf_iou)
        all_pred_bb_sorted_by_conf_iou.append(pred_box_sorted_by_conf_iou)
        all_confidence_sorted_by_conf_iou.append(pred_confidence_sorted_by_conf_iou)
        all_iou_sorted_by_conf_iou.append(iou_sorted_by_conf_iou)

        all_pred_labels_sorted_by_conf_iou_pr.append(pred_label_sorted_by_conf_iou_pr)
        all_pred_bb_sorted_by_conf_iou_pr.append(pred_box_sorted_by_conf_iou_pr)
        all_confidence_sorted_by_conf_iou_pr.append(pred_confidence_sorted_by_conf_iou_pr)
        all_iou_sorted_by_conf_iou_pr.append(iou_sorted_by_conf_iou_pr)


    return all_pred_labels_sorted_by_conf, all_pred_bb_sorted_by_conf, all_confidence_sorted_by_conf, all_iou_sorted_by_conf, \
            all_pred_labels_sorted_by_iou, all_pred_bb_sorted_by_iou, all_confidence_sorted_by_iou, all_iou_sorted_by_iou, \
            all_pred_labels_sorted_by_conf_iou, all_pred_bb_sorted_by_conf_iou, all_confidence_sorted_by_conf_iou, all_iou_sorted_by_conf_iou, \
            all_pred_labels_sorted_by_conf_iou_pr, all_pred_bb_sorted_by_conf_iou_pr, all_confidence_sorted_by_conf_iou_pr, all_iou_sorted_by_conf_iou_pr, \
            all_precisions, all_recalls, all_F_measure

