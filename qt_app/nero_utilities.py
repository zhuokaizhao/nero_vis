# the script contains utility functions
import copy
import torch
import numpy as np
from PIL import Image
from PySide6 import QtGui


# load the class names from the COCO class file
def load_coco_classes_file(path):

    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

# some helper functions
def qt_image_to_tensor(img, share_memory=False):
    """
    Creates a PyTorch Tensor from a QImage via converting to Numpy array first.

    If share_memory is True, the numpy array and the QImage is shared.
    Be careful: make sure the numpy array is destroyed before the image,
                otherwise the array will point to unreserved memory!!
    """
    assert isinstance(img, QtGui.QImage), "img must be a QtGui.QImage object"
    if img.format() == QtGui.QImage.Format.Format_Grayscale8:
        num_channel = 1
    elif img.format() == QtGui.QImage.Format.Format_RGB32:
        # RGBA
        num_channel = 4

    img_size = img.size()
    buffer = img.constBits()

    # Sanity check
    n_bits_buffer = len(buffer) * 8
    n_bits_image  = img_size.width() * img_size.height() * img.depth()

    assert n_bits_buffer == n_bits_image, "size mismatch: {} != {}".format(n_bits_buffer, n_bits_image)

    # Note the different width height parameter order!
    arr = np.ndarray(shape  = (img_size.height(), img_size.width(), num_channel),
                     buffer = buffer,
                     dtype  = np.uint8)

    if share_memory:
        return torch.from_numpy(arr)
    else:
        return torch.from_numpy(copy.deepcopy(arr))


def tensor_to_qt_image(img_pt, new_size=None):

    # convert input torch tensor to numpy array
    img_np = img_pt.numpy()

    # make sure numpy array is c contiguous, as required by QImage
    if not img_np.flags['C_CONTIGUOUS']:
        img_np = np.ascontiguousarray(img_np)

    if img_np.shape[-1] == 1:
        # qt image uses width, height
        img_qt = QtGui.QImage(img_np, img_np.shape[1], img_np.shape[0], QtGui.QImage.Format_Grayscale8)
    elif img_np.shape[-1] == 3:
        img_qt = QtGui.QImage(img_np, img_np.shape[1], img_np.shape[0], QtGui.QImage.Format_RGB888)

    if new_size != None:
        img_qt = img_qt.scaled(new_size, new_size)

    return img_qt


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

# lerp between two ranges
def lerp(pos, in_min, in_max, out_min, out_max):

    # an array of numbers of be lerped
    if type(pos) is np.ndarray:
        alpha = np.true_divide(np.subtract(pos, in_min), float(in_max - in_min))
        new_pos = np.multiply(np.subtract(1.0, alpha), float(out_min)) + np.multiply(alpha, float(out_max))
    # a float number
    else:
        alpha = float(pos - in_min) / float(in_max - in_min)
        new_pos = float(1.0 - alpha) * float(out_min) + alpha * float(out_max)

    return new_pos


# compute average precision
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


# loss that might be used during training or testing
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    # to align with MSELoss's convention on reduction setting
    def forward(self, x, y, reduction='mean'):
        mse_loss = torch.nn.MSELoss(reduction=reduction)
        loss = torch.sqrt(mse_loss(x, y))
        return loss


class AEELoss(torch.nn.Module):
    def __init__(self):
        super(AEELoss, self).__init__()

    def forward(self, x, y, reduction='mean'):
        aee_loss = ((x[:, :, 0] - y[:, :, 0]).square() + (x[:, :, 1] - y[:, :, 1]).square()).sqrt()

        if reduction == 'mean':
            return aee_loss.mean()
        elif reduction == 'none':
            return aee_loss
