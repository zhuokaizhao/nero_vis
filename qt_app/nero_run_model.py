# the script gets called by nero_app when running the model
import os
import sys
import time
import tqdm
import torch
import torchvision
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
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

        elif network_model == 'pre_trained':
            # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
            #                                                                 min_size=image_size)
            model = models.Pre_Trained_FastRCNN()

        model.load_state_dict(torch.load(model_dir, map_location=device))

    # set model in evaluation mode
    model.eval()

    print(f'{network_model} loaded from {model_dir}')

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
            dataset = datasets.MnistDataset(test_image, test_label, transform=transform)
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


# run model on either on a single COCO image or a batch of COCO images
def run_coco_once(model_name, model, test_image, original_names, custom_names, pytorch_names, batch_size=1):

    # prepare input image shapes
    img_size = test_image.shape[0]
    if len(test_image.shape) == 3:
        # reformat input image from (height, width, channel) to (batch size, channel, height, width)
        test_image = test_image.permute((2, 0, 1))[None, :, :, :].float()
    elif len(test_image.shape) == 4:
        # reformat input image from (batch_size, height, width, channel) to (batch_size, batch size, channel, height, width)
        test_image = test_image.permute((0, 3, 1, 2)).float()
    else:
        raise Exception('Wrong input image shape')

    # ensure the model is in evaluation mode
    model.eval()

    # create dataloader to take care of classes conversion, etc
    EVALUATE_TRANSFORMS = transforms.Compose([nero_transform.ConvertLabel(original_names, custom_names),
                                            nero_transform.ToTensor()])

    dataset = datasets.CreateDataset(path, img_size=img_size, transform=EVALUATE_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn
    )


    result_size = int(np.sqrt(len(dataloader.dataset)))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # save the top 3 results (if available)
    # targets labels and bounding boxes
    num_keep = 3

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


    for batch_i, (path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # array index (needed when doing normal evaluation)
        row = batch_i // result_size
        col = batch_i % result_size

        if targets is None:
            continue

        # when we are using pretrained model
        if model_name == 'Pre-trained FasterRCNN':
            for i in range(len(targets)):
                targets[i, 1] = pytorch_names.index(custom_names[int(targets[i, 1]-1)]) + 1

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
                cur_precision, cur_recall, cur_F_measure = nero_utilities.process_model_outputs(outputs, targets)
            # print(cur_confidence_sorted_by_iou)
            if len(cur_pred_label_sorted_by_conf) > 1:
                raise Exception('Multi-object case has not been considered')

            # when the output is not empty
            if cur_pred_label_sorted_by_conf != []:
                # target labels and bounding boxes
                all_target_label[row, col] = targets.numpy()[0, 1]
                all_target_bb[row, col, :] = targets.numpy()[0, 2:]


                # for results sorted by conf
                # extract label information from the highest-confidence result
                # if len(cur_pred_label_sorted_by_conf[0]) < 3:
                #     print('ahhhhh less than 3')
                for i in range(min(len(cur_pred_label_sorted_by_conf[0]), num_keep)):


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

