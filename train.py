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
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from gluoncv import data, utils
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa

from terminaltables import AsciiTable

import test
# import model
import prepare_dataset
import data_transform


os.environ['CUDA_VISIBLE_DEVICES']='0'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # purpose of generating dataset (shift-equivariance, rotation-equivariance, scale-equivariance)
    parser.add_argument('-p', '--purpose', action='store', nargs=1, dest='purpose')
    # input data related
    parser.add_argument("--data_name", type=str, help="Defines the name of the dataset (mnist or coco)")
    parser.add_argument("--data_config", type=str, help="path to data config file")
    # training parameters
    parser.add_argument("--percentage", type=int, help="percentage of jittering or scailing")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="size of each image batch")
    parser.add_argument("--checkpoint_path", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--log_dir", type=str, help="Defines the directory where the training log files are stored")
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose")
    # output model directory
    parser.add_argument('-m', '--model_dir', action='store', type=str, dest='model_dir')
    # model type (normal or shift-invariant)
    parser.add_argument('--type', action='store', type=str, dest='model_type')

    opt = parser.parse_args()

    # read parameters
    purpose = opt.purpose[0]
    data_name = opt.data_name
    if opt.model_type != None:
        model_type = opt.model_type
    else:
        model_type = 'normal'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    data_config = prepare_dataset.parse_data_config(opt.data_config)
    num_classes = int(data_config['classes'])
    train_path = data_config['train']
    valid_path = data_config['valid']

    # get the percentage from mode (20-jittering mode has 20 percent jittering)
    percentage = int(opt.percentage)

    # make model and log directory
    model_dir = opt.model_dir
    log_dir = opt.log_dir
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # different dataset configuration files are different
    if data_name == 'mnist':
        # original mnist names and desired mnist names are the same
        original_names = prepare_dataset.load_classes(data_config["desired_names"])
        desired_names = prepare_dataset.load_classes(data_config["desired_names"])
    elif data_name == 'coco':
        original_names = prepare_dataset.load_classes(data_config["original_names"])
        desired_names = prepare_dataset.load_classes(data_config["desired_names"])

    # print out inputs info
    print(f'\nPurpose of training: {purpose}')
    print(f'Dataset name: {data_name}')
    print(f'Training model type: {model_type}\n')
    if purpose == 'shift-equivariance':
        print(f'Jittering percentage: {percentage}')
    elif purpose == 'rotation-equivariance':
        print(f'Rotation percentage: {percentage}')
    elif purpose == 'scale-equivariance':
        print(f'Scale percentage: {percentage}')
    print(f'Model output dir: {model_dir}')
    print(f'log dir: {log_dir}')
    print(f'Training data path: {train_path}')
    print(f'Validation data path: {valid_path}\n')

    # Initiate model
    # model = model.faster_rcnn(num_classes, img_size=opt.img_size, backbone_name='mobilenet_v2')
    backbone_name = 'resnet50'
    # backbone_name = 'mobilenet-v2'
    if 'resnet' in backbone_name:
        # model = model.faster_rcnn(num_classes, img_size=opt.img_size, backbone_name=backbone_name)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                     num_classes=num_classes+1,
                                                                     pretrained_backbone=True,
                                                                     min_size=opt.img_size)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    model.to(device)

    # If specified we start from checkpoint
    previous_epochs = 0
    if opt.checkpoint_path:
        import re
        previous_epochs = int(re.findall(r'\d+', opt.checkpoint_path)[-1])
        model.load_state_dict(torch.load(opt.checkpoint_path))
        print(f'\nPrevious checkpoint model weight which has been trained {previous_epochs} epochs has been loaded.')

    # Get dataloader
    # dataset augmentation (jittering on the fly)
    if purpose == 'shift-equivariance':
        AUGMENTATION_TRANSFORMS = transforms.Compose([data_transform.GaussianJittering(opt.img_size, percentage),
                                                        data_transform.ShiftEqvAug(),
                                                        data_transform.ConvertLabel(original_names, desired_names),
                                                        data_transform.ToTensor()])
    elif purpose == 'rotation-equivariance':
        AUGMENTATION_TRANSFORMS = transforms.Compose([data_transform.RandomRotationShift(img_size=opt.img_size,
                                                                                         fixed_rot=False,
                                                                                         fixed_shift=False,
                                                                                         rot_percentage=percentage,
                                                                                         shift_percentage=100),
                                                        data_transform.RotEqvAug(),
                                                        data_transform.ConvertLabel(original_names, desired_names),
                                                        data_transform.ToTensor()])
    dataset = prepare_dataset.CreateDataset(list_path=train_path,
                                            img_size=opt.img_size,
                                            transform=AUGMENTATION_TRANSFORMS)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # save the train loss
    all_train_losses = []
    all_val_losses = []
    all_val_mAPs = []

    # load the previous train/val loss and append
    if purpose == 'shift-equivariance':
        prev_train_loss_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-jittered_train_loss_1_{previous_epochs}.npy')
        prev_val_loss_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-jittered_val_loss_1_{previous_epochs}.npy')
        prev_val_mAP_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-jittered_val_mAP_1_{previous_epochs}.npy')
    elif purpose == 'rotation-equivariance':
        prev_train_loss_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-rotated_train_loss_1_{previous_epochs}.npy')
        prev_val_loss_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-rotated_val_loss_1_{previous_epochs}.npy')
        prev_val_mAP_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-rotated_val_mAP_1_{previous_epochs}.npy')


    if os.path.isfile(prev_train_loss_path):
        prev_train_loss = np.load(prev_train_loss_path)
        prev_val_loss = np.load(prev_val_loss_path)
        prev_val_mAP = np.load(prev_val_mAP_path)

        # append the new train and val loss
        all_train_losses = list(prev_train_loss) + all_train_losses
        all_val_losses = list(prev_val_loss) + all_val_losses
        all_val_mAPs = list(prev_val_mAP) + all_val_mAPs

        print(f'\nPrevious training stats loaded')
        print(f'Number of prev_train_loss: {len(prev_train_loss)}')
        print(f'Number of prev_val_loss: {len(prev_val_loss)}')
        print(f'Number of prev_val_mAP: {len(prev_val_mAP)}\n')


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(model.parameters())
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    for epoch in range(previous_epochs+1, previous_epochs+1+int(opt.epochs)):
        model.train()
        start_time = time.time()
        train_batch_loss = []

        for batch_i, (_, imgs, labels) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):

            # visualize for debugging purposes
            # utils.viz.plot_bbox(imgs[0].permute(1, 2, 0).numpy()*255, labels[:, 2:].numpy(), scores=None,
            #                     labels=labels[:, 1].numpy()-1, class_names=desired_names)
            # plt.show()

            # put images and targets on GPU
            imgs = Variable(imgs.to(device))
            labels = Variable(labels.to(device), requires_grad=False)

            # put targets in dictionary format that faster-rcnn takes
            targets = []
            for i in range(opt.batch_size):
                target = {}
                # target box has to be in shape (N, 4)
                target['boxes'] = labels[i:i+1, 2:]
                target['labels'] = labels[i:i+1, 1].type(torch.int64)
                targets.append(target)

            # print(targets)
            # exit()

            # forward pass
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            train_batch_loss.append(losses.item())

            # backward and optimizer step
            # optimizer.zero_grad()
            losses.backward()

            batches_done = len(dataloader) * epoch + batch_i
            if batches_done % 2 == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

        all_train_losses.append(np.mean(train_batch_loss))
        print(f'Epoch {epoch} loss: {all_train_losses[-1]}\n')

        if epoch % opt.evaluation_interval == 0:
            print("---- Evaluating Model ----")

            # Evaluate the model on the validation set
            if purpose == 'shift-equivariance':
                loss, metrics_output = test.evaluate(
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
                    rot_percentage=100,
                    shift_percentage=percentage
                )
            elif purpose == 'rotation-equivariance':
                loss, metrics_output = test.evaluate(
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
                    rot_percentage=percentage,
                    shift_percentage=100
                )

            all_val_losses.append(np.mean(loss))

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean()),
                ]

                all_val_mAPs.append(AP.mean())

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, desired_names[c-1], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}\n\n")
            else:
                print( "---- mAP not measured (no detections found by model)")

        if epoch % opt.checkpoint_interval == 0:
            if purpose == 'shift-equivariance':
                model_path = os.path.join(model_dir, f'faster_rcnn_{backbone_name}_{percentage}-jittered_ckpt_{epoch}.pth')
            elif purpose == 'rotation-equivariance':
                model_path = os.path.join(model_dir, f'faster_rcnn_{backbone_name}_{percentage}-rotated_ckpt_{epoch}.pth')

            torch.save(model.state_dict(), model_path)
            print(f'\nmodel has been saved to {model_path}')

            # save the train losses and val mAPs
            start_epoch = previous_epochs+1
            end_epoch = previous_epochs+int(opt.epochs)

            # save the losses
            print(f'Number of all_train_losses: {len(all_train_losses)}')
            print(f'Number of all_val_losses: {len(all_val_losses)}')
            print(f'Number of all_val_mAPs: {len(all_val_mAPs)}')
            if purpose == 'shift-equivariance':
                train_loss_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-jittered_train_loss_1_{epoch}.npy')
                val_loss_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-jittered_val_loss_1_{epoch}.npy')
                val_mAP_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-jittered_val_mAP_1_{epoch}.npy')
            elif purpose == 'rotation-equivariance':
                train_loss_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-rotated_train_loss_1_{epoch}.npy')
                val_loss_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-rotated_val_loss_1_{epoch}.npy')
                val_mAP_path = os.path.join(log_dir, f'faster_rcnn_{backbone_name}_{percentage}-rotated_val_mAP_1_{epoch}.npy')

            # save them
            np.save(train_loss_path, all_train_losses)
            np.save(val_loss_path, all_val_losses)
            np.save(val_mAP_path, all_val_mAPs)
            print(f'\ntraining loss has been saved/updated to {train_loss_path}')
            print(f'val loss has been saved/updated to {val_loss_path}')
            print(f'val mAP has been saved/updated to {val_mAP_path}')
