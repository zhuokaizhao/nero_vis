# the scripts processes the training dataset from COCO datasets for rotation-equivariant experiments
import os
import cv2
import csv
import json
import torch
import random
import plotly
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from shutil import copy
from matplotlib import colors
from gluoncv import data, utils
import plotly.offline as offline
from pycocotools.coco import COCO
import plotly.graph_objects as go
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from gluoncv.data.transforms import mask as tmask

# all randomness are seedable
random.seed(0)
np.random.seed(0)


# helper function that computes labels for cut out images
def compute_label(cur_bounding_box, x_min, y_min, image_size):
    # convert key object bounding box to be based on extracted image
    x_min_center_bb = cur_bounding_box[0] - x_min
    y_min_center_bb = cur_bounding_box[1] - y_min
    x_max_center_bb = cur_bounding_box[2] - x_min
    y_max_center_bb = cur_bounding_box[3] - y_min

    # compute the center of the object in the extracted image
    object_center_x = (x_min_center_bb + x_max_center_bb) / 2
    object_center_y = (y_min_center_bb + y_max_center_bb) / 2

    # compute the width and height of the real bounding box of this object
    original_bb_width = cur_bounding_box[2] - cur_bounding_box[0]
    original_bb_height = cur_bounding_box[3] - cur_bounding_box[1]

    # compute the range of the bounding box, do the clamping if go out of extracted image
    bb_min_x = max(0, object_center_x - original_bb_width/2)
    bb_max_x = min(image_size[1]-1, object_center_x + original_bb_width/2)
    bb_min_y = max(0, object_center_y - original_bb_height/2)
    bb_max_y = min(image_size[0]-1, object_center_y + original_bb_height/2)

    return bb_min_x, bb_min_y, bb_max_x, bb_max_y


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Preprocessing COCO dataset for rotation-equivariant experiments')
    # train, val, test, single-test or single-test-all-candidates
    parser.add_argument('-t', '--data-type', action='store', nargs=1, dest='data_type', required=True)
    # input directory for original COCO dataset
    parser.add_argument('-i', '--input-dir', action='store', nargs=1, dest='input_dir', required=True)
    # output directory for preprocessed COCO
    parser.add_argument('-o', '--output-dir', action='store', nargs=1, dest='output_dir', required=True)
    # image cut out size (default 128x128)
    parser.add_argument('-s', '--image-size', action='store', nargs=1, dest='image_size')
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    # visualizing samples
    parser.add_argument('--vis', action='store_true', dest='vis', default=False)

    args = parser.parse_args()

    # input variables
    data_type = args.data_type[0]
    input_dir = args.input_dir[0]
    output_dir = args.output_dir[0]
    os.makedirs(output_dir, exist_ok=True)
    # target image cut out size (height * width)
    if args.image_size != None:
        image_size = (int(args.image_size[0].split('x')[0]), int(args.image_size[0].split('x')[1]))
    else:
        image_size = (128, 128)
    verbose = args.verbose
    vis = args.vis

    # parameters
    # desired classes for extracted dataset
    if data_type == 'single-test-all-candidates':
        desired_classes = ['car', 'bottle', 'cup', 'chair', 'book']
    else:
        desired_classes = ['car', 'bottle', 'cup', 'chair', 'book']
    # downsample factor when determining if an image is qualified
    downsample_factor = 1
    # object sizes upper limit
    size_limit = 1.0

    # sets of parameters for different types of datasets
    if data_type == 'train' or data_type == 'val' or data_type == 'test':
        # number of images in output dataset
        if data_type == 'train':
            # image folder directory for later copying the original images (instead of re-saving)
            images_folder_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/train2017'
            json_path = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/annotations/instances_train2017.json'
            dataset = data.COCODetection(root=input_dir, splits=['instances_train2017'], skip_empty=False)
            num_images = len(dataset)
            start_index = 0
        else:
            # image folder directory for later copying the original images (instead of re-saving)
            images_folder_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/val2017'
            json_path = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/annotations/instances_val2017.json'
            dataset = data.COCODetection(root=input_dir, splits=['instances_val2017'], skip_empty=False)
            num_images = len(dataset)
            start_index = 0

        # translation range used to check to make sure current object supports all possible translation
        # happened during training (in the augmentation scheme)
        x_translation = [-64, 64]
        y_translation = [-64, 64]
        # all the class names from the original COCO
        class_names = np.array(dataset.classes)

        # load json for getting image/label paths
        all_image_names = []

        _coco = COCO(json_path)
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(input_dir, dirname, filename)
            all_image_names.append(abs_path)


    elif data_type == 'single-test':
        # the images are from both train and val
        dataset = data.COCODetection(root=input_dir, splits=['instances_val2017'], skip_empty=True)

        # car, bottle, cup, chair, book
        all_image_indices = [174, 2623, 1680, 2760, 1144]
        all_label_indices = [0, 9, 0, 4, 6]

        x_translation = list(range(-image_size[0]//2, image_size[0]//2))
        y_translation = list(range(-image_size[1]//2, image_size[1]//2))
        # all the class names from the original COCO
        class_names = np.array(dataset.classes)


    # one object class, all candidates single test
    elif data_type == 'single-test-all-candidates':
        # cup class
        dataset = data.COCODetection(root=input_dir, splits=['instances_val2017'], skip_empty=True)

        # translations in x and y
        positive_trans = [1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 29, 37, 48, 63]
        negative_trans = list(reversed(positive_trans))
        negative_trans = [-x for x in negative_trans]
        x_translation = negative_trans + [0] + positive_trans
        y_translation = negative_trans + [0] + positive_trans

        # all the class names from the original COCO
        class_names = np.array(dataset.classes)

    if verbose:
        print(f'\nData type: {data_type}')
        print(f'Input data dir: {input_dir}')
        print(f'Target image size: {image_size}')
        print(f'Output data dir: {output_dir}')
        if not 'single-test' in data_type:
            print(f'Output number of images (per dataset): {num_images}')

    # generating training and validation dataset that is used for all levels of jittering training
    if data_type == 'train' or data_type == 'val' or data_type == 'test':

        print(f'\nGenerating {data_type} dataset for all jittering levels')
        # extracted image count
        num_extracted = 0

        # create images and labels folder inside the (current) test_dir
        image_dir = os.path.join(output_dir, 'images')
        label_dir = os.path.join(output_dir, 'labels')
        # create image and label dirs
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        # all the image paths
        all_image_paths = []

        # some information we want to collect when generating train datasets
        # number of potential target bb rejected for not having enough surrounding area
        num_rej_target_no_space = 0
        # number of potential target bb rejected for bounding boxes being too large
        num_rej_target_large_bb = 0
        # number of potential target bb rejected for bounding boxes being too small
        num_rej_target_small_bb = 0
        # total number of potential target bb
        num_target_bb = 0
        # total number of qualified target bb
        num_target_passed = 0

        # go through all images in dataset
        for i in tqdm(range(start_index, len(dataset))):

            # stop if all done
            if num_extracted >= num_images:
                break

            # current image and its label
            cur_image, cur_label = dataset[i]
            # convert image to numpy
            cur_image = cur_image.asnumpy()
            # height and width of the image
            height, width = cur_image.shape[:2]

            # randomly permutate the labels
            cur_label_randomized = np.random.permutation(cur_label)
            # all the bounding boxes and their correponding class ids of the current single image
            cur_bounding_boxes = cur_label_randomized[:, :4]
            cur_class_ids = cur_label_randomized[:, 4:5][:, 0].astype(int)

            if downsample_factor != 1:
                # convert to PIL image for up/downsampling
                cur_image_scaled = Image.fromarray(cur_image)
                # resize the image by downsample_factor
                cur_image_scaled = cur_image_scaled.resize((width//downsample_factor, height//downsample_factor), Image.LANCZOS)
                # convert to numpy array
                cur_image_scaled = np.array(cur_image_scaled)
                # scale the bounding boxes accordingly
                cur_bounding_boxes_scaled = cur_bounding_boxes // downsample_factor
                # height and width are scaled too
                height, width = cur_image_scaled.shape[:2]
            else:
                cur_image_scaled = cur_image
                cur_bounding_boxes_scaled = cur_bounding_boxes

            # go through all the labels in this current image
            for m, k in enumerate(cur_class_ids):
                # stop if all done
                if num_extracted >= num_images:
                    break

                # find objects within desired labels
                if class_names[k] in desired_classes:

                    # current bounding box
                    cur_bb = cur_bounding_boxes_scaled[m]

                    # potential target_bb number that has not gone through any checks
                    num_target_bb += 1

                    # obtain the list of labels for this extracted image
                    cut_out_labels = []

                    # compute the center of the current key object
                    key_bb_min_x = cur_bb[0]
                    key_bb_min_y = cur_bb[1]
                    key_bb_max_x = cur_bb[2]
                    key_bb_max_y = cur_bb[3]
                    center_x = (key_bb_min_x + key_bb_max_x) / 2
                    center_y = (key_bb_min_y + key_bb_max_y) / 2

                    # get the fov locations wrt original image
                    not_qualified = False
                    for x_tran in x_translation:
                        for y_tran in y_translation:
                            x_min = center_x - image_size[1]//2 - x_tran
                            y_min = center_y - image_size[0]//2 - y_tran
                            x_max = center_x + image_size[1]//2 - x_tran
                            y_max = center_y + image_size[0]//2 - y_tran

                            # make sure that the extracted image fits in the original image
                            if x_min < 0 or x_max >= width or y_min < 0 or y_max >= height:
                                num_rej_target_no_space += 1
                                not_qualified = True
                                break

                        if not_qualified:
                            break

                    if not_qualified:
                        continue

                    # reject the candidates whose bounding box is too large for the extracted image
                    if (key_bb_max_x-key_bb_min_x)*(key_bb_max_y-key_bb_min_y) > 0.5*size_limit*image_size[0]*image_size[1]:
                        num_rej_target_large_bb += 1
                        continue
                    # reject the candidates whose bounding box is too small
                    elif (key_bb_max_x-key_bb_min_x)*(key_bb_max_y-key_bb_min_y) < 0.01*image_size[0]*image_size[1]:
                        num_rej_target_small_bb += 1
                        continue
                    # current target object passes all criteria
                    else:
                        num_target_passed += 1

                    # save the original label but record which bb is the target
                    # i indicates its original index in COCO (for human debug)
                    # m indicates its target bb index (for training loading data)
                    cut_out_labels_path = os.path.join(label_dir, f'COCO_{i}_bb_{m}.txt')

                    # save the scaled label
                    with open(cut_out_labels_path, 'w') as myfile:
                        for a in range(len(cur_label_randomized)):
                            class_id = cur_class_ids[a]
                            x_min, y_min, x_max, y_max = cur_bounding_boxes_scaled[a]
                            line = str(int(class_id)) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max)
                            myfile.write(line + '\n')

                    # copy the original image over if no scale is present
                    if downsample_factor == 1:
                        cur_image_name = all_image_names[i]
                        img_extension = cur_image_name.split('.')[-1]
                        cut_out_image_path = os.path.join(image_dir, f'COCO_{i}_bb_{m}.' + img_extension)
                        copy(os.path.join(images_folder_dir, cur_image_name), cut_out_image_path)
                    else:
                        # save the image
                        cut_out_image_path = os.path.join(image_dir, f'COCO_{i}_bb_{m}.png')
                        # get a RGB copy for saving
                        cut_out_image_rgb = cv2.cvtColor(cur_image_scaled, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(cut_out_image_path, cut_out_image_rgb)

                    # visualize scaled image with labels
                    if vis:
                        print(cut_out_image_path)
                        utils.viz.plot_bbox(cur_image_scaled, cur_bounding_boxes_scaled, scores=None,
                                                labels=cur_class_ids, class_names=class_names)
                        plt.show()
                        cur_image_scaled = Image.open(cut_out_image_path)
                        cur_image_scaled = np.asarray(cur_image_scaled)
                        utils.viz.plot_bbox(cur_image_scaled, cur_bounding_boxes_scaled, scores=None,
                                                labels=cur_class_ids, class_names=class_names)
                        plt.show()

                    # update the list of saved images
                    all_image_paths.append(cut_out_image_path)

                    # update the number of extracted image
                    num_extracted += 1

        # save all the image paths
        path = os.path.join(output_dir, f'{data_type}.txt')
        with open(path, 'w') as myfile:
            for img_path in all_image_paths:
                myfile.write('%s\n' % img_path)

        print(f'\n{num_extracted} train image/label pairs have been generated and saved')

        # save the label names
        name_path = os.path.join(output_dir, 'custom.names')
        if not os.path.isfile(name_path):
            with open(name_path, 'w') as name_file:
                for label in desired_classes:
                    name_file.write('%s\n' % label)

            print(f'\ncustom.names have been generated and saved')

        # show the rejection rate
        print(f'\ntarget object passed: {num_target_passed}/{num_target_bb}')
        print(f'target object rejected: no enough area: {num_rej_target_no_space}/{num_target_bb}')
        print(f'target object rejected: bb too large: {num_rej_target_large_bb}/{num_target_bb}')
        print(f'target object rejected: bb too small: {num_rej_target_small_bb}/{num_target_bb}')
        rejection_path = os.path.join(output_dir, f'rej_information.npz')
        np.savez(rejection_path,
                num_target_bb=num_target_bb,
                num_target_passed=num_target_passed,
                num_rej_target_no_space=num_rej_target_no_space,
                num_rej_target_large_bb=num_rej_target_large_bb,
                num_rej_target_small_bb=num_rej_target_small_bb)
        print(f'Rejection information has been saved to {rejection_path}')

        # save the pie chart of rejection rate
        # target objects rejection status figure
        labels = 'passed', 'not enough space\n(original image\ntoo small)', f'bb too large\n({size_limit*100}% threshold)', 'bb too small\n(1% threshold)'
        sizes = [num_target_passed*100/num_target_bb,
                num_rej_target_no_space*100/num_target_bb,
                num_rej_target_large_bb*100/num_target_bb,
                num_rej_target_small_bb*100/num_target_bb]
        explode = (0.1, 0, 0, 0)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        target_rej_fig_path = os.path.join(output_dir, f'target_object_rej_stats.png')
        plt.savefig(target_rej_fig_path)
        print(f'\nTarget rejection status figure has been saved to {target_rej_fig_path}')


    # generating single-test dataset
    elif data_type == 'single-test':

        # each object class has a representing image
        for i, cur_class in enumerate(desired_classes):
            print(f'\nCurrent processing object class: {cur_class}')
            cur_class = cur_class.replace(" ", "_")

            image_index = all_image_indices[i]
            label_index = all_label_indices[i]
            print(f'Image index: {image_index}')
            print(f'Label index: {label_index}')

            # define and create paths
            test_dir = os.path.join(output_dir, cur_class)
            os.makedirs(test_dir, exist_ok=True)
            image_dir = os.path.join(test_dir, 'images')
            label_dir = os.path.join(test_dir, 'labels')
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            # current image and its label
            cur_image, cur_label = dataset[image_index]
            # convert image to numpy
            cur_image = cur_image.asnumpy()
            # height and width of the image
            height, width = cur_image.shape[:2]

            if downsample_factor != 1:
                # convert to PIL image for downsampling
                cur_image = Image.fromarray(cur_image.asnumpy())
                # downsample the image by downsample_factor
                cur_image = cur_image.resize((cur_image.width//downsample_factor, cur_image.height//downsample_factor), Image.LANCZOS)
                # convert to numpy array
                cur_image = np.array(cur_image)
                # update height and width of the image
                height, width = cur_image.shape[:2]
                # all the bounding boxes and their correponding class ids of the current single image
                # because of downsampling, all label values are divided by downsample_factor
                cur_bounding_boxes = cur_label[:, :4] / downsample_factor
            else:
                cur_bounding_boxes = cur_label[:, :4]

            cur_class_ids = cur_label[:, 4:5][:, 0].astype(int)

            # the label we are using
            label = cur_class_ids[label_index]

            # all the image paths
            all_image_paths = []
            # all cut out images' x_min and y_min (used for visualization)
            all_cut_out_xy = []

            for tran in tqdm(range(len(y_translation)*len(x_translation))):
                x_tran = x_translation[tran % len(x_translation)]
                y_tran = y_translation[tran // len(x_translation)]

                # obtain the list of labels for this extracted image
                # for yolo's convention, bounding boxes labels have (id, center_x, center_y, width, height)
                cut_out_labels = []

                # compute the center of the current key object
                center_x = (cur_bounding_boxes[label_index][0] + cur_bounding_boxes[label_index][2]) / 2
                center_y = (cur_bounding_boxes[label_index][1] + cur_bounding_boxes[label_index][3]) / 2

                # the extracted image positions in the original image
                x_min = center_x - image_size[1]//2 - x_tran
                y_min = center_y - image_size[0]//2 - y_tran
                x_max = center_x + image_size[1]//2 - x_tran
                y_max = center_y + image_size[0]//2 - y_tran

                # compute the label
                key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y, \
                    = compute_label(cur_bounding_boxes[label_index], x_min, y_min, image_size)

                # construct the lable list
                cut_out_labels.append([label,
                                        key_bb_min_x,
                                        key_bb_min_y,
                                        key_bb_max_x,
                                        key_bb_max_y])

                # save the class id and its bounding box, i indicates its original index in COCO
                cut_out_labels_path = os.path.join(label_dir, f'{data_type}_{x_tran}_{y_tran}.txt')
                # save the label
                with open(cut_out_labels_path, 'w') as myfile:
                    line = str(int(label)) + ' ' + str(key_bb_min_x) + ' ' + str(key_bb_min_y) + ' ' + str(key_bb_max_x) + ' ' + str(key_bb_max_y)
                    myfile.write(line + '\n')

                # save the image
                # obtain the cut out image (y is row, x is col)
                cut_out_image = cur_image[int(y_min):int(y_max),
                                        int(x_min):int(x_max),
                                        :]
                cut_out_image_path = os.path.join(image_dir, f'{data_type}_{x_tran}_{y_tran}.png')
                # get a RGB copy for saving
                cut_out_image_rgb = cv2.cvtColor(cut_out_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(cut_out_image_path, cut_out_image_rgb)

                # all cut out image x_min and y_min
                all_cut_out_xy.append([x_min, y_min])

                # update the list of saved images
                all_image_paths.append(cut_out_image_path)

            # save all the image paths
            path = os.path.join(test_dir, f'single-test.txt')
            with open(path, 'w') as myfile:
                for img_path in all_image_paths:
                    myfile.write('%s\n' % img_path)

            print(f'\n{len(all_image_paths)} image/label pair have been generated and saved')

            # save the label names
            name_path = os.path.join(test_dir, 'custom.names')
            if not os.path.isfile(name_path):
                with open(name_path, 'w') as name_file:
                    for label in desired_classes:
                        name_file.write('%s\n' % label)

            print(f'\ncustom.names have been generated and saved')

            # save the x and y min values as csv
            pos_path = os.path.join(test_dir, f'image_cutout_pos_{cur_class}.csv')
            np.savetxt(pos_path, all_cut_out_xy, delimiter=',', fmt=['%d', '%d'], header='x_min,y_min', comments='')
            print(f'\nall cut out image position csv has been saved to {pos_path}\n')


    # generating single-test-all-candidates dataset
    elif data_type == 'single-test-all-candidates':
        # each object class has a representing image
        for cur_class in desired_classes:

            cur_class = cur_class.replace(" ", "_")
            print(f'\nCurrent processing object class: {cur_class}')

            # create folder for this current class
            class_dir = os.path.join(output_dir, cur_class)
            os.makedirs(class_dir, exist_ok=True)

            # path to file that includes all the candidates
            all_candidates_path = f'/home/zhuokai/Desktop/nvme1n1p1/Data/processed_coco/small_dataset/shift_equivariance/single_test/all_candidates/{cur_class}_single_test_candidates_val.npz'
            all_image_indices = np.load(all_candidates_path)['image_index']
            all_label_indices = np.load(all_candidates_path)['key_label_index']

            if len(all_image_indices) != len(all_label_indices):
                raise Exception(f'Unmatched number of image indices and label indices')

            for i in tqdm(range(len(all_image_indices))):

                # for each candidate image in this class
                image_index = all_image_indices[i]
                label_index = all_label_indices[i]
                print(f'Image index: {image_index}')
                print(f'Label index: {label_index}')

                # create folder for current image and label
                test_dir = os.path.join(class_dir, f'image_{image_index}_{label_index}')
                os.makedirs(test_dir, exist_ok=True)

                # define and create paths
                image_dir = os.path.join(test_dir, 'images')
                label_dir = os.path.join(test_dir, 'labels')
                os.makedirs(image_dir, exist_ok=True)
                os.makedirs(label_dir, exist_ok=True)

                # current image and its label
                cur_image, cur_label = dataset[image_index]
                # convert image to numpy
                cur_image = cur_image.asnumpy()
                # height and width of the image
                height, width = cur_image.shape[:2]

                if downsample_factor != 1:
                    # convert to PIL image for downsampling
                    cur_image = Image.fromarray(cur_image.asnumpy())
                    # downsample the image by downsample_factor
                    cur_image = cur_image.resize((cur_image.width//downsample_factor, cur_image.height//downsample_factor), Image.LANCZOS)
                    # convert to numpy array
                    cur_image = np.array(cur_image)
                    # update height and width of the image
                    height, width = cur_image.shape[:2]
                    # all the bounding boxes and their correponding class ids of the current single image
                    # because of downsampling, all label values are divided by downsample_factor
                    cur_bounding_boxes = cur_label[:, :4] / downsample_factor
                else:
                    cur_bounding_boxes = cur_label[:, :4]

                cur_class_ids = cur_label[:, 4:5][:, 0].astype(int)

                # the label we are using
                label = cur_class_ids[label_index]

                # all the image paths
                all_image_paths = []
                # all cut out images' x_min and y_min (used for visualization)
                all_cut_out_xy = []

                for tran in range(len(y_translation)*len(x_translation)):
                    x_tran = x_translation[tran % len(x_translation)]
                    y_tran = y_translation[tran // len(x_translation)]

                    # obtain the list of labels for this extracted image
                    # for yolo's convention, bounding boxes labels have (id, center_x, center_y, width, height)
                    cut_out_labels = []

                    # compute the center of the current key object
                    center_x = (cur_bounding_boxes[label_index][0] + cur_bounding_boxes[label_index][2]) / 2
                    center_y = (cur_bounding_boxes[label_index][1] + cur_bounding_boxes[label_index][3]) / 2

                    # the extracted image positions in the original image
                    x_min = center_x - image_size[1]//2 - x_tran
                    y_min = center_y - image_size[0]//2 - y_tran
                    x_max = center_x + image_size[1]//2 - x_tran
                    y_max = center_y + image_size[0]//2 - y_tran

                    # compute the label
                    key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y, \
                        = compute_label(cur_bounding_boxes[label_index], x_min, y_min, image_size)

                    # construct the lable list
                    cut_out_labels.append([label,
                                            key_bb_min_x,
                                            key_bb_min_y,
                                            key_bb_max_x,
                                            key_bb_max_y])

                    # save the class id and its bounding box, i indicates its original index in COCO
                    cut_out_labels_path = os.path.join(label_dir, f'{data_type}_{x_tran}_{y_tran}.txt')
                    # save the label
                    with open(cut_out_labels_path, 'w') as myfile:
                        line = str(int(label)) + ' ' + str(key_bb_min_x) + ' ' + str(key_bb_min_y) + ' ' + str(key_bb_max_x) + ' ' + str(key_bb_max_y)
                        myfile.write(line + '\n')

                    # save the image
                    # obtain the cut out image (y is row, x is col)
                    cut_out_image = cur_image[int(y_min):int(y_max),
                                            int(x_min):int(x_max),
                                            :]
                    cut_out_image_path = os.path.join(image_dir, f'{data_type}_{x_tran}_{y_tran}.png')
                    # get a RGB copy for saving
                    cut_out_image_rgb = cv2.cvtColor(cut_out_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(cut_out_image_path, cut_out_image_rgb)

                    # all cut out image x_min and y_min
                    all_cut_out_xy.append([x_min, y_min])

                    # update the list of saved images
                    all_image_paths.append(cut_out_image_path)

                # save all the image paths
                path = os.path.join(test_dir, f'single-test.txt')
                with open(path, 'w') as myfile:
                    for img_path in all_image_paths:
                        myfile.write('%s\n' % img_path)

                print(f'\n{len(all_image_paths)} image/label pair have been generated and saved')

                # save the label names
                name_path = os.path.join(test_dir, 'custom.names')
                if not os.path.isfile(name_path):
                    with open(name_path, 'w') as name_file:
                        for label in desired_classes:
                            name_file.write('%s\n' % label)

                print(f'\ncustom.names have been generated and saved')

                # save the x and y min values as csv
                pos_path = os.path.join(test_dir, f'image_cutout_pos_{cur_class}.csv')
                np.savetxt(pos_path, all_cut_out_xy, delimiter=',', fmt=['%d', '%d'], header='x_min,y_min', comments='')
                print(f'\nall cut out image position csv has been saved to {pos_path}\n')


if __name__ == '__main__':
    main()