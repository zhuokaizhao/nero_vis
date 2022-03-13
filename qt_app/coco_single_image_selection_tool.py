# the scripts uses data-driven way to find a list of images for single-test case

import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from gluoncv import data, utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


# a few parameters
data_type = 'val'
start_index = 0
purpose = 'shift_equivariance'

# parameters common for all purposes
model_type = 'normal'
target_type = 'single-test'
downsample_factor = 1
image_size = (128, 128)
# coco input dir
input_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco'

# parameters different for each purpose
if purpose == 'shift_equivariance':
    desired_classes = ['car', 'bottle', 'cup', 'chair', 'book']
    x_translation = [-image_size[0], image_size[0]-1]
    y_translation = [-image_size[1], image_size[1]-1]


# output directories
candidate_dir = f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/qt_app/example_data/object_detection/single/all_candidates/'
os.makedirs(candidate_dir, exist_ok=True)

# dir for original images
candidate_original_image_dir = os.path.join(candidate_dir, 'original_images')
os.makedirs(candidate_original_image_dir, exist_ok=True)

# load the dataset
dataset = data.COCODetection(root=input_dir, splits=[f'instances_{data_type}2017'], skip_empty=False)
class_names = np.array(dataset.classes)

# analyzing each image in the dataset
selected_images = {}
for cur_class in desired_classes:
    selected_images[cur_class] = {'image_index': [], 'label_index': [], 'bb_area': []}

for i in tqdm(range(start_index, len(dataset))):

    # current image and its label
    cur_image, cur_label = dataset[i]
    if downsample_factor != 1:
        # convert to PIL image for downsampling
        cur_image = Image.fromarray(cur_image.asnumpy())
        # downsample the image by downsample_factor
        cur_image = cur_image.resize((cur_image.width//downsample_factor, cur_image.height//downsample_factor), Image.LANCZOS)
        # convert to numpy array
        cur_image = np.array(cur_image)
        # all the bounding boxes and their correponding class ids of the current single image
        # because of downsampling, all label values are divided by downsample_factor
        cur_bounding_boxes = cur_label[:, :4] / downsample_factor
    else:
        cur_bounding_boxes = cur_label[:, :4]

    cur_class_ids = cur_label[:, 4:5][:, 0].astype(int)

    # height and width of the image
    height, width = cur_image.shape[:2]

    # all the objects in the current image
    for m, k in enumerate(cur_class_ids):
        # find objects with desired labels
        if class_names[k] in desired_classes:
            # convert id k to custom list index
            label_custom = desired_classes.index(class_names[k])

            # current bounding box
            cur_bb = cur_bounding_boxes[m]

            # compute the center of the current key object
            center_x = (cur_bounding_boxes[m][0] + cur_bounding_boxes[m][2]) / 2
            center_y = (cur_bounding_boxes[m][1] + cur_bounding_boxes[m][3]) / 2

            # determine if the bounding box is too large
            if (cur_bounding_boxes[m][2]-cur_bounding_boxes[m][0])*(cur_bounding_boxes[m][3]-cur_bounding_boxes[m][1]) >= 0.5*image_size[0]*image_size[1]:
                continue

            # test if this target object can serve for all translations (testing extremes only is sufficient)
            for j in range(len(x_translation)*len(y_translation)):
                x_tran = x_translation[j % len(y_translation)]
                y_tran = y_translation[j // len(y_translation)]

                # the extracted image positions in the original image
                x_min = center_x - image_size[1]//2 - x_tran
                y_min = center_y - image_size[0]//2 - y_tran
                x_max = center_x + image_size[1]//2 - x_tran
                y_max = center_y + image_size[0]//2 - y_tran

                # make sure that the extracted image fits in the original image
                if x_min < 0 or x_max >= width or y_min < 0 or y_max >= height:
                    break

                # compute the label
                key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y \
                    = compute_label(cur_bb, x_min, y_min, image_size)

                bb_area = (key_bb_max_x-key_bb_min_x)*(key_bb_max_y-key_bb_min_y)

                # reject the candidates whose bounding box is too large for the extracted image
                if bb_area > 0.5*image_size[0]*image_size[1]:
                    continue
                # reject the candidates whose bounding box is too small
                elif bb_area < 0.01*image_size[0]*image_size[1]:
                    continue

                # this object can serve for all translation
                if j == len(x_translation)*len(y_translation)-1:
                    selected_images[desired_classes[label_custom]]['image_index'].append(i)
                    selected_images[desired_classes[label_custom]]['label_index'].append(m)
                    selected_images[desired_classes[label_custom]]['bb_area'].append(bb_area)


# save these images for hand picking later
# for each object class, rank all the qulified results based on bb area
for cur_class in desired_classes:
    print(f"\n{len(selected_images[cur_class]['image_index'])} potential {cur_class} images have been found")

    cur_class_image_indices = selected_images[cur_class]['image_index']
    cur_class_label_indices = selected_images[cur_class]['label_index']
    cur_class_bb_areas = selected_images[cur_class]['bb_area']

    # zip three lists together for sorting
    three_lists = zip(cur_class_bb_areas, cur_class_image_indices, cur_class_label_indices)
    three_lists = sorted(three_lists)

    cur_class_bb_areas_sorted = [x for x, _, _ in three_lists]
    cur_class_image_indices_sorted = [x for _, x, _ in three_lists]
    cur_class_label_indices_sorted = [x for _, _, x in three_lists]

    # save the largest for current class
    for i in range(len(cur_class_image_indices_sorted)):
        print(f'Saving {i}th of the best {cur_class} image')
        cur_image_index = cur_class_image_indices_sorted[i]
        cur_label_index = cur_class_label_indices_sorted[i]
        cur_image, cur_label = dataset[cur_image_index]
        cur_object_id = desired_classes.index(class_names[int(cur_label[cur_label_index, -1])])
        cur_bounding_box = cur_label[cur_label_index, :4]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax = utils.viz.plot_bbox(cur_image,
                                    np.array([cur_bounding_box]),
                                    labels=np.array([cur_object_id]),
                                    class_names=desired_classes,
                                    ax=ax)
        original_image_dir = os.path.join(candidate_original_image_dir, f'from_{data_type}')
        os.makedirs(original_image_dir, exist_ok=True)
        cur_image_path = os.path.join(original_image_dir, f'{target_type}_candidate_{data_type}_{cur_class}_{cur_image_index}_{cur_label_index}.jpg')
        plt.savefig(cur_image_path)
        plt.close('all')

    # save selection result as npz
    selected_path = os.path.join(candidate_dir, f'{cur_class}_single_test_candidates_{data_type}.npz')
    np.savez(selected_path,
                image_index=cur_class_image_indices_sorted,
                key_label_index=cur_class_label_indices_sorted)
