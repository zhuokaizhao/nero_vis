# handles data transforms that are used while doing jittering and data augmentations
import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torchvision.transforms as transforms
from scipy.stats import truncnorm
from scipy import ndimage
import random

from gluoncv import data, utils
from matplotlib import pyplot as plt
import os
import pandas as pd

from PIL import Image



# convert both images and bounding boxes to PyTorch tensors
class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        label_path, img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets


class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        label_path, img, boxes = data

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage([BoundingBox(*box[1:], label=box[0]) for box in boxes],
                                              shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(image=img,
                                                 bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x_min, y_min, x_max, y_max)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = x1
            boxes[box_idx, 2] = y1
            boxes[box_idx, 3] = x2
            boxes[box_idx, 4] = y2
        return label_path, img, boxes


class ShiftEqvAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.2)),
            # iaa.Affine(rotate=(-180, 180)),  # rotate by -90 to 90 degrees (affects segmaps)
            iaa.AddToBrightness((-30, 30)),
            iaa.AddToHue((-20, 20)),
            # iaa.Fliplr(0.5),
            # iaa.ScaleX((0.5, 1.5)),
            # iaa.ScaleY((0.5, 1.5))
            # small scale
            # iaa.ScaleX((0.8, 1.2)),
            # iaa.ScaleY((0.8, 1.2))
        ], random_order=True)


# perform random jittering during the data augmentation phase
class GaussianJittering(object):
    def __init__(self, img_size, percentage):

        random.seed(0)
        np.random.seed(0)

        self.img_size = img_size
        self.percentage = percentage

        if percentage == 0:
            self.x_tran = 0
            self.y_tran = 0
        else:
            # with gaussian, percentage is converted to different sigma value
            self.sigma_x = self.img_size*self.percentage/200/2
            self.sigma_y = self.img_size*self.percentage/200/2
            # distribution that has the support as the image boundary
            self.a = (-64 - 0) / self.sigma_x
            self.b = (63 - 0) / self.sigma_y

    # helper function that computes labels for cut out images
    def compute_label(self, cur_bounding_box, x_min, y_min, image_size):
        # convert key object bounding box to be based on extracted image
        bb_x_min = cur_bounding_box[0] - x_min
        bb_y_min = cur_bounding_box[1] - y_min
        bb_x_max = cur_bounding_box[2] - x_min
        bb_y_max = cur_bounding_box[3] - y_min

        # compute the center of the object in the extracted image
        object_center_x = (bb_x_min + bb_x_max) / 2
        object_center_y = (bb_y_min + bb_y_max) / 2

        # compute the width and height of the real bounding box of this object
        original_bb_width = cur_bounding_box[2] - cur_bounding_box[0]
        original_bb_height = cur_bounding_box[3] - cur_bounding_box[1]

        # compute the range of the bounding box, do the clamping if go out of extracted image
        bb_min_x = max(0, object_center_x - original_bb_width/2)
        bb_max_x = min(image_size-1, object_center_x + original_bb_width/2)
        bb_min_y = max(0, object_center_y - original_bb_height/2)
        bb_max_y = min(image_size-1, object_center_y + original_bb_height/2)

        return bb_min_x, bb_min_y, bb_max_x, bb_max_y

    # helper function that extracts the FOV from original image, padding if needed
    def extract_img_with_pad(self, original_img, extracted_img_size, x_min, y_min, x_max, y_max):
        x_min = round(x_min)
        y_min = round(y_min)
        x_max = round(x_max)
        y_max = round(y_max)
        height, width = original_img.shape[:2]
        extracted_img = np.zeros((extracted_img_size, extracted_img_size, 3), dtype=np.uint8)

        # completed within
        if x_min >= 0 and y_min >= 0 and x_max < width-1 and y_max < height-1:
            extracted_img = original_img[y_min:y_max, x_min:x_max, :]

        # if top left part went out
        elif x_min < 0 and y_min < 0 and x_max < width-1 and y_max < height-1:
            # print('top left went out')
            extracted_img[abs(y_min):, abs(x_min):, :] = original_img[0:y_max, 0:x_max, :]
            # mirror padding
            extracted_img[:abs(y_min), :abs(x_min), :] = original_img[0:abs(y_min), 0:abs(x_min), :]
            extracted_img[abs(y_min):, :abs(x_min), :] = original_img[0:y_max, 0:abs(x_min), :]
            extracted_img[:abs(y_min), abs(x_min):, :] = original_img[0:abs(y_min), 0:x_max, :]

        # if left part went out
        elif x_min < 0 and y_min >= 0 and x_max < width-1 and y_max < height-1:
            # print('left went out')
            extracted_img[:, abs(x_min):, :] = original_img[y_min:y_max, 0:x_max, :]
            # mirror padding
            extracted_img[:, :abs(x_min), :] = original_img[y_min:y_max, 0:abs(x_min), :]

        # if bottom left part went out
        elif x_min < 0 and y_min >= 0 and x_max < width-1 and y_max > height-1:
            # print('bottom left went out')
            extracted_img[0:extracted_img_size-(y_max-height), abs(x_min):, :] = original_img[y_min:, 0:x_max, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, 0:abs(x_min), :] = original_img[2*height-y_max:, 0:abs(x_min), :]
            extracted_img[0:extracted_img_size-(y_max-height), 0:abs(x_min), :] = original_img[y_min:, 0:abs(x_min), :]
            extracted_img[extracted_img_size-(y_max-height):, abs(x_min):, :] = original_img[2*height-y_max:, 0:extracted_img_size-abs(x_min), :]

        # if bottom part went out
        elif x_min >= 0 and y_min >= 0 and x_max < width-1 and y_max > height-1:
            # print('bottom went out')
            extracted_img[0:extracted_img_size-(y_max-height), :, :] = original_img[y_min:, x_min:x_max, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, :, :] = original_img[2*height-y_max:, x_min:x_max, :]

        # if bottom right part went out
        elif x_min >= 0 and y_min >= 0 and x_max > width-1 and y_max > height-1:
            # print('bottom right went out')
            extracted_img[0:extracted_img_size-(y_max-height), 0:extracted_img_size-(x_max-width), :] = original_img[y_min:, x_min:, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, extracted_img_size-(x_max-width):, :] = original_img[height-y_max:, 2*width-x_max:, :]
            extracted_img[extracted_img_size-(y_max-height):, 0:extracted_img_size-(x_max-width), :] = original_img[height-y_max:, x_min:, :]
            extracted_img[0:extracted_img_size-(y_max-height), extracted_img_size-(x_max-width):, :] = original_img[y_min:, 2*width-x_max:, :]

        # if right part went out
        elif x_min >= 0 and y_min >= 0 and x_max > width-1 and y_max < height-1:
            # print('\nright went out')
            extracted_img[:, 0:extracted_img_size-(x_max-width), :] = original_img[y_min:y_max, x_min:, :]
            # mirror padding
            extracted_img[:, extracted_img_size-(x_max-width):, :] = original_img[y_min:y_max, 2*width-x_max:, :]

        # if top right part went out
        elif x_min >= 0 and y_min < 0 and x_max >= width-1 and y_max < height-1:
            # print('top right went out')
            extracted_img[abs(y_min):, 0:extracted_img_size-(x_max-width), :] = original_img[0:y_max, x_min:, :]
            # mirror padding
            extracted_img[0:abs(y_min), extracted_img_size-(x_max-width):, :] = original_img[0:abs(y_min), 2*width-x_max:, :]
            extracted_img[0:abs(y_min), 0:extracted_img_size-(x_max-width), :] = original_img[0:abs(y_min), x_min:, :]
            extracted_img[abs(y_min):, extracted_img_size-(x_max-width):, :] = original_img[0:y_max, 2*width-x_max:, :]

        # if top part went out
        elif x_min >= 0 and y_min < 0 and x_max < width-1 and y_max < height-1:
            # print('top went out')
            extracted_img[abs(y_min):, :, :] = original_img[0:y_max, x_min:x_max, :]
            # mirror padding
            extracted_img[0:abs(y_min), :, :] = original_img[0:abs(y_min), x_min:x_max, :]

        return extracted_img


    def __call__(self, data):
        label_path, img, labels = data

        all_ids = labels[:, 0].astype(int)
        all_bbs = labels[:, 1:]

        # randomly choose a translation for the key object
        if self.percentage == 0:
            x_tran = self.x_tran
            y_tran = self.y_tran
        else:
            # random jittering method 1
            x_tran = float(truncnorm.rvs(self.a, self.b, scale=self.sigma_x, size = 1)[0])
            y_tran = float(truncnorm.rvs(self.a, self.b, scale=self.sigma_y, size = 1)[0])
            self.x_tran = x_tran
            self.y_tran = y_tran

            # random jittering method 2
            # x_tran = random.gauss(0, self.sigma_x)
            # while (-64 <= x_tran < 64) == False:
            #     x_tran = random.gauss(0, self.sigma_x)
            # y_tran = random.gauss(0, self.sigma_y)
            # while (-64 <= y_tran < 64) == False:
            #     y_tran = random.gauss(0, self.sigma_y)

        # save the x_tran and y_tran out
        # all_trans_path = f'/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/detection/logs_zz/shift_equivariance/all_trans_{self.percentage}.npz'
        # # when the file exists, load and append
        # if os.path.isfile(all_trans_path):
        #     loaded_x_trans = np.load(all_trans_path)['x_trans']
        #     loaded_y_trans = np.load(all_trans_path)['y_trans']
        #     all_x_trans = np.append(loaded_x_trans, [x_tran], axis=0)
        #     all_y_trans = np.append(loaded_y_trans, [y_tran], axis=0)
        # # when the file does not exist, save the first one
        # else:
        #     all_x_trans = np.array([x_tran])
        #     all_y_trans = np.array([y_tran])
        # # save the trans
        # np.savez(all_trans_path,
        #         x_trans=all_x_trans,
        #         y_trans=all_y_trans)

        # labels for current image
        processed_labels = []
        # analyze the img path to get the key bb index
        key_label_index = int(label_path.split('_')[-1].split('.')[0])
        # get the target bb
        key_id = int(all_ids[key_label_index])
        key_bb = all_bbs[key_label_index]

        # print(f'Before Image shape {img.shape}')
        # print(f'Before Labels shape {labels.shape}')
        # utils.viz.plot_bbox(img, np.array([key_bb]), scores=None,
        #                     labels=np.array([key_id]), class_names=self.coco_classes)
        # plt.show()

        # center of the target object
        key_bb_min_x = key_bb[0]
        key_bb_min_y = key_bb[1]
        key_bb_max_x = key_bb[2]
        key_bb_max_y = key_bb[3]
        key_center_x = (key_bb_min_x + key_bb_max_x) / 2
        key_center_y = (key_bb_min_y + key_bb_max_y) / 2

        # the cut-out image positions in the original image
        # the move of actual object and the window are opposite, thus minus
        x_min = key_center_x - self.img_size//2 - x_tran
        y_min = key_center_y - self.img_size//2 - y_tran
        x_max = key_center_x + self.img_size//2 - x_tran
        y_max = key_center_y + self.img_size//2 - y_tran

        # compute the bounding box wrt extrated image
        key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y \
            = self.compute_label(key_bb, x_min, y_min, self.img_size)

        processed_labels.append([key_id,
                                key_bb_min_x,
                                key_bb_min_y,
                                key_bb_max_x,
                                key_bb_max_y])

        # check if other objects (surround) are also inside the current extracted image
        # for n, l in enumerate(all_ids):
        #     # if we are on the key object, continue to prevent double labelling
        #     if n != key_label_index and self.coco_classes[l] in self.desired_classes:
        #         # convert id k to custom list index
        #         l_custom = self.desired_classes.index(self.coco_classes[l])

        #         # scale the surrounding objects as well
        #         cur_surround_bb = labels[n, 1:]

        #         # check if the current object center is in the custom designed area
        #         center_x = (cur_surround_bb[0] + cur_surround_bb[2]) / 2
        #         center_y = (cur_surround_bb[1] + cur_surround_bb[3]) / 2
        #         if (center_x > x_min and center_x < x_max-1
        #             and center_y > y_min and center_y < y_max-1):

        #             # create surround label
        #             surround_bb_min_x, surround_bb_min_y, surround_bb_max_x, surround_bb_max_y \
        #                 = self.compute_label(cur_surround_bb, x_min, y_min, self.img_size)

        #             # reject the candidates whose bounding box is too large
        #             if (surround_bb_max_x-surround_bb_min_x)*(surround_bb_max_y-surround_bb_min_y) > 0.5*self.img_size*self.img_size:
        #                 continue

        #             # construct the lable list
        #             processed_labels.append([l_custom,
        #                                     surround_bb_min_x,
        #                                     surround_bb_min_y,
        #                                     surround_bb_max_x,
        #                                     surround_bb_max_y])

        # extract the image, pad if needed
        extracted_img = self.extract_img_with_pad(img, self.img_size, x_min, y_min, x_max, y_max)
        # make the RGB from [0, 255] to [0, 1] - cannot do it here because imgaug does not accept float image
        # extracted_img = extracted_img / 255
        processed_labels = np.array(processed_labels)

        # print(f'Extracted Image shape {extracted_img.shape}')
        # print(f'Extracted Labels shape {processed_labels.shape}')
        # # vis takes [x_min, y_min, x_max, y_max] as bb inputs
        # utils.viz.plot_bbox(extracted_img, processed_labels[:, 1:], scores=None,
        #                     labels=processed_labels[:, 0], class_names=self.desired_classes)
        # plt.show()

        # save all the processed labels in a file
        # processed_labels_path = f'/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/detection/logs_zz/shift_equivariance/pattern_investigation/processed_labels_{self.percentage}-jittered.csv'
        # labels_df = pd.DataFrame(extracted_labels)
        # if os.path.isfile(processed_labels_path):
        #     labels_df.to_csv(processed_labels_path, mode='a', header=False, index=False)
        # else:
        #     labels_df.to_csv(processed_labels_path, header=['class_id', 'center_x', 'center_y', 'width', 'height'], index=False)

        # # vis takes [x_min, y_min, x_max, y_max] as bb inputs
        # print(f'After Image shape {extracted_img.shape}')
        # print(f'After Labels shape {processed_labels.shape}')
        # utils.viz.plot_bbox(extracted_img, all_bb_for_vis, scores=None,
        #                     labels=processed_labels[:, 0], class_names=self.desired_classes)
        # plt.show()

        return label_path, extracted_img, processed_labels


# perform jittering with a specific x_tran and y_tran during the test phase
class FixedJittering(object):
    def __init__(self, img_size, x_tran, y_tran):

        self.img_size = img_size
        self.x_tran = x_tran
        self.y_tran = y_tran

    # helper function that computes labels for cut out images
    def compute_label(self, cur_bounding_box, x_min, y_min, image_size):
        # convert key object bounding box to be based on extracted image
        bb_x_min = cur_bounding_box[0] - x_min
        bb_y_min = cur_bounding_box[1] - y_min
        bb_x_max = cur_bounding_box[2] - x_min
        bb_y_max = cur_bounding_box[3] - y_min

        # compute the center of the object in the extracted image
        object_center_x = (bb_x_min + bb_x_max) / 2
        object_center_y = (bb_y_min + bb_y_max) / 2

        # compute the width and height of the real bounding box of this object
        original_bb_width = cur_bounding_box[2] - cur_bounding_box[0]
        original_bb_height = cur_bounding_box[3] - cur_bounding_box[1]

        # compute the range of the bounding box, do the clamping if go out of extracted image
        bb_min_x = max(0, object_center_x - original_bb_width/2)
        bb_max_x = min(image_size-1, object_center_x + original_bb_width/2)
        bb_min_y = max(0, object_center_y - original_bb_height/2)
        bb_max_y = min(image_size-1, object_center_y + original_bb_height/2)

        return bb_min_x, bb_min_y, bb_max_x, bb_max_y

    # helper function that extracts the FOV from original image, padding if needed
    def extract_img_with_pad(self, original_img, extracted_img_size, x_min, y_min, x_max, y_max):
        x_min = round(x_min)
        y_min = round(y_min)
        x_max = round(x_max)
        y_max = round(y_max)
        height, width = original_img.shape[:2]
        extracted_img = np.zeros((extracted_img_size, extracted_img_size, 3), dtype=np.uint8)

        # completed within
        if x_min >= 0 and y_min >= 0 and x_max < width-1 and y_max < height-1:
            extracted_img = original_img[y_min:y_max, x_min:x_max, :]

        # if top left part went out
        elif x_min < 0 and y_min < 0 and x_max < width-1 and y_max < height-1:
            # print('top left went out')
            extracted_img[abs(y_min):, abs(x_min):, :] = original_img[0:y_max, 0:x_max, :]
            # mirror padding
            extracted_img[:abs(y_min), :abs(x_min), :] = original_img[0:abs(y_min), 0:abs(x_min), :]
            extracted_img[abs(y_min):, :abs(x_min), :] = original_img[0:y_max, 0:abs(x_min), :]
            extracted_img[:abs(y_min), abs(x_min):, :] = original_img[0:abs(y_min), 0:x_max, :]

        # if left part went out
        elif x_min < 0 and y_min >= 0 and x_max < width-1 and y_max < height-1:
            # print('left went out')
            extracted_img[:, abs(x_min):, :] = original_img[y_min:y_max, 0:x_max, :]
            # mirror padding
            extracted_img[:, :abs(x_min), :] = original_img[y_min:y_max, 0:abs(x_min), :]

        # if bottom left part went out
        elif x_min < 0 and y_min >= 0 and x_max < width-1 and y_max > height-1:
            # print('bottom left went out')
            extracted_img[0:extracted_img_size-(y_max-height), abs(x_min):, :] = original_img[y_min:, 0:x_max, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, 0:abs(x_min), :] = original_img[2*height-y_max:, 0:abs(x_min), :]
            extracted_img[0:extracted_img_size-(y_max-height), 0:abs(x_min), :] = original_img[y_min:, 0:abs(x_min), :]
            extracted_img[extracted_img_size-(y_max-height):, abs(x_min):, :] = original_img[2*height-y_max:, 0:extracted_img_size-abs(x_min), :]

        # if bottom part went out
        elif x_min >= 0 and y_min >= 0 and x_max < width-1 and y_max > height-1:
            # print('bottom went out')
            extracted_img[0:extracted_img_size-(y_max-height), :, :] = original_img[y_min:, x_min:x_max, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, :, :] = original_img[2*height-y_max:, x_min:x_max, :]

        # if bottom right part went out
        elif x_min >= 0 and y_min >= 0 and x_max > width-1 and y_max > height-1:
            # print('bottom right went out')
            extracted_img[0:extracted_img_size-(y_max-height), 0:extracted_img_size-(x_max-width), :] = original_img[y_min:, x_min:, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, extracted_img_size-(x_max-width):, :] = original_img[height-y_max:, 2*width-x_max:, :]
            extracted_img[extracted_img_size-(y_max-height):, 0:extracted_img_size-(x_max-width), :] = original_img[height-y_max:, x_min:, :]
            extracted_img[0:extracted_img_size-(y_max-height), extracted_img_size-(x_max-width):, :] = original_img[y_min:, 2*width-x_max:, :]

        # if right part went out
        elif x_min >= 0 and y_min >= 0 and x_max > width-1 and y_max < height-1:
            # print('\nright went out')
            extracted_img[:, 0:extracted_img_size-(x_max-width), :] = original_img[y_min:y_max, x_min:, :]
            # mirror padding
            extracted_img[:, extracted_img_size-(x_max-width):, :] = original_img[y_min:y_max, 2*width-x_max:, :]

        # if top right part went out
        elif x_min >= 0 and y_min < 0 and x_max >= width-1 and y_max < height-1:
            # print('top right went out')
            extracted_img[abs(y_min):, 0:extracted_img_size-(x_max-width), :] = original_img[0:y_max, x_min:, :]
            # mirror padding
            extracted_img[0:abs(y_min), extracted_img_size-(x_max-width):, :] = original_img[0:abs(y_min), 2*width-x_max:, :]
            extracted_img[0:abs(y_min), 0:extracted_img_size-(x_max-width), :] = original_img[0:abs(y_min), x_min:, :]
            extracted_img[abs(y_min):, extracted_img_size-(x_max-width):, :] = original_img[0:y_max, 2*width-x_max:, :]

        # if top part went out
        elif x_min >= 0 and y_min < 0 and x_max < width-1 and y_max < height-1:
            # print('top went out')
            extracted_img[abs(y_min):, :, :] = original_img[0:y_max, x_min:x_max, :]
            # mirror padding
            extracted_img[0:abs(y_min), :, :] = original_img[0:abs(y_min), x_min:x_max, :]

        return extracted_img


    def __call__(self, data):
        label_path, img, labels = data

        all_ids = labels[:, 0].astype(int)
        all_bbs = labels[:, 1:]

        # labels for current image
        processed_labels = []
        # analyze the img path to get the key bb index
        key_label_index = int(label_path.split('_')[-1].split('.')[0])
        # get the target bb
        key_id = int(all_ids[key_label_index])
        key_bb = all_bbs[key_label_index]

        # center of the target object
        key_bb_min_x = key_bb[0]
        key_bb_min_y = key_bb[1]
        key_bb_max_x = key_bb[2]
        key_bb_max_y = key_bb[3]
        key_center_x = (key_bb_min_x + key_bb_max_x) / 2
        key_center_y = (key_bb_min_y + key_bb_max_y) / 2

        # the cut-out image positions in the original image
        # the move of actual object and the window are opposite, thus minus
        x_min = key_center_x - self.img_size//2 - self.x_tran
        y_min = key_center_y - self.img_size//2 - self.y_tran
        x_max = key_center_x + self.img_size//2 - self.x_tran
        y_max = key_center_y + self.img_size//2 - self.y_tran

        # compute the bounding box wrt extrated image
        key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y \
            = self.compute_label(key_bb, x_min, y_min, self.img_size)

        processed_labels.append([key_id,
                                key_bb_min_x,
                                key_bb_min_y,
                                key_bb_max_x,
                                key_bb_max_y])

        # extract the image, pad if needed
        extracted_img = self.extract_img_with_pad(img, self.img_size, x_min, y_min, x_max, y_max)
        processed_labels = np.array(processed_labels)

        return label_path, extracted_img, processed_labels


# convert object classes from coco to custom class labels
class ConvertLabel(object):
    def __init__(self, coco_classes, desired_classes):

        self.coco_classes = coco_classes
        self.desired_classes = desired_classes

    def __call__(self, data):

        label_path, img, labels = data

        # convert key id from COCO classes to custom desired classes
        for i, cur_label in enumerate(labels):
            # 0 is always background for pytorch faster-rcnn
            cur_id = cur_label[0].astype(int)
            key_id_custom = int(self.desired_classes.index(self.coco_classes[cur_id]) + 1)
            labels[i, 0] = key_id_custom

        return label_path, img, labels


class RotEqvAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.2)),
            iaa.AddToBrightness((-30, 30)),
            iaa.AddToHue((-20, 20)),
            # iaa.Fliplr(0.5),
            iaa.ScaleX((0.8, 1.2)),
            iaa.ScaleY((0.8, 1.2))
        ], random_order=True)


# perform random rotation during the data augmentation phase
class GaussianRotation(object):
    def __init__(self, img_size, fixed_rot, percentage=None, rot=None):

        random.seed(0)
        np.random.seed(0)

        self.img_size = img_size
        self.fixed_rot = fixed_rot

        if fixed_rot == True:
            self.rot = rot
        else:
            self.percentage = percentage
            if percentage == 0:
                self.rot = 0
            else:
                # # with gaussian, percentage is converted to different sigma value
                # # rotation is the full 360 degree
                # self.sigma = 360*self.percentage/150/2
                # # distribution that has the support as the rotation limit
                # self.a = (0 - 0) / self.sigma
                # self.b = (360 - 0) / self.sigma
                self.mu = 0
                if percentage == 33:
                    self.kappa = 100/(4*percentage) * np.pi
                elif percentage == 66:
                    self.kappa = 100/(5*percentage) * np.pi
                elif percentage == 100:
                    self.kappa = 100/(8*percentage) * np.pi



    # helper function that rotates the object by its bounding box center (origin)
    # returns the rotated bounding box in the same format (x_min, y_min, x_max, y_max)
    def rotate_object(self, cur_bounding_box, origin, theta):

        # decode inputs
        x_min, y_min, x_max, y_max = cur_bounding_box
        cx, cy = origin
        theta = theta / 180.0 * np.pi

        # assemble four corners (order doesn't matter)
        four_corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
        four_corners_rotated = []

        # rotate four corners in order
        for cur_corner in four_corners:

            # translate points to origin
            x_local = cur_corner[0] - cx
            y_local = cur_corner[1] - cy

            # apply rotation
            # x_rot = x*cos(theta) - y*sin(theta)
            # y_rot = x*sin(theta) + y*cos(theta)
            x_local_rotated = x_local * np.cos(theta) - y_local * np.sin(theta)
            y_local_rotated = x_local * np.sin(theta) + y_local * np.cos(theta)

            four_corners_rotated.append((x_local_rotated, y_local_rotated))

        # extract the min and max so that it matches with the bounding box format
        four_corners_rotated = np.array(four_corners_rotated)
        # get min, max and converts back
        x_min = np.min(four_corners_rotated[:, 0]) + cx
        y_min = np.min(four_corners_rotated[:, 1]) + cy
        x_max = np.max(four_corners_rotated[:, 0]) + cx
        y_max = np.max(four_corners_rotated[:, 1]) + cy

        rotated_bounding_box = [x_min, y_min, x_max, y_max]

        return rotated_bounding_box

    # helper function that computes labels for cut out images
    def compute_label(self, cur_bounding_box, x_min, y_min, image_size):
        # convert key object bounding box to be based on extracted image
        bb_x_min = cur_bounding_box[0] - x_min
        bb_y_min = cur_bounding_box[1] - y_min
        bb_x_max = cur_bounding_box[2] - x_min
        bb_y_max = cur_bounding_box[3] - y_min

        # compute the center of the object in the extracted image
        object_center_x = (bb_x_min + bb_x_max) / 2
        object_center_y = (bb_y_min + bb_y_max) / 2

        # compute the width and height of the real bounding box of this object
        original_bb_width = cur_bounding_box[2] - cur_bounding_box[0]
        original_bb_height = cur_bounding_box[3] - cur_bounding_box[1]

        # compute the range of the bounding box, do the clamping if go out of extracted image
        bb_min_x = max(0, object_center_x - original_bb_width/2)
        bb_max_x = min(image_size-1, object_center_x + original_bb_width/2)
        bb_min_y = max(0, object_center_y - original_bb_height/2)
        bb_max_y = min(image_size-1, object_center_y + original_bb_height/2)

        return bb_min_x, bb_min_y, bb_max_x, bb_max_y

    # helper function that extracts the FOV from original image, padding if needed
    def extract_img_with_pad(self, original_img, extracted_img_size, x_min, y_min, x_max, y_max):
        x_min = int(round(x_min))
        y_min = int(round(y_min))
        x_max = int(round(x_max))
        y_max = int(round(y_max))
        height, width = original_img.shape[:2]
        extracted_img = np.zeros((extracted_img_size, extracted_img_size, 3), dtype=np.uint8)

        # completed within
        if x_min >= 0 and y_min >= 0 and x_max < width-1 and y_max < height-1:
            extracted_img = original_img[y_min:y_max, x_min:x_max, :]

        # if top left part went out
        if x_min < 0 and y_min < 0 and x_max < width-1 and y_max < height-1:
            # print('top left went out')
            extracted_img[abs(y_min):, abs(x_min):, :] = original_img[0:y_max, 0:x_max, :]
            # mirror padding
            extracted_img[:abs(y_min), :abs(x_min), :] = original_img[0:abs(y_min), 0:abs(x_min), :]
            extracted_img[abs(y_min):, :abs(x_min), :] = original_img[0:y_max, 0:abs(x_min), :]
            extracted_img[:abs(y_min), abs(x_min):, :] = original_img[0:abs(y_min), 0:x_max, :]

        # if left part went out
        if x_min < 0 and y_min >= 0 and x_max < width-1 and y_max < height-1:
            # print('left went out')
            extracted_img[:, abs(x_min):, :] = original_img[y_min:y_max, 0:x_max, :]
            # mirror padding
            extracted_img[:, :abs(x_min), :] = original_img[y_min:y_max, 0:abs(x_min), :]

        # if bottom left part went out
        if x_min < 0 and y_min >= 0 and x_max < width-1 and y_max > height-1:
            # print('bottom left went out')
            extracted_img[0:extracted_img_size-(y_max-height), abs(x_min):, :] = original_img[y_min:, 0:x_max, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, 0:abs(x_min), :] = original_img[2*height-y_max:, 0:abs(x_min), :]
            extracted_img[0:extracted_img_size-(y_max-height), 0:abs(x_min), :] = original_img[y_min:, 0:abs(x_min), :]
            extracted_img[extracted_img_size-(y_max-height):, abs(x_min):, :] = original_img[2*height-y_max:, 0:extracted_img_size-abs(x_min), :]

        # if bottom part went out
        if x_min >= 0 and y_min >= 0 and x_max < width-1 and y_max > height-1:
            # print('bottom went out')
            extracted_img[0:extracted_img_size-(y_max-height), :, :] = original_img[y_min:, x_min:x_max, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, :, :] = original_img[2*height-y_max:, x_min:x_max, :]

        # if bottom right part went out
        if x_min >= 0 and y_min >= 0 and x_max > width-1 and y_max > height-1:
            # print('bottom right went out')
            extracted_img[0:extracted_img_size-(y_max-height), 0:extracted_img_size-(x_max-width), :] = original_img[y_min:, x_min:, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, extracted_img_size-(x_max-width):, :] = original_img[height-y_max:, 2*width-x_max:, :]
            extracted_img[extracted_img_size-(y_max-height):, 0:extracted_img_size-(x_max-width), :] = original_img[height-y_max:, x_min:, :]
            extracted_img[0:extracted_img_size-(y_max-height), extracted_img_size-(x_max-width):, :] = original_img[y_min:, 2*width-x_max:, :]

        # if right part went out
        if x_min >= 0 and y_min >= 0 and x_max > width-1 and y_max < height-1:
            # print('\nright went out')
            extracted_img[:, 0:extracted_img_size-(x_max-width), :] = original_img[y_min:y_max, x_min:, :]
            # mirror padding
            extracted_img[:, extracted_img_size-(x_max-width):, :] = original_img[y_min:y_max, 2*width-x_max:, :]

        # if top right part went out
        if x_min >= 0 and y_min < 0 and x_max >= width-1 and y_max < height-1:
            # print('top right went out')
            extracted_img[abs(y_min):, 0:extracted_img_size-(x_max-width), :] = original_img[0:y_max, x_min:, :]
            # mirror padding
            extracted_img[0:abs(y_min), extracted_img_size-(x_max-width):, :] = original_img[0:abs(y_min), 2*width-x_max:, :]
            extracted_img[0:abs(y_min), 0:extracted_img_size-(x_max-width), :] = original_img[0:abs(y_min), x_min:, :]
            extracted_img[abs(y_min):, extracted_img_size-(x_max-width):, :] = original_img[0:y_max, 2*width-x_max:, :]

        # if top part went out
        if x_min >= 0 and y_min < 0 and x_max < width-1 and y_max < height-1:
            # print('top went out')
            extracted_img[abs(y_min):, :, :] = original_img[0:y_max, x_min:x_max, :]
            # mirror padding
            extracted_img[0:abs(y_min), :, :] = original_img[0:abs(y_min), x_min:x_max, :]


        return extracted_img


    def __call__(self, data):
        label_path, img, labels = data

        all_ids = labels[:, 0].astype(int)
        all_bbs = labels[:, 1:]

        # randomly choose a rotation for the key object
        if not self.fixed_rot:
            if self.percentage == 0:
                rot = self.rot
            else:
                # random rotation
                rot = float(np.random.vonmises(self.mu, self.kappa, 1)/np.pi*180)
                self.rot = rot

        # # save the rot stats out
        # all_rots_path = f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/logs/rotation_equivariance/object_{self.percentage}-rotated/all_rots_{self.percentage}.npz'
        # # when the file exists, load and append
        # if os.path.isfile(all_rots_path):
        #     loaded_rots = np.load(all_rots_path)['all_rotations']
        #     all_rots = np.append(loaded_rots, [rot], axis=0)
        # # when the file does not exist, save the first one
        # else:
        #     all_rots = np.array([rot])
        # # save the trans
        # np.savez(all_rots_path,
        #         all_rotations=all_rots)

        # labels for current image
        processed_labels = []
        # analyze the img path to get the key bb index
        key_label_index = int(label_path.split('_')[-1].split('.')[0])
        # get the target bb
        key_id = int(all_ids[key_label_index])
        key_bb = all_bbs[key_label_index]

        # print(f'Original Image shape {img.shape}')
        # print(f'Original Labels shape {labels.shape}')
        # utils.viz.plot_bbox(img, np.array([key_bb]), scores=None,
        #                     labels=np.array([key_id]))
        # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/figs/rotation_equivariance/original_{int(rot)}.png')
        # plt.show()

        # center of the target object
        key_bb_min_x = key_bb[0]
        key_bb_min_y = key_bb[1]
        key_bb_max_x = key_bb[2]
        key_bb_max_y = key_bb[3]
        key_center_x = (key_bb_min_x + key_bb_max_x) / 2
        key_center_y = (key_bb_min_y + key_bb_max_y) / 2

        # make the largest possible image from original in preparation of rotation
        # (scipy.ndimage.rotate only supports rotation by center)
        img_width = img.shape[1]
        img_height = img.shape[0]
        temp_width = min(np.abs(key_center_x-0), np.abs(key_center_x-img_width))
        temp_height = min(np.abs(key_center_y-0), np.abs(key_center_y-img_height))
        temp_img_min_x = int(key_center_x - temp_width)
        temp_img_min_y = int(key_center_y - temp_height)
        temp_img_max_x = int(key_center_x + temp_width)
        temp_img_max_y = int(key_center_y + temp_height)

        temp_img = img[temp_img_min_y:temp_img_max_y, temp_img_min_x:temp_img_max_x]
        temp_bb = [key_bb_min_x-temp_img_min_x, key_bb_min_y-temp_img_min_y,
                    key_bb_max_x-temp_img_min_x, key_bb_max_y-temp_img_min_y]
        # # visualize the temp img
        # utils.viz.plot_bbox(temp_img, np.array([temp_bb]), scores=None,
        #                     labels=np.array([key_id]))
        # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/figs/rotation_equivariance/temp_{int(rot)}.png')
        # plt.show()

        # rotate the image
        temp_image_rotated = ndimage.rotate(temp_img, self.rot, reshape=False)

        # object center wrt temp image
        temp_center_x = (temp_bb[0] + temp_bb[2]) / 2
        temp_center_y = (temp_bb[1] + temp_bb[3]) / 2

        # rotate the labels
        temp_bb_rotated = self.rotate_object(temp_bb, (temp_center_x, temp_center_y), self.rot)

        # # visualize the rotated temp img
        # print(f'Rotated {self.rot} degrees')
        # utils.viz.plot_bbox(temp_image_rotated, np.array([temp_bb_rotated]), scores=None,
        #                     labels=np.array([key_id]))
        # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/figs/rotation_equivariance/rotated_{int(rot)}.png')
        # plt.show()

        # the extracted image locations wrt temp image
        x_min = temp_center_x - self.img_size//2
        y_min = temp_center_y - self.img_size//2
        x_max = temp_center_x + self.img_size//2
        y_max = temp_center_y + self.img_size//2

        # extract the image, pad if needed
        extracted_img = self.extract_img_with_pad(temp_image_rotated, self.img_size, x_min, y_min, x_max, y_max)

        # compute the bounding box wrt extrated image
        key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y \
            = self.compute_label(temp_bb_rotated, x_min, y_min, self.img_size)

        # final_bb = [key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y]
        processed_labels.append([key_id,
                                key_bb_min_x,
                                key_bb_min_y,
                                key_bb_max_x,
                                key_bb_max_y])

        processed_labels = np.array(processed_labels)

        # print(f'Extracted Image shape {extracted_img.shape}')
        # print(f'Extracted Labels shape {processed_labels.shape}')

        # # vis takes [x_min, y_min, x_max, y_max] as bb inputs
        # utils.viz.plot_bbox(extracted_img, processed_labels[:, 1:], scores=None,
        #                     labels=processed_labels[:, 0])
        # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/figs/rotation_equivariance/extracted_{int(rot)}.png')
        # plt.show()

        # save all the processed labels in a file
        # processed_labels_path = f'/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/detection/logs_zz/shift_equivariance/pattern_investigation/processed_labels_{self.percentage}-jittered.csv'
        # labels_df = pd.DataFrame(extracted_labels)
        # if os.path.isfile(processed_labels_path):
        #     labels_df.to_csv(processed_labels_path, mode='a', header=False, index=False)
        # else:
        #     labels_df.to_csv(processed_labels_path, header=['class_id', 'center_x', 'center_y', 'width', 'height'], index=False)

        # # vis takes [x_min, y_min, x_max, y_max] as bb inputs
        # print(f'After Image shape {extracted_img.shape}')
        # print(f'After Labels shape {processed_labels.shape}')
        # utils.viz.plot_bbox(extracted_img, all_bb_for_vis, scores=None,
        #                     labels=processed_labels[:, 0], class_names=self.desired_classes)
        # plt.show()

        return label_path, extracted_img, processed_labels


# performs random rotation and shift during the data augmentation phase
class RandomRotationShift(object):
    def __init__(self,
                 img_size,
                 fixed_rot,
                 fixed_shift,
                 rot_percentage=None,
                 shift_percentage=None,
                 rot=None,
                 x_tran=None,
                 y_tran=None):

        # random.seed(0)
        # np.random.seed(0)

        self.img_size = img_size
        self.fixed_rot = fixed_rot
        self.fixed_shift = fixed_shift

        if fixed_rot == True:
            self.rot = rot
        else:
            self.rot_percentage = rot_percentage
            if self.rot_percentage == 0:
                self.rot = 0
            else:
                # von mises distribution
                self.mu = 0
                if self.rot_percentage == 33:
                    self.kappa = 100/(4*self.rot_percentage) * np.pi
                elif self.rot_percentage == 66:
                    self.kappa = 100/(5*self.rot_percentage) * np.pi
                elif self.rot_percentage == 100:
                    self.kappa = 100/(8*self.rot_percentage) * np.pi

        if fixed_shift == True:
            self.x_tran = x_tran
            self.y_tran = y_tran
        else:
            self.shift_percentage = shift_percentage
            if self.shift_percentage == 0:
                self.x_tran = 0
                self.y_tran = 0
            else:
                # with gaussian, percentage is converted to different sigma value
                self.sigma_x = self.img_size*self.shift_percentage/200/2
                self.sigma_y = self.img_size*self.shift_percentage/200/2
                # distribution that has the support as the image boundary
                self.a = (-64 - 0) / self.sigma_x
                self.b = (63 - 0) / self.sigma_y


    # helper function that rotates the object by its bounding box center (origin)
    # returns the rotated bounding box in the same format (x_min, y_min, x_max, y_max)
    def rotate_object(self, cur_bounding_box, origin, theta):

        # decode inputs
        x_min, y_min, x_max, y_max = cur_bounding_box
        cx, cy = origin
        theta = theta / 180.0 * np.pi

        # assemble four corners (order doesn't matter)
        four_corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
        four_corners_rotated = []

        # rotate four corners in order
        for cur_corner in four_corners:

            # translate points to origin
            x_local = cur_corner[0] - cx
            y_local = cur_corner[1] - cy

            # apply rotation
            # x_rot = x*cos(theta) - y*sin(theta)
            # y_rot = x*sin(theta) + y*cos(theta)
            x_local_rotated = x_local * np.cos(theta) - y_local * np.sin(theta)
            y_local_rotated = x_local * np.sin(theta) + y_local * np.cos(theta)

            four_corners_rotated.append((x_local_rotated, y_local_rotated))

        # extract the min and max so that it matches with the bounding box format
        four_corners_rotated = np.array(four_corners_rotated)
        # get min, max and converts back
        x_min = np.min(four_corners_rotated[:, 0]) + cx
        y_min = np.min(four_corners_rotated[:, 1]) + cy
        x_max = np.max(four_corners_rotated[:, 0]) + cx
        y_max = np.max(four_corners_rotated[:, 1]) + cy

        rotated_bounding_box = [x_min, y_min, x_max, y_max]

        return rotated_bounding_box

    # helper function that computes labels for cut out images
    def compute_label(self, cur_bounding_box, x_min, y_min, image_size):
        # convert key object bounding box to be based on extracted image
        bb_x_min = cur_bounding_box[0] - x_min
        bb_y_min = cur_bounding_box[1] - y_min
        bb_x_max = cur_bounding_box[2] - x_min
        bb_y_max = cur_bounding_box[3] - y_min

        # compute the center of the object in the extracted image
        object_center_x = (bb_x_min + bb_x_max) / 2
        object_center_y = (bb_y_min + bb_y_max) / 2

        # compute the width and height of the real bounding box of this object
        original_bb_width = cur_bounding_box[2] - cur_bounding_box[0]
        original_bb_height = cur_bounding_box[3] - cur_bounding_box[1]

        # compute the range of the bounding box, do the clamping if go out of extracted image
        bb_min_x = max(0, object_center_x - original_bb_width/2)
        bb_max_x = min(image_size-1, object_center_x + original_bb_width/2)
        bb_min_y = max(0, object_center_y - original_bb_height/2)
        bb_max_y = min(image_size-1, object_center_y + original_bb_height/2)

        return bb_min_x, bb_min_y, bb_max_x, bb_max_y

    # helper function that extracts the FOV from original image, padding if needed
    def extract_img(self, original_img, extracted_img_size, x_min, y_min, x_max, y_max):
        x_min = int(round(x_min))
        y_min = int(round(y_min))
        x_max = int(round(x_max))
        y_max = int(round(y_max))
        height, width = original_img.shape[:2]
        extracted_img = np.zeros((extracted_img_size, extracted_img_size, 3), dtype=np.uint8)

        # extract the image
        extracted_img = original_img[y_min:y_max, x_min:x_max, :]

        return extracted_img


    def __call__(self, data):
        label_path, img, labels = data

        all_ids = labels[:, 0].astype(int)
        all_bbs = labels[:, 1:]

        # randomly choose a rotation for the key object
        if not self.fixed_rot:
            if self.rot_percentage == 0:
                rot = self.rot
            else:
                # random rotation
                rot = float(np.random.vonmises(self.mu, self.kappa, 1)/np.pi*180)
                self.rot = rot

        # randomly choose a shift for the key object
        if not self.fixed_shift:
            # randomly choose a translation for the key object
            if self.shift_percentage == 0:
                x_tran = self.x_tran
                y_tran = self.y_tran
            else:
                # random jittering method 1
                x_tran = float(truncnorm.rvs(self.a, self.b, scale=self.sigma_x, size = 1)[0])
                y_tran = float(truncnorm.rvs(self.a, self.b, scale=self.sigma_y, size = 1)[0])
                self.x_tran = x_tran
                self.y_tran = y_tran

        # labels for current image
        processed_labels = []
        # analyze the img path to get the key bb index
        key_label_index = int(label_path.split('_')[-1].split('.')[0])
        # get the target bb
        key_id = int(all_ids[key_label_index])
        key_bb = all_bbs[key_label_index]

        # visualize the original image
        # print(f'Original Image shape {img.shape}')
        # print(f'Original Labels shape {labels.shape}')
        # utils.viz.plot_bbox(img, np.array([key_bb]), scores=None,
        #                     labels=np.array([key_id]))
        # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/figs/rotation_equivariance/original_{int(rot)}.png')
        # plt.show()

        # center of the target object
        key_bb_min_x = key_bb[0]
        key_bb_min_y = key_bb[1]
        key_bb_max_x = key_bb[2]
        key_bb_max_y = key_bb[3]
        key_center_x = (key_bb_min_x + key_bb_max_x) / 2
        key_center_y = (key_bb_min_y + key_bb_max_y) / 2

        # make the largest possible image from original in preparation of rotation
        # (scipy.ndimage.rotate only supports rotation by center)
        img_width = img.shape[1]
        img_height = img.shape[0]
        temp_width = min(np.abs(key_center_x-0), np.abs(key_center_x-img_width))
        temp_height = min(np.abs(key_center_y-0), np.abs(key_center_y-img_height))
        temp_img_min_x = int(key_center_x - temp_width)
        temp_img_min_y = int(key_center_y - temp_height)
        temp_img_max_x = int(key_center_x + temp_width)
        temp_img_max_y = int(key_center_y + temp_height)

        temp_img = img[temp_img_min_y:temp_img_max_y, temp_img_min_x:temp_img_max_x]
        temp_bb = [key_bb_min_x-temp_img_min_x, key_bb_min_y-temp_img_min_y,
                    key_bb_max_x-temp_img_min_x, key_bb_max_y-temp_img_min_y]

        # visualize the temp img
        # utils.viz.plot_bbox(temp_img, np.array([temp_bb]), scores=None,
        #                     labels=np.array([key_id]))
        # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/figs/rotation_equivariance/temp_{int(rot)}.png')
        # plt.show()

        # rotate the image
        # original image shape
        temp_image_rotated = ndimage.rotate(temp_img, self.rot, reshape=True, mode='reflect')

        # object center wrt temp image
        temp_center_x = (temp_bb[0] + temp_bb[2]) / 2
        temp_center_y = (temp_bb[1] + temp_bb[3]) / 2

        # rotate the labels
        temp_bb_rotated = self.rotate_object(temp_bb, (temp_center_x, temp_center_y), self.rot)

        # add the padding offsets
        width_offset = temp_image_rotated.shape[1] - temp_img.shape[1]
        height_offset = temp_image_rotated.shape[0] - temp_img.shape[0]
        temp_bb_rotated = [
            temp_bb_rotated[0] + width_offset/2,
            temp_bb_rotated[1] + height_offset/2,
            temp_bb_rotated[2] + width_offset/2,
            temp_bb_rotated[3] + height_offset/2
        ]

        # visualize the rotated temp img
        # print(f'Rotated {self.rot} degrees')
        # utils.viz.plot_bbox(temp_image_rotated, np.array([temp_bb_rotated]), scores=None,
        #                     labels=np.array([key_id]))
        # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/figs/rotation_equivariance/rotated_{int(rot)}.png')
        # plt.show()

        # the cut-out image positions in the original image
        # the move of actual object and the window are opposite, thus minus
        temp_center_x_rotated = (temp_bb_rotated[0] + temp_bb_rotated[2]) / 2
        temp_center_y_rotated = (temp_bb_rotated[1] + temp_bb_rotated[3]) / 2
        x_min = temp_center_x_rotated - self.img_size//2 - x_tran
        y_min = temp_center_y_rotated - self.img_size//2 - y_tran
        x_max = temp_center_x_rotated + self.img_size//2 - x_tran
        y_max = temp_center_y_rotated + self.img_size//2 - y_tran

        # extract the image, pad if needed
        extracted_img = self.extract_img(temp_image_rotated, self.img_size, x_min, y_min, x_max, y_max)

        # compute the bounding box wrt extrated image
        key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y \
            = self.compute_label(temp_bb_rotated, x_min, y_min, self.img_size)

        # final_bb = [key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y]
        processed_labels.append([key_id,
                                key_bb_min_x,
                                key_bb_min_y,
                                key_bb_max_x,
                                key_bb_max_y])

        processed_labels = np.array(processed_labels)

        # vis takes [x_min, y_min, x_max, y_max] as bb inputs
        # print(f'Shifted (x_tran, y_tran) = ({x_tran}, {y_tran})')
        # utils.viz.plot_bbox(extracted_img, processed_labels[:, 1:], scores=None,
        #                     labels=processed_labels[:, 0])
        # plt.savefig(f'/home/zhuokai/Desktop/UChicago/Research/nero_vis/figs/rotation_equivariance/extracted_{int(rot)}.png')
        # plt.show()

        return label_path, extracted_img, processed_labels
