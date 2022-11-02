# script that deals with dataset related stuff
import glob
import random
import os
import sys
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms

import data_transform

# parse the content in data configuration file, returns a dict
def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

# load the class names from the class file
def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


# helper function that resizes the image
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class CreateDataset(Dataset):
    def __init__(self, list_path, img_size=256, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        # get label path from image path
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.transform = transform


    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            # image_id = self.img_files.index(self.img_files[index])
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)

        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        # if self.transform == data_transform.DEFAULT_TRANSFORMS:
        #     img, bb_targets = self.transform((img, boxes))
        # else:
        img, bb_targets = self.transform((label_path, img, boxes))

        # return img_path, image_id, img, bb_targets
        return img_path, img, bb_targets


    def collate_fn(self, batch):

        # Drop invalid images
        batch = [data for data in batch if data is not None]
        paths, imgs, bb_targets = list(zip(*batch))

        # Resize images to input shape (if needed)
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i

        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)