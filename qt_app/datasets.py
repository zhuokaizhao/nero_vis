import os
import torch
import warnings
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

from pointnet_util import farthest_point_sample, pc_normalize

# MNIST dataset for running in aggregate mode
class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels, transform=None, vis=False):

        self.transform = transform
        self.vis = vis
        self.images = images
        self.labels = labels

        self.num_samples = len(self.labels)


    def __getitem__(self, index):

        image, label = self.images[index], self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)


# COCO dataset class
class COCODataset(Dataset):
    def __init__(self, list_path, img_size=128, transform=None):
        # with open(list_path, "r") as file:
        #     self.img_files = file.readlines()
        self.img_files = list_path
        # get label path from image path
        self.label_files = [
            path.replace('images', 'labels').replace('.png', '.npy').replace('.jpg', '.npy')
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
                # boxes = np.loadtxt(label_path).reshape(-1, 5)
                boxes = np.load(label_path)

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
        imgs = torch.stack([self.resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i

        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

    # helper function that resizes the image
    def resize(self, image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image


# ModelNet dataset
class ModelNetDataset(Dataset):
    def __init__(
        self,
        catfile_path,
        point_cloud_paths,
        npoint=1024, uniform=False, normal_channel=False, cache_size=15000,
    ):
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = catfile_path

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel
        self.datapath = point_cloud_paths

        # how many data points to cache in memory
        self.cache_size = cache_size
        # from index to (point_set, cls) tuple
        self.cache = {}

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)
