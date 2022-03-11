import torch
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch.nn.functional as F
from torch.utils.data import Dataset

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
