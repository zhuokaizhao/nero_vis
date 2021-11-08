import os
import scipy
import torch
import numpy as np
import torchvision
from PIL import Image
from mlxtend.data import loadlocal_mnist


# MNIST dataset class
class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, mode, transform=True, rotation=None):
        assert mode in ['train', 'test']
        data_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/MNIST/original'

        if mode == 'train':
            images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
            labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
        elif mode == 'test':
            self.rotation = rotation
            images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
            labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')

        self.mode = mode
        self.transform = transform

        self.images, self.labels = loadlocal_mnist(images_path=images_path,
                                                   labels_path=labels_path)

        self.images = self.images.reshape(-1, 28, 28).astype(np.float32)
        self.labels = self.labels.astype(np.int64)
        self.num_samples = len(self.labels)

        # transform includes padding to 29x29, upsample, rotate, downsample and totensor
        self.pad = torchvision.transforms.Pad((0, 0, 1, 1), fill=0)
        self.totensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.mode == 'train':
            if self.transform is True:
                image = self.pad(image)
                image = self.totensor(np.array(image))
                image = self.normalize(image)
        elif self.mode == 'test':
            if self.transform is True:
                image = self.pad(image)
                image = scipy.ndimage.rotate(input=image,
                                             angle=self.rotation,
                                             axes=(-2, -1),
                                             reshape=False,
                                             order=2)
                image = self.totensor(np.array(image))
                image = self.normalize(image)

        return image, label

    def __len__(self):
        return len(self.labels)


class MnistRotDataset(torch.utils.data.Dataset):

    def __init__(self, mode, transform=True, rotation=None):
        assert mode in ['train', 'test']
        self.mode = mode
        if mode == 'train':
            data_path = '/home/zhuokai/Desktop/nvme1n1p1/Data/MNIST/rotated/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat'
        elif mode == 'test':
            self.rotation = rotation
            data_path = '/home/zhuokai/Desktop/nvme1n1p1/Data/MNIST/rotated/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat'

        self.transform = transform

        data = np.loadtxt(data_path, delimiter=' ')

        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
        self.transform = transform

        # transform includes padding to 29x29, upsample, rotate, downsample and totensor
        self.pad = torchvision.transforms.Pad((0, 0, 1, 1), fill=0)
        self.resize1 = torchvision.transforms.Resize(29*3)
        self.rotate = torchvision.transforms.RandomRotation(180, resample=Image.BILINEAR, expand=False)
        self.resize2 = torchvision.transforms.Resize(29)
        self.totensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.mode == 'train':
            if self.transform is True:
                image = self.pad(image)
                # image = self.resize1(image)
                # image = self.rotate(image)
                # image = self.resize2(image)
                image = self.totensor(np.array(image))
                image = self.normalize(image)
        elif self.mode == 'test':
            if self.transform is True:
                image = self.pad(image)
                image = scipy.ndimage.rotate(input=image,
                                             angle=self.rotation,
                                             axes=(-2, -1),
                                             reshape=False,
                                             order=2)
                image = self.totensor(np.array(image))
                image = self.normalize(image)

        return image, label

    def __len__(self):
        return len(self.labels)
