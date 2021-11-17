import os
import torch
import numpy as np
import torchvision
from PIL import Image
from mlxtend.data import loadlocal_mnist


# MNIST dataset class
class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, mode, transform=None, vis=False):
        assert mode in ['train', 'test']
        data_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/MNIST/original'

        if mode == 'train':
            images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
            labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
        elif mode == 'test':
            images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
            labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')

        self.mode = mode
        self.transform = transform
        self.vis = vis

        self.images, self.labels = loadlocal_mnist(images_path=images_path,
                                                   labels_path=labels_path)

        self.images = self.images.reshape(-1, 28, 28).astype(np.float32)
        self.labels = self.labels.astype(np.int64)
        self.num_samples = len(self.labels)

        # normalization and conversion
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
            if self.vis:
                # Image.fromarray(np.asarray(image).astype(np.uint8)).save('/home/zhuokai/Desktop/UChicago/Research/nero_vis/nero_mnist/figs/rot_eqv_test/rotated.png')
                # Image.fromarray(np.asarray(image).astype(np.uint8)).save('/home/zhuokai/Desktop/UChicago/Research/nero_vis/nero_mnist/figs/shift_eqv_test/shifted.png')
                Image.fromarray(np.asarray(image).astype(np.uint8)).save('/home/zhuokai/Desktop/UChicago/Research/nero_vis/nero_mnist/figs/scale_eqv_test_2/scaled.png')
                exit()
            image = self.to_tensor(np.array(image))
            image = self.normalize(image)


        return image, label

    def __len__(self):
        return len(self.labels)

