# provide custom data augmentation for training MNIST models
import random
import torchvision
import numpy as np
from PIL import Image
import truncated_mvn_sampler
from scipy.stats import truncnorm


# perform random rotation during the data augmentation phase
class RandomRotate(object):
    def __init__(self, img_size, target_img_size):

        # maximum padding in one side
        self.img_size = img_size
        self.target_img_size = target_img_size
        self.max_padding = int(target_img_size - img_size)

        self.sigma = 180
        self.a = (-180 - 0) / self.sigma
        self.b = (180 - 0) / self.sigma


    def __call__(self, image):

        r = np.array(truncnorm.rvs(self.a, self.b, scale=self.sigma, size=1))
        r = np.rint(r).astype(int)

        # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
        image = torchvision.transforms.Resize(self.img_size*3)(image)
        image = torchvision.transforms.RandomRotation(degrees=(r, r), resample=Image.BILINEAR, expand=False)(image)
        image = torchvision.transforms.Resize(self.img_size)(image)
        image = torchvision.transforms.Pad(((self.target_img_size-self.img_size)//2, (self.target_img_size-self.img_size)//2, (self.target_img_size-self.img_size)//2+1, (self.target_img_size-self.img_size)//2+1), fill=0, padding_mode='constant')(image)

        return image


# perform random shift during the data augmentation phase
class RandomShift(object):
    def __init__(self, img_size, target_img_size):

        # random.seed(0)
        # np.random.seed(0)

        # maximum padding in one side
        self.img_size = img_size
        self.target_img_size = target_img_size
        self.max_padding = int(target_img_size - img_size)


    def __call__(self, image):

        # random number on the left and top padding amount
        tmvn = truncated_mvn_sampler.TruncatedMVN(self.mu, self.cov, self.lower_bound, self.upper_bound)
        x, y = tmvn.sample(1)
        self.left_padding = np.rint(x).astype(int)
        self.top_padding = np.rint(y).astype(int)

        # the rest of padding is taken care of by right and bottom padding
        self.right_padding = self.max_padding - self.left_padding
        self.bottom_padding = self.max_padding - self.top_padding

        # do the padding
        image = torchvision.transforms.Pad((self.left_padding, self.top_padding, self.right_padding, self.bottom_padding), fill=0, padding_mode='constant')(image)

        return image


# perform random scale during the data augmentation phase
class RandomScale(object):
    def __init__(self, scale_factors, img_size, target_img_size):

        # random.seed(0)
        # np.random.seed(0)
        self.scale_factors = scale_factors
        self.img_size = img_size
        self.target_img_size = target_img_size


    def __call__(self, image):

        # randomly pick a scale factor
        self.scale = 0
        while self.scale <= 0.3 or self.scale > 1:
            self.scale = np.random.normal(1.0, 0.35)
        # resize and pad the image equally to image_size
        resize_size = int(np.floor(self.img_size * self.scale))
        # size difference must be odd
        if (self.img_size - resize_size) % 2 == 0:
            resize_size += 1

        image = torchvision.transforms.Resize(resize_size)(image)
        image = torchvision.transforms.Pad(((self.target_img_size-resize_size)//2, (self.target_img_size-resize_size)//2, (self.target_img_size-resize_size)//2+1, (self.target_img_size-resize_size)//2+1), fill=0, padding_mode='constant')(image)

        return image
