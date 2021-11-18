# provide custom data augmentation for training MNIST models
import random
import torchvision
import numpy as np

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
        self.left_padding = np.random.randint(low=0, high=self.max_padding+1, size=1, dtype=int)
        self.top_padding = np.random.randint(low=0, high=self.max_padding+1, size=1, dtype=int)
        # the rest of padding is taken care of by right and bottom padding
        self.right_padding = self.target_img_size - self.img_size - self.left_padding
        self.bottom_padding = self.target_img_size - self.img_size - self.top_padding

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
        self.scale = random.choice(self.scale_factors)
        # resize and pad the image equally to image_size
        resize_size = int(np.floor(self.img_size * self.scale))
        # size difference must be odd
        if (self.img_size - resize_size) % 2 == 0:
            resize_size += 1

        image = torchvision.transforms.Resize(resize_size)(image)
        image = torchvision.transforms.Pad(((self.target_img_size-resize_size)//2, (self.target_img_size-resize_size)//2, (self.target_img_size-resize_size)//2+1, (self.target_img_size-resize_size)//2+1), fill=0, padding_mode='constant')(image)

        return image
