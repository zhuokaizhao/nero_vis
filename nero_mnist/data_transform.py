# provide custom data augmentation for training MNIST models
import torchvision
import numpy as np

# perform random jittering during the data augmentation phase
class RandomShift(object):
    def __init__(self, img_size, target_img_size):

        # random.seed(0)
        # np.random.seed(0)

        # maximum padding in one side
        max_padding = int(target_img_size - img_size)

        # random number on the left and top padding amount
        self.left_padding = np.random.randint(low=0, high=max_padding+1, size=1, dtype=int)
        self.top_padding = np.random.randint(low=0, high=max_padding+1, size=1, dtype=int)
        # the rest of padding is taken care of by right and bottom padding
        self.right_padding = target_img_size - img_size - self.left_padding
        self.bottom_padding = target_img_size - img_size - self.top_padding


    def __call__(self, image):

        # do the padding
        image = torchvision.transforms.Pad((self.left_padding, self.top_padding, self.right_padding, self.bottom_padding), fill=0, padding_mode='constant')(image)

        return image
