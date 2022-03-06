# the script contains all the transformations applied within nero_app
import torch
import torchvision
from PIL import Image


# rotate mnist image
def rotate_mnist_image(image, angle):

    img_size = len(image)
    # rearrange image format from (28, 28, 1) to (1, 28, 28)
    image = torch.permute(image, (2, 0, 1))
    # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
    image = torchvision.transforms.Resize(img_size*3)(image)
    image = torchvision.transforms.RandomRotation(degrees=(angle, angle), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, expand=False)(image)
    image = torchvision.transforms.Resize(img_size)(image)
    # permute back
    image = torch.permute(image, (1, 2, 0))

    return image

# prepare mnist image tensor
def prepare_mnist_image(image):
    # rearrange image format from (28, 28, 1) to (1, 28, 28)
    image = torch.permute(image, (2, 0, 1)).float()
    image = torchvision.transforms.Normalize((0.1307,), (0.3081,))(image)
    image = torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant')(image)
    # permute back
    image = torch.permute(image, (1, 2, 0))

    return image


# Below for COCO iamges
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


# convert label file from coco original classes to target classes
class ConvertLabel(object):
    def __init__(self, original_classes, desired_classes):

        self.original_classes = original_classes
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
