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


# prepare mnist image tensor
# def prepare_coco_image(image):
#     # rearrange image format from (28, 28, 1) to (1, 28, 28)
#     image = torch.permute(image, (2, 0, 1)).float()
#     image = torchvision.transforms.Normalize((0.1307,), (0.3081,))(image)
#     image = torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant')(image)
#     # permute back
#     image = torch.permute(image, (1, 2, 0))

#     return image
