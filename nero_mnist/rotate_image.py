# the script rotates mnist images
import numpy as np
import torch
import torchvision
from PIL import Image

image_path = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/qt_app/example_data/digit_recognition/MNIST_500/label_4_sample_6924.png'

save_dir = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/qt_app/example_data/digit_recognition/rotated_examples/'

loaded_image = Image.open(image_path).convert('RGB')
img_size = loaded_image.width
loaded_image_pt = torch.from_numpy(np.asarray(loaded_image))
# rearrange image format from (28, 28, 1) to (1, 28, 28)
image = torch.permute(loaded_image_pt, (2, 0, 1))
# enlarge the image before rotation
image = torchvision.transforms.Resize(img_size * 3)(image)

# rotate image
for angle in range(0, 365, 5):
    rotated_image_pt = torchvision.transforms.RandomRotation(
        degrees=(angle, angle),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        expand=False,
    )(image)
    # resize back to original size
    rotated_image_pt = torchvision.transforms.Resize(img_size)(rotated_image_pt)
    # convert to white background and black texts
    rotated_image_pt = 255 - rotated_image_pt

    # save image
    save_path = save_dir + f'label_4_sample_6924_rotated_{angle}.png'
    rotated_image_pil = torchvision.transforms.ToPILImage()(rotated_image_pt)
    rotated_image_pil.save(save_path)
