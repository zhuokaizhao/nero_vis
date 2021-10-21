# the script extracts a single image with some preprosessing from COCO dataset

import os
import numpy as np
from PIL import Image
from gluoncv import data, utils
from matplotlib import pyplot as plt


input_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/'
test_dir = './shift_equivariance_d3_vis/data/images/'

downsample_factor = 1
image_indices = [174, 2623, 1680, 2760, 1144]
image_labels = ['car', 'bottle', 'cup', 'chair', 'book']
label_indices = [0, 9, 0, 4, 6]
plot_label = False
desired_classes = ['car', 'bottle', 'cup', 'chair', 'book']
dataset = data.COCODetection(root=input_dir, splits=[f'instances_val2017'], skip_empty=True)

for i, image_index in enumerate(image_indices):

    label_index = label_indices[i]

    class_names = np.array(dataset.classes)
    # current image and its label
    cur_image, cur_label = dataset[image_index]

    if downsample_factor != 1:
        # convert to PIL image for downsampling
        cur_image = Image.fromarray(cur_image.asnumpy())
        # downsample the image by downsample_factor
        cur_image = cur_image.resize((cur_image.width//downsample_factor, cur_image.height//downsample_factor), Image.LANCZOS)
        # convert to numpy array
        cur_image = np.array(cur_image)
        # height and width of the image
        height, width = cur_image.shape[:2]

        # all the bounding boxes and their correponding class ids of the current single image
        # because of downsampling, all label values are divided by downsample_factor
        cur_bounding_boxes = cur_label[:, :4] / downsample_factor
        cur_class_ids = cur_label[:, 4:5][:, 0].astype(int)
    else:
        cur_image = cur_image.asnumpy()
        cur_bounding_boxes = cur_label[:, :4]
        cur_class_ids = cur_label[:, 4:5][:, 0].astype(int)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if plot_label:
        ax = utils.viz.plot_bbox(cur_image,
                                np.array([cur_bounding_boxes[label_index]]),
                                labels=np.array([desired_classes.index(class_names[cur_class_ids[label_index]])]),
                                class_names=desired_classes,
                                ax=ax)
        cur_image_path = os.path.join(test_dir, f'{image_index}_labeled.jpg')
        plt.savefig(cur_image_path)
    else:
        cur_image = Image.fromarray(cur_image)
        cur_image_path = os.path.join(test_dir, f'{image_labels[i]}.jpg')
        cur_image.save(cur_image_path)

    print(f'Chosen original image has been saved to {cur_image_path}')