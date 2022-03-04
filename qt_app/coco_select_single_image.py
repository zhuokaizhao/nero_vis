# the script extracts a single image with some preprosessing from COCO dataset

import os
import argparse
import numpy as np
from PIL import Image
from gluoncv import data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Select single images from COCO dataset')
    # input directory
    parser.add_argument('-i', '--input_dir', action='store', nargs=1, dest='input_dir')
    # output figs directory
    parser.add_argument('-o', '--output_dir', action='store', nargs=1, dest='output_dir')
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)

    args = parser.parse_args()

    # input variables
    if args.input_dir:
        input_dir = args.input_dir[0]
    else:
        input_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/'
    if args.output_dir:
        output_dir = args.output_dir[0]
    else:
        output_dir = './example_data/object_detection/single/'

    downsample_factor = 1
    # images are selected through coco_single_image_selection_tool.py
    image_indices = [174, 2623, 1680, 2760, 1144]
    image_labels = ['car', 'bottle', 'cup', 'chair', 'book']
    label_indices = [0, 9, 0, 4, 6]
    dataset = data.COCODetection(root=input_dir, splits=[f'instances_val2017'], skip_empty=True)

    for i, image_index in enumerate(image_indices):

        # current image and its label
        cur_image, cur_label = dataset[image_index]

        if downsample_factor != 1:
            # convert to PIL image for downsampling
            cur_image = Image.fromarray(cur_image.asnumpy())
            # downsample the image by downsample_factor
            cur_image = cur_image.resize((cur_image.width//downsample_factor, cur_image.height//downsample_factor), Image.LANCZOS)
            # convert to numpy array
            cur_image = np.array(cur_image)

            # all the bounding boxes and their correponding class ids of the current single image
            # because of downsampling, all label values are divided by downsample_factor
            cur_bounding_boxes = cur_label[:, :4] / downsample_factor
            cur_class_ids = cur_label[:, 4:5][:, 0].astype(int)
        else:
            cur_image = cur_image.asnumpy()
            cur_bounding_boxes = cur_label[:, :4]
            cur_class_ids = cur_label[:, 4:5][:, 0].astype(int)

        # save image and label
        cur_image = Image.fromarray(cur_image)
        cur_image_path = os.path.join(output_dir, f'{image_labels[i]}_{image_indices[i]}_{label_indices[i]}.png')
        cur_image.save(cur_image_path)

        cur_label = np.zeros(5)
        cur_label[:4] = cur_bounding_boxes[label_indices[i]]
        cur_label[4] = cur_class_ids[label_indices[i]]
        cur_label_path = os.path.join(output_dir, f'{image_labels[i]}_{image_indices[i]}_{label_indices[i]}.npy')
        np.save(cur_label_path, cur_label)

        print(f'Chosen original image has been saved to {cur_image_path}')


if __name__ == '__main__':
    main()