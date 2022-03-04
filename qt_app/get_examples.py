# the script extracts example data
from genericpath import exists
import os
import torch
import random
import argparse
import torchvision
import numpy as np
from PIL import Image
from gluoncv import data
from mlxtend.data import loadlocal_mnist

random.seed(10)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Get data examples for NERO demo application')
    # name of data (mnist, coco, etc)
    parser.add_argument('--name', required=True, action='store', nargs=1, dest='name')
    # number of samples
    parser.add_argument('--num', required=True, action='store', nargs=1, dest='num')
    # input data directory
    parser.add_argument('-i', '--input_dir', action='store', nargs=1, dest='input_dir')
    # output figs directory
    parser.add_argument('-o', '--output_dir', required=True, action='store', nargs=1, dest='output_dir')
    # if visualizing data
    parser.add_argument('--vis', action='store_true', dest='vis', default=False)
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)

    args = parser.parse_args()

    # input variables
    name = args.name[0]
    num_samples = int(args.num[0])
    vis = args.vis
    verbose = args.verbose

    # digit recognition case
    if name == 'mnist':
        # input and output graph directory
        if args.input_dir:
            input_dir = args.input_dir[0]
        else:
            input_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/MNIST/original'

        output_dir = args.output_dir[0]

        if verbose:
            print(f'\nData name: {name}')
            print(f'Number of samples: {num_samples}')
            print(f'Input directory: {input_dir}')
            print(f'Output directory: {output_dir}')
            print(f'Visualizing selected data: {vis}')
            print(f'Verbosity: {verbose}\n')

        # select from test images
        images_path = os.path.join(input_dir, 't10k-images-idx3-ubyte')
        labels_path = os.path.join(input_dir, 't10k-labels-idx1-ubyte')

        images, labels = loadlocal_mnist(images_path=images_path,
                                            labels_path=labels_path)

        images = images.reshape(-1, 28, 28).astype(np.float32)
        labels = labels.astype(np.int64)
        data_size = len(labels)

        # normalization and conversion
        # to_tensor = torchvision.transforms.ToTensor()
        # normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))

        # randomly pick 10 indices where each class has one
        sample_indices = []
        existed_classes = []
        while len(sample_indices) < num_samples:
            cur_sample_index = random.sample(range(0, data_size-1), 1)[0]

            if existed_classes.count(labels[cur_sample_index]) < num_samples//10 and cur_sample_index not in sample_indices:
                existed_classes.append(labels[cur_sample_index])
                sample_indices.append(cur_sample_index)

        for index in sample_indices:
            image, label = images[index], labels[index]
            image = np.asarray(image).astype(np.uint8)

            # save image as png
            Image.fromarray(image).save(os.path.join(output_dir, f'label_{label}_sample_{index}.png'))


    elif name == 'coco':
        # input variables
        if args.input_dir:
            input_dir = args.input_dir[0]
        else:
            input_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/'

        output_dir = args.output_dir[0]

        if verbose:
            print(f'\nData name: {name}')
            print(f'Number of samples: {num_samples}')
            print(f'Input directory: {input_dir}')
            print(f'Output directory: {output_dir}')
            print(f'Visualizing selected data: {vis}')
            print(f'Verbosity: {verbose}\n')

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

    print(f'{len(sample_indices)} {name} images have been selected and saved')


if __name__ == '__main__':
    main()