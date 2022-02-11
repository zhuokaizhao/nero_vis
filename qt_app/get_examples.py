# the script extracts example data
from genericpath import exists
import os
import torch
import random
import argparse
import torchvision
import numpy as np
from PIL import Image
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
    parser.add_argument('-i', '--input_dir', required=True, action='store', nargs=1, dest='input_dir')
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
    # input and output graph directory
    input_dir = args.input_dir[0]
    output_dir = args.output_dir[0]
    vis = args.vis
    verbose = args.verbose

    if verbose:
        print(f'\nData name: {name}')
        print(f'Number of samples: {num_samples}')
        print(f'Input directory: {input_dir}')
        print(f'Output directory: {output_dir}')
        print(f'Visualizing selected data: {vis}')
        print(f'Verbosity: {verbose}\n')


    if name == 'mnist':
        # data_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/MNIST/original'
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

        # randomly pick sample indices
        sample_indices = random.sample(range(0, data_size-1), num_samples)

        for index in sample_indices:
            image, label = images[index], labels[index]
            image = np.asarray(image).astype(np.uint8)

            # save image as png
            # image_dir = os.path.join(output_dir, f'{label}')
            # if not os.path.exists(image_dir):
            #     os.mkdir(image_dir)
            Image.fromarray(image).save(os.path.join(output_dir, f'label_{label}_sample_{index}.png'))

    print(f'{num_samples} {name} images have been selected and saved')


if __name__ == '__main__':
    main()