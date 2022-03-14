# the script extracts example data
import os
import cv2
import torch
import random
import argparse
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from shutil import copy
from gluoncv import data
from pycocotools.coco import COCO
# from mlxtend.data import loadlocal_mnist

random.seed(10)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Get data examples for NERO demo application')
    # name of data (mnist, coco, etc)
    parser.add_argument('--name', required=True, action='store', nargs=1, dest='name')
    # aggregate or single data
    parser.add_argument('--mode', required=True, action='store', nargs=1, dest='mode')
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
    mode = args.mode[0]
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
        image_size = (128, 128)
        output_dir = args.output_dir[0]
        # input variables
        if args.input_dir:
            input_dir = args.input_dir[0]
        else:
            input_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/'

        if verbose:
            print(f'\nData name: {name}')
            print(f'Mode: {mode}')
            print(f'Number of samples: {num_samples}')
            print(f'Input directory: {input_dir}')
            print(f'Output directory: {output_dir}')
            print(f'Visualizing selected data: {vis}')
            print(f'Verbosity: {verbose}\n')

        # folder dir that contains all the original images
        images_folder_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/val2017'
        json_path = '/home/zhuokai/Desktop/nvme1n1p1/Data/coco/annotations/instances_val2017.json'
        dataset = data.COCODetection(root=input_dir, splits=[f'instances_val2017'], skip_empty=False)

        # all the class names from the original COCO
        class_names = np.array(dataset.classes)

        # load json for getting image/label paths
        all_image_names = []
        _coco = COCO(json_path)
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(input_dir, dirname, filename)
            all_image_names.append(abs_path)

        if mode == 'single':

            downsample_factor = 1
            # images are selected through coco_single_image_selection_tool.py
            # these indices are empty-included dataset indices
            image_indices = [1253, 2646, 1692, 2787, 1154]
            image_labels = ['car', 'bottle', 'cup', 'chair', 'book']
            label_indices = [0, 9, 0, 4, 6]


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
                # create images and labels folder inside the (current) test_dir
                image_dir = os.path.join(output_dir, 'images')
                label_dir = os.path.join(output_dir, 'labels')
                # create image and label dirs
                os.makedirs(image_dir, exist_ok=True)
                os.makedirs(label_dir, exist_ok=True)
                # copy the original image over if no scale is present
                if downsample_factor == 1:
                    cur_image_name = all_image_names[image_indices[i]]
                    img_extension = cur_image_name.split('.')[-1]
                    cur_image_path = os.path.join(image_dir, f'{image_labels[i]}_{image_index}_{label_indices[i]}.' + img_extension)
                    copy(os.path.join(images_folder_dir, cur_image_name), cur_image_path)
                else:
                    cur_image = Image.fromarray(cur_image)
                    cur_image_path = os.path.join(image_dir, f'{image_labels[i]}_{image_index}_{label_indices[i]}.png')
                    cur_image.save(cur_image_path)

                cur_label = np.zeros((1, 5))
                cur_label[0, :4] = cur_bounding_boxes[label_indices[i]]
                cur_label[0, 4] = cur_class_ids[label_indices[i]]
                cur_label_path = os.path.join(label_dir, f'{image_labels[i]}_{image_index}_{label_indices[i]}.npy')
                np.save(cur_label_path, cur_label)

                print(f'Chosen original image has been saved to {cur_image_path}')

                print(f'{mode} mode: {len(image_indices)} {name} images have been selected and saved')

        elif mode == 'aggregate':
            desired_classes = ['car', 'bottle', 'cup', 'chair', 'book']
            # downsample factor when determining if an image is qualified
            downsample_factor = 1

            # maximum translation range used to check to make sure current object supports all possible translation
            x_translation = [-64, 64]
            y_translation = [-64, 64]
            # all the class names from the original COCO
            class_names = np.array(dataset.classes)

            # load json for getting image/label paths
            all_image_names = []
            _coco = COCO(json_path)
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                dirname, filename = entry['coco_url'].split('/')[-2:]
                abs_path = os.path.join(input_dir, dirname, filename)
                all_image_names.append(abs_path)

            # generating dataset that is used for all levels of jittering training
            # extracted image count
            num_extracted = 0

            # create images and labels folder inside the (current) test_dir
            image_dir = os.path.join(output_dir, 'images')
            label_dir = os.path.join(output_dir, 'labels')
            # create image and label dirs
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            # go through all images in dataset
            for i in tqdm(range(len(dataset))):

                # stop if all done
                if num_extracted >= num_samples:
                    break

                # current image and its label
                cur_image, cur_label = dataset[i]
                # convert image to numpy
                cur_image = cur_image.asnumpy()
                # height and width of the image
                height, width = cur_image.shape[:2]

                # randomly permutate the labels
                # cur_label_randomized = np.random.permutation(cur_label)
                # all the bounding boxes and their correponding class ids of the current single image
                cur_bounding_boxes = cur_label[:, :4]
                cur_class_ids = cur_label[:, 4:5][:, 0].astype(int)

                if downsample_factor != 1:
                    # convert to PIL image for up/downsampling
                    cur_image_scaled = Image.fromarray(cur_image)
                    # resize the image by downsample_factor
                    cur_image_scaled = cur_image_scaled.resize((width//downsample_factor, height//downsample_factor), Image.LANCZOS)
                    # convert to numpy array
                    cur_image_scaled = np.array(cur_image_scaled)
                    # scale the bounding boxes accordingly
                    cur_bounding_boxes_scaled = cur_bounding_boxes // downsample_factor
                    # height and width are scaled too
                    height, width = cur_image_scaled.shape[:2]
                else:
                    cur_image_scaled = cur_image
                    cur_bounding_boxes_scaled = cur_bounding_boxes

                # go through all the labels in this current image
                for m, k in enumerate(cur_class_ids):
                    # stop if all done
                    if num_extracted >= num_samples:
                        break

                    # find objects within desired labels
                    if class_names[k] in desired_classes:

                        # current bounding box
                        cur_bb = cur_bounding_boxes_scaled[m]

                        # compute the center of the current key object
                        key_bb_min_x = cur_bb[0]
                        key_bb_min_y = cur_bb[1]
                        key_bb_max_x = cur_bb[2]
                        key_bb_max_y = cur_bb[3]
                        center_x = (key_bb_min_x + key_bb_max_x) / 2
                        center_y = (key_bb_min_y + key_bb_max_y) / 2

                        # get the fov locations wrt original image
                        not_qualified = False
                        for x_tran in x_translation:
                            for y_tran in y_translation:
                                x_min = center_x - image_size[1]//2 - x_tran
                                y_min = center_y - image_size[0]//2 - y_tran
                                x_max = center_x + image_size[1]//2 - x_tran
                                y_max = center_y + image_size[0]//2 - y_tran

                                # make sure that the extracted image fits in the original image
                                if x_min < 0 or x_max >= width or y_min < 0 or y_max >= height:
                                    not_qualified = True
                                    break

                                bb_area = (key_bb_max_x-key_bb_min_x)*(key_bb_max_y-key_bb_min_y)

                                # reject the candidates whose bounding box is too large or too small
                                if bb_area > 0.5*image_size[0]*image_size[1] or bb_area < 0.1*image_size[0]*image_size[1]:
                                    not_qualified = True
                                    break

                            if not_qualified:
                                break

                        if not_qualified:
                            continue

                        # save the scaled label
                        cur_label = np.zeros((1, 5))
                        cur_label[0, :4] = cur_bb
                        # keep the original label and record which bb is the target
                        cur_label[0, 4] = k
                        # i indicates its original index in COCO (for human debug)
                        # m indicates its target bb index (for training loading data)
                        cur_label_path = os.path.join(label_dir, f'{class_names[k]}_{i}_{m}.npy')
                        np.save(cur_label_path, cur_label)

                        # copy the original image over if no scale is present
                        if downsample_factor == 1:
                            cur_image_name = all_image_names[i]
                            img_extension = cur_image_name.split('.')[-1]
                            cut_out_image_path = os.path.join(image_dir, f'{class_names[k]}_{i}_{m}.' + img_extension)
                            copy(os.path.join(images_folder_dir, cur_image_name), cut_out_image_path)
                        else:
                            # save the image
                            cut_out_image_path = os.path.join(image_dir, f'{class_names[k]}_{i}_{m}.png')
                            # get a RGB copy for saving
                            cut_out_image_rgb = cv2.cvtColor(cur_image_scaled, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(cut_out_image_path, cut_out_image_rgb)

                        # update the number of extracted image
                        num_extracted += 1

                        # only take one image
                        break

            # save the label names
            # name_path = os.path.join(output_dir, 'custom.names')
            # if not os.path.isfile(name_path):
            #     with open(name_path, 'w') as name_file:
            #         for label in desired_classes:
            #             name_file.write('%s\n' % label)

            #     print(f'\ncustom.names have been generated and saved')

            print(f'{mode} mode: {num_extracted} {name} images have been selected and saved')





if __name__ == '__main__':
    main()