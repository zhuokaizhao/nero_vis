# The file preprocesses Cai's original dataset to add restrictions that demonstrate NERO plots
import os
import glob
import shutil
import argparse
import numpy as np
import flowiz as fz
from PIL import Image



def main():

	# input arguments
    parser = argparse.ArgumentParser()
    # input dataset directory
    parser.add_argument('-i', '--input_dir', action='store', nargs=1, dest='input_dir', required=True)
    # output directory
    parser.add_argument('-o', '--output_dir', action='store', nargs=1, dest='output_dir', required=True)
    # flow restriction (choose between left, top, right, bottom)
    parser.add_argument('-r', '--restriction', action='store', nargs=1, dest='restriction', required=True)
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    args = parser.parse_args()

    input_dir = args.input_dir[0]
    output_dir = args.output_dir[0]
    restriction = args.restriction[0]
    verbose = args.verbose

    if verbose:
        print(f'\nInput data dir: {input_dir}')
        print(f'Output data dir: {output_dir}')
        print(f'Data restriction direction: {restriction}\n')

    # obtain all the data paths
    img1_name_list = []
    img2_name_list = []
    label_name_list = []
    for cur_dir in os.listdir(input_dir):
        img1_name_list.extend(glob.glob(os.path.join(input_dir, cur_dir, '*img1.tif')))
        img2_name_list.extend(glob.glob(os.path.join(input_dir, cur_dir,'*img2.tif')))
        label_name_list.extend(glob.glob(os.path.join(input_dir, cur_dir, '*flow.flo')))

    # sort the list so that they match in order
    img1_name_list.sort()
    img2_name_list.sort()
    label_name_list.sort()

    if len(img1_name_list) != len(img2_name_list):
        raise Exception(f'Number of image 1 names ({len(img1_name_list)}) does not match with the number of image 2 names({len(img2_name_list)})')
    if len(img2_name_list) != len(label_name_list):
        raise Exception(f'Number of image names ({len(img1_name_list)}) does not match with the number of labels({len(label_name_list)})')

    amount = len(label_name_list)
    # load the first sample to determine the image size
    temp = np.asarray(Image.open(img1_name_list[0])) * 1.0 / 255.0
    img_height, img_width = temp.shape

    # load labels to check for later restrictions
    labels = np.zeros((amount, img_height, img_width, 2))
    for i in range(amount):
        labels[i] = fz.read_flow(label_name_list[i])


    # different preprocessing for different restrictions
    restricted_img1_name_list = []
    restricted_img2_name_list = []
    restricted_label_name_list = []

    for i in range(amount):
        cur_label = labels[i]
        qualified = False
        if restriction == 'left-only':
            # compute the average of x velocity
            x_avg = np.mean(cur_label[:, :, 0])

            if x_avg < 0:
                qualified = True

        elif restriction == 'top-only':
            # compute the average of y velocity
            x_avg = np.mean(cur_label[:, :, 1])

            if x_avg > 0:
                qualified = True

        elif restriction == 'right-only':
            # compute the average of y velocity
            x_avg = np.mean(cur_label[:, :, 0])

            if x_avg > 0:
                qualified = True

        elif restriction == 'down-only':
            # compute the average of y velocity
            x_avg = np.mean(cur_label[:, :, 1])

            if x_avg < 0:
                qualified = True

        if qualified:
            restricted_img1_name_list.append(img1_name_list[i])
            restricted_img2_name_list.append(img2_name_list[i])
            restricted_label_name_list.append(label_name_list[i])

    print(f'{len(restricted_img1_name_list)} ({len(restricted_img1_name_list)/len(img1_name_list)*100:.2f}%) samples qualified\n')

    # copy and paste these qualified data to output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(restricted_img1_name_list)):
        cur_img1_name = restricted_img1_name_list[i].split('/')[-1]
        cur_img2_name = restricted_img2_name_list[i].split('/')[-1]
        cur_label_name = restricted_label_name_list[i].split('/')[-1]
        cur_img1_path = os.path.join(output_dir, cur_img1_name)
        cur_img2_path = os.path.join(output_dir, cur_img2_name)
        cur_label_path = os.path.join(output_dir, cur_label_name)
        shutil.copy(restricted_img1_name_list[i], cur_img1_path)
        shutil.copy(restricted_img2_name_list[i], cur_img2_path)
        shutil.copy(restricted_label_name_list[i], cur_label_path)

    print(f'All qualified data has been saved to {output_dir}')







if __name__ == "__main__":
    main()
