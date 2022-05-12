# The file preprocesses Cai's original dataset to add restrictions that demonstrate NERO plots
import os
import glob
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
        img2_name_list.extend(glob.glob(os.path.join(input_dir, cur_dir + '*img2.tif')))
        label_name_list.extend(glob.glob(os.path.join(input_dir, cur_dir + '/*flow.flo')))

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
    restructed_label_name_list = []
    if restriction == 'right':




if __name__ == "__main__":
    main()
