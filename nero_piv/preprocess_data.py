# The file preprocesses Cai's original dataset to add restrictions that demonstrate NERO plots
import os
import sys
import time
import torch
import argparse
import numpy as np
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


    if restriction == 'right':
        print('hh')



if __name__ == "__main__":
    main()
