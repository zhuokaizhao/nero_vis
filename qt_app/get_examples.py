# the script extracts example data
import argparse


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Get data examples for NERO demo application')
    # type of data (mnist)
    parser.add_argument('--type', required=True, action='store', nargs=1, dest='type')
    # input data directory
    parser.add_argument('-i', '--input_dir', action='store', nargs=1, dest='input_dir')
    # output figs directory
    parser.add_argument('-o', '--output_dir', action='store', nargs=1, dest='output_dir')
    # if visualizing data
    parser.add_argument('--vis', action='store_true', dest='vis', default=False)
    # verbosity
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)

    args = parser.parse_args()

    # input variables
    type = args.type[0]
    # input and output graph directory
    input_dir = args.input_dir[0]
    output_dir = args.output_dir[0]
    vis_data = args.vis
    verbose = args.verbose

    if verbose:
        print(f'\nData type/name: {type}')
        print(f'Input directory: {input_dir}')
        print(f'Output directory: {output_dir}')
        print(f'Visualizing selected data: {vis_data}')
        print(f'Verbosity: {verbose}\n')



data_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/MNIST/original'

if mode == 'train':
    images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
    labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
elif mode == 'test':
    images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')

self.mode = mode
self.transform = transform
self.vis = vis

self.images, self.labels = loadlocal_mnist(images_path=images_path,
                                            labels_path=labels_path)

self.images = self.images.reshape(-1, 28, 28).astype(np.float32)
self.labels = self.labels.astype(np.int64)
self.num_samples = len(self.labels)

# normalization and conversion
self.to_tensor = torchvision.transforms.ToTensor()
self.normalize = torchvision.transforms.Normalize((0.1307,), (0.3081,))