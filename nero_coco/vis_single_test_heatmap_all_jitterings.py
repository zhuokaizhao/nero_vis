# The script visualizes the single test heatmap from different models (-16 to 16)
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


lite_data = False
iou_shresh = 0.5
output_dir = 'figs/'
plot_quantity = ['confidence', 'iou', 'all_precision', 'all_recall', 'all_F_measure']
# plot_quantity = ['confidence']
data_name = 'coco'
# data_name = 'mnist'
# aug_method = 'rotate'
# aug_method = 'scale'

if data_name == 'coco':
    if lite_data:
        input_dir = 'output/single_test/'
    else:
        input_dir = 'output/single_test/'

    jittering_levels = ['0', '20', '40', '60', '80', '100', 'pt']
    jittering_levels = ['0', '33', '66', '100', 'pt']
    object_classes = ['car', 'bottle', 'cup', 'chair', 'book']
elif data_name == 'mnist':
    input_dir = '/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/detection/output/mnist/shift_equivariance/single_test/normal/'
    jittering_levels = ['0', '20', '40']
    # object_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    object_classes = ['0', '1', '2']


# create an figure with 5*7 subplots
if lite_data:
    file_path = os.path.join(input_dir, f'{object_classes[0]}', f'yolo3_single_test_{jittering_levels[0]}-jittered_normal.npz')
    for p in plot_quantity:
        loaded_dataset = np.load(file_path)[f'all_{p}_sorted_by_iou']
        output_path = os.path.join(output_dir, f'single_test_{data_name}_60_jitterings_cup_lite.png')
        plt.scatter(list(range(122)), loaded_dataset[:, 0], s=8, label=f'{p}')
        plt.legend()
        plt.savefig(output_path)

    print(f'single test lite plot saved to {output_path}')

else:
    fig, axes = plt.subplots(nrows=len(object_classes), ncols=len(jittering_levels), figsize=(3*len(jittering_levels), 3*len(object_classes)))

    # Set the ticks and ticklabels for all axes
    plt.setp(axes, xticks=[0, 63, 127],
                    xticklabels=['-64', '0', '63'],
                    yticks=[0, 63, 127],
                    yticklabels=['-64', '0', '63'])

    for pq in plot_quantity:
        for i, jittering in enumerate(jittering_levels):
            for j, object_class in enumerate(object_classes):
                if jittering == 'pt':
                    file_path = os.path.join(input_dir, 'pt', f'{object_class}', f'faster_rcnn_resnet50_single-test_pt_{iou_shresh}.npz')
                else:
                    file_path = os.path.join(input_dir, 'normal', f'{object_class}', f'faster_rcnn_resnet50_single-test_{jittering}-jittered_normal_{iou_shresh}.npz')
                # print(np.load(file_path).files)
                # exit()
                if pq == 'all_precision' or pq == 'all_recall' or pq == 'all_F_measure':

                    loaded_dataset = np.load(file_path)[pq]

                    if len(jittering_levels) == 1 and len(object_classes) == 1:
                        axes.imshow(loaded_dataset[:, :], cmap=plt.get_cmap('viridis'), vmin=0, vmax=1)
                    elif len(jittering_levels) == 1 and len(object_classes) != 1:
                        axes[j].imshow(loaded_dataset[:, :], cmap=plt.get_cmap('viridis'), vmin=0, vmax=1)
                    elif len(object_classes) == 1 and len(jittering_levels) != 1:
                        axes[i].imshow(loaded_dataset[:, :], cmap=plt.get_cmap('viridis'), vmin=0, vmax=1)
                    else:
                        axes[j, i].imshow(loaded_dataset[:, :], cmap=plt.get_cmap('viridis'), vmin=0, vmax=1)
                else:
                    loaded_dataset = np.load(file_path)[f'all_{pq}_sorted_by_conf']

                    if len(jittering_levels) == 1 and len(object_classes) == 1:
                        axes.imshow(loaded_dataset[:, :, 0], cmap=plt.get_cmap('viridis'), vmin=0, vmax=1)
                    elif len(jittering_levels) == 1 and len(object_classes) != 1:
                        axes[j].imshow(loaded_dataset[:, :, 0], cmap=plt.get_cmap('viridis'), vmin=0, vmax=1)
                    elif len(object_classes) == 1 and len(jittering_levels) != 1:
                        axes[i].imshow(loaded_dataset[:, :, 0], cmap=plt.get_cmap('viridis'), vmin=0, vmax=1)
                    else:
                        axes[j, i].imshow(loaded_dataset[:, :, 0], cmap=plt.get_cmap('viridis'), vmin=0, vmax=1)

        # row titles are the object classes
        if len(jittering_levels) == 1 and len(object_classes) == 1:
            axes.set_ylabel(object_classes[0], rotation=0, size='x-large')
            axes.set_title(jittering_levels[0], rotation=0, size='x-large')
        elif len(jittering_levels) == 1:
            for ax, row in zip(axes, object_classes):
                ax.set_ylabel(row, rotation=0, size='x-large')
        elif len(object_classes) == 1:
            for ax, col in zip(axes, jittering_levels):
                if col == 'pt':
                    ax.set_title('pre-trained', rotation=0, size='x-large')
                else:
                    ax.set_title(col + ' jittering', rotation=0, size='x-large')
        else:
            for ax, row in zip(axes[:, 0], object_classes):
                ax.set_ylabel(row, rotation=0, size='x-large')
            # col titles are the model variants
            for ax, col in zip(axes[0], jittering_levels):
                if col == 'pt':
                    ax.set_title('pre-trained', rotation=0, size='x-large')
                else:
                    ax.set_title(col + ' jittering', rotation=0, size='x-large')

        # output_path = os.path.join(output_dir, f'single_test_{data_name}_all_jitterings.svg')
        fig.suptitle(f'Single-image test plotting {pq}', size='xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        output_path = os.path.join(output_dir, f'single_test_vis_{pq}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=200)

        print(f'Single-image test plotting {pq} has been saved to {output_path}')


