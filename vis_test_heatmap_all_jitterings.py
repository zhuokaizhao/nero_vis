# The script visualizes the test 2 heatmap (-16 to 16)
import os
import numpy as np
from matplotlib import pyplot as plt

input_dir = 'output/test/'
output_dir = 'figs/'

# plot_quantities = ['all_mean_AP', 'all_max_conf', 'all_mean_conf']
plot_quantities = ['all_mean_AP', 'all_mean_precision', 'all_mean_recall', 'all_mean_f1']
# jittering_levels = ['0', '20', '40', '60', '80', '100', 'pt']
jittering_levels = ['0', '33', '66', '100', 'pt']
object_classes = ['car', 'bottle', 'cup', 'chair', 'book']


for plot_quantity in plot_quantities:

    print(f'\nPlotting {plot_quantity}')

    # create an figure with 5*6 subplots
    fig, axes = plt.subplots(nrows=len(object_classes), ncols=len(jittering_levels), figsize=(3*len(jittering_levels), 3*len(object_classes)))

    # Set the ticks and ticklabels for all axes
    positive_trans = [1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 29, 37, 48, 63]
    negative_trans = list(reversed(positive_trans))
    negative_trans = [-x for x in negative_trans]
    x_translation = negative_trans + [0] + positive_trans
    y_translation = negative_trans + [0] + positive_trans
    plt.setp(axes, xticks=[0, len(x_translation)//2, len(x_translation)-1],
                    xticklabels=['-64', '0', '63'],
                    yticks=[0, len(x_translation)//2, len(y_translation)-1],
                    yticklabels=['-64', '0', '63'])

    for i, jittering in enumerate(jittering_levels):
        if jittering == 'pt':
            file_path = os.path.join(input_dir, 'pt', f'faster_rcnn_resnet50_test_pt.npz')
        else:
            file_path = os.path.join(input_dir, 'normal', f'faster_rcnn_resnet50_test_{jittering}-jittered_normal.npz')

        loaded_dataset = np.load(file_path)[plot_quantity]

        # all object classes
        for j in range(len(object_classes)):
            axes[j, i].imshow(loaded_dataset[:, :, j])

    # row titles are the object classes
    for ax, row in zip(axes[:, 0], object_classes):
        ax.set_ylabel(row, rotation=0, size='x-large')
    # col titles are the jittering levels
    for ax, col in zip(axes[0], jittering_levels):
        if col == 'pt':
            ax.set_title('pre-trained', rotation=0, size='x-large')
        else:
            ax.set_title(col + ' jittering', rotation=0, size='x-large')

    fig.suptitle(f'Aggregated test plotting {plot_quantity}', size='xx-large')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    output_path = os.path.join(output_dir, f'aggregated_test_{plot_quantity}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=200)

    print(f'Aggregated test plotting {plot_quantity} has been saved to {output_path}')


