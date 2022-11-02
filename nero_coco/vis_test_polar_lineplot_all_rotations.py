# The script visualizes the test 2 heatmap (-16 to 16)
import os
import plotly
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def plot_interactive_line_polar(all_rotations, object_index, all_results, title, fig_path):

    fig = go.Figure()

    for rot in all_rotations:

        cur_result = all_results[f'{rot}'][:, object_index]
        angles = list(range(len(cur_result)))

        # plot accuracy
        fig.add_trace(go.Scatterpolar(
            r = cur_result,
            theta = angles,
            # customdata=non_eqv_categorical_accuuracy,
            mode = 'lines',
            name = f'{rot}-rotated mAP',
            # line_color = 'peru'
        ))

        fig.update_traces(
            hovertemplate=
            "Rotated angle: %{theta:.0f}<br>" +
            "General accuracy: %{r:.1%}<br>"
            # "Digit 0 accuracy: %{customdata[0]:.1%}<br>" +
            # "Digit 1 accuracy: %{customdata[1]:.1%}<br>" +
            # "Digit 2 accuracy: %{customdata[2]:.1%}<br>" +
            # "Digit 3 accuracy: %{customdata[3]:.1%}<br>" +
            # "Digit 4 accuracy: %{customdata[4]:.1%}<br>" +
            # "Digit 5 accuracy: %{customdata[5]:.1%}<br>" +
            # "Digit 6 accuracy: %{customdata[6]:.1%}<br>" +
            # "Digit 7 accuracy: %{customdata[7]:.1%}<br>" +
            # "Digit 8 accuracy: %{customdata[8]:.1%}<br>" +
            # "Digit 9 accuracy: %{customdata[9]:.1%}"
        )

        fig.update_layout(
            title = title,
            showlegend = True,
            hovermode='x',
            title_x=0.5
        )

    fig.show()
    # fig.write_html(fig_path)
    print(f'\nInteractive polar plot has been saved to {fig_path}')


if __name__ == "__main__":

    input_dir = 'output/rotation_equivariance/test/'
    output_dir = 'figs/rotation_equivariance/test/'

    plot_quantities = ['all_single_class_AP', 'all_single_class_precision', 'all_single_class_recall', 'all_single_class_f1']
    # all_rotations = ['0', '33', '66', '100', 'pt']
    all_rotations = ['0', '33', '66', '100', 'pt']
    object_classes = ['car', 'bottle', 'cup', 'chair', 'book']
    limits = [[0.6, 0.4, 0.5, 0.3, 0.3],
              [0.1, 0.1, 0.1, 0.1, 0.1],
              [1, 1, 1, 1, 1],
              [0.6, 0.4, 0.5, 0.3, 0.3]]

    for k, plot_quantity in enumerate(plot_quantities):

        print(f'\nPlotting {plot_quantity}')

        # create an figure with 5*6 subplots
        fig, axes = plt.subplots(nrows=len(object_classes),
                                 ncols=len(all_rotations),
                                 figsize=(3*len(all_rotations), 3*len(object_classes)),
                                 subplot_kw={'projection': 'polar'})

        # all the rotations
        all_degrees = list(range(0, 365, 5))

        for i, rot in enumerate(all_rotations):
            if rot == 'pt':
                file_path = os.path.join(input_dir, 'pt', f'faster_rcnn_resnet50_test_pt.npz')
            else:
                file_path = os.path.join(input_dir, 'normal', f'faster_rcnn_resnet50_{rot}-rotated_test_normal.npz')

            loaded_dataset = np.load(file_path)[plot_quantity]

            # all object classes
            for j in range(len(object_classes)):
                # to close the ring
                plot_data = list(loaded_dataset[:, j])
                plot_data.append(loaded_dataset[0, j])
                # degrees needs to be converted to rad
                axes[j, i].plot(np.array(all_degrees)/180*np.pi, plot_data)
                axes[j, i].set_rlim(0, limits[k][j])

        # row titles are the object classes
        for ax, row in zip(axes[:, 0], object_classes):
            ax.set_ylabel(row, rotation=0, size='x-large')
        # col titles are the jittering levels
        for ax, col in zip(axes[0], all_rotations):
            if col == 'pt':
                ax.set_title('pre-trained', rotation=0, size='x-large')
            else:
                ax.set_title(col + ' rotated', rotation=0, size='x-large')

        fig.suptitle(f'Aggregated test plotting {plot_quantity}', size='xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        output_path = os.path.join(output_dir, f'aggregated_test_{plot_quantity}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        # plt.show()

        print(f'Aggregated test plotting {plot_quantity} has been saved to {output_path}')


