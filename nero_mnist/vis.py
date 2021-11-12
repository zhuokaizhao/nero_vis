import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def plot_save_line(all_degree_accuracy, title, result_path):
    plt.plot(all_degree_accuracy, label='Percent of correctness')
    plt.title(title)
    plt.xlabel('Rotation (degree)')
    plt.ylabel('Percent of correctness')
    plt.legend(loc='upper right')

    plt.savefig(result_path, bbox_inches='tight')


def plot_save_circle(all_degree_losses, all_degree_accuracy, title, result_path):
    # make the color proportional to the loss
    # gets the RGBA values from a float
    colors = [cm.jet(loss) for loss in all_degree_losses]

    # create a new figure
    plt.figure()
    ax = plt.gca()
    # plot circles using the RGBA colors
    for i in range(len(all_degree_losses)):
        cur_accuracy = all_degree_accuracy[i]
        cur_color = colors[i]
        circle = plt.Circle((i, cur_accuracy), 1, color=cur_color, fill=True)
        ax.add_patch(circle)

    # make aspect ratio square
    # ax.set_aspect(1.0)

    # plot the scatter plot
    plt.scatter(list(range(len(all_degree_losses))), all_degree_losses,
                s=0,
                c=all_degree_losses,
                cmap='jet',
                facecolors='none')

    # this works because of the scatter
    plt.colorbar()
    plt.clim(0, max(all_degree_losses))
    plt.title(title)
    plt.xlabel('Rotation (degree)')
    plt.ylabel('Percent of correctness')
    plt.legend(loc='upper right')
    plt.savefig(result_path, bbox_inches='tight')


def plot_interactive_line_polar(digits, non_eqv_results, eqv_results, plot_title, result_path):

    # plot for all digits
    # fig = go.Figure()
    subplot_names = []
    for digit in digits:
        subplot_names.append(f'Digit {digit}')

    fig = make_subplots(rows=2, cols=5,
                        subplot_titles=subplot_names,
                        # specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
                        #        [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]]
                               )
    for i, digit in enumerate(digits):

        # subplot position (start with 1)
        row = i // 5 + 1
        col = i % 5 + 1

        non_eqv_accuracy = non_eqv_results['categorical_accuracy_heatmap'][:, int(digit)]
        eqv_accuracy = eqv_results['categorical_accuracy_heatmap'][:, int(digit)]
        angles = list(range(len(non_eqv_accuracy)))

        # non eqv accuracy
        fig.add_trace(go.Scatterpolar(r = non_eqv_accuracy,
                                        theta = angles,
                                        # customdata=non_eqv_categorical_accuuracy,
                                        mode = 'lines',
                                        name = 'Non-Eqv',
                                        type = 'polar',
                                        line_color = 'peru'),
            row = row,
            col = col
        )

        # eqv accuracy
        fig.add_trace(go.Scatterpolar(r = eqv_accuracy,
                                        theta = angles,
                                        # customdata=eqv_categorical_accuuracy,
                                        mode = 'lines',
                                        name = 'Rot-Eqv',
                                        line_color = 'darkviolet'),
            row = row,
            col = col
        )

        fig.update_traces(
            hovertemplate=
            "Rotated angle: %{theta:.0f}<br>" +
            "Accuracy: %{r:.1%}<br>"
        )

        fig.update_annotations(yshift=20)

    # only show one legend
    for i, trace in enumerate(fig['data']):
        if i > 1:
            trace['showlegend'] = False

    fig.update_layout(
        title = plot_title,
        hovermode='x'
    )

    # fig.for_each_annotation(lambda a: a.update(text = subplot_names[a.text]))
    if result_path[-4:] == 'html':
        fig.write_html(result_path)
    else:
        fig.write_image(result_path)

    print(f'\nInteractive polar plot has been saved to {result_path}')


def plot_interactive_heatmap(digits, results, plot_title, result_path):

    # plot for all digits
    # fig = go.Figure()
    subplot_names = []
    for digit in digits:
        subplot_names.append(f'Digit {digit}')

    fig = make_subplots(rows=2, cols=5,
                        subplot_titles=subplot_names,)

    for i, digit in enumerate(digits):

        # subplot position (start with 1)
        row = i // 5 + 1
        col = i % 5 + 1

        accuracy = results['categorical_accuracy_heatmap'][:, :, int(digit)]

        vertical_translation = len(accuracy)
        horizontal_translation = len(accuracy[0])

        # plot accuracy
        fig.add_trace(go.Heatmap(z = accuracy,
                                    x = np.array(list(range(horizontal_translation)))-20,
                                    y = np.array(list(range(vertical_translation)))-20,
                                    customdata=accuracy,
                                    name = 'Non-eqv accuracy',
                                    type='heatmap',
                                    coloraxis='coloraxis1'),
            row = row,
            col = col
        )

        # set x axes on top
        # fig.update_xaxes(side='top')
        # fig.update_xaxes(side='top')
        # reverse both y axes
        fig.update_yaxes(autorange='reversed')
        fig.update_yaxes(autorange='reversed')

        fig.update_layout(
            title = plot_title,
            showlegend = False,
            hovermode='x',
            # title_x=0.5,
            # xaxis=dict(title='x translation'),
            # yaxis=dict(title='y translation'),
            coloraxis=dict(colorscale = 'Viridis')
        )

        fig.update_coloraxes(colorbar=dict(title='Accuracy'))

        fig.update_annotations(yshift=20)

    # only show one legend
    for i, trace in enumerate(fig['data']):
        if i > 1:
            trace['showlegend'] = False

    fig.update_layout(
        title = plot_title,
        hovermode='x',
    )

    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )

    # fig.for_each_annotation(lambda a: a.update(text = subplot_names[a.text]))
    if result_path[-4:] == 'html':
        fig.write_html(result_path)
    else:
        fig.write_image(result_path)

    print(f'\nInteractive heatmap has been saved to {result_path}')


def plot_interactive_polar_heatmap(data_name, non_eqv_results, trans_eqv_results, rot_eqv_results, plot_title, result_path):
    non_eqv_gen_losses = non_eqv_results['general_loss_heatmap']
    non_eqv_gen_accuracy = non_eqv_results['general_accuracy_heatmap']
    non_eqv_categorical_losses = non_eqv_results['categorical_loss_heatmap']
    non_eqv_categorical_accuracy = non_eqv_results['categorical_accuracy_heatmap']

    trans_eqv_gen_losses = trans_eqv_results['general_loss_heatmap']
    trans_eqv_gen_accuracy = trans_eqv_results['general_accuracy_heatmap']
    trans_eqv_categorical_losses = trans_eqv_results['categorical_loss_heatmap']
    trans_eqv_categorical_accuracy = trans_eqv_results['categorical_accuracy_heatmap']

    rot_eqv_gen_losses = rot_eqv_results['general_loss_heatmap']
    rot_eqv_gen_accuracy = rot_eqv_results['general_accuracy_heatmap']
    rot_eqv_categorical_losses = rot_eqv_results['categorical_loss_heatmap']
    rot_eqv_categorical_accuracy = rot_eqv_results['categorical_accuracy_heatmap']

    # the number of rotation angles and translation distance are defined by the number of rows and cols in heatmap
    num_angles, num_trans = non_eqv_gen_losses.shape
    angle_increment = 360//num_angles

    translations, rotations = np.mgrid[0:num_trans:1, 0:360:angle_increment]

    non_eqv_custom_data = np.concatenate((np.expand_dims(non_eqv_gen_accuracy.transpose().ravel(), 1),
                                          non_eqv_categorical_accuracy.reshape((int(num_angles*num_trans), 10))),
                                          axis=1)

    trans_eqv_custom_data = np.concatenate((np.expand_dims(trans_eqv_gen_accuracy.transpose().ravel(), 1),
                                          trans_eqv_categorical_accuracy.reshape((int(num_angles*num_trans), 10))),
                                          axis=1)

    # two plots
    fig = plotly.subplots.make_subplots(rows=1, cols=2,
                                        specs=[[{'type': 'polar'}]*2],
                                        subplot_titles=[
                                                        'Non-equivariant',
                                                        'Trans-equivariant',
                                                        ],)

    # non-equivariant plot
    fig.add_trace(go.Barpolar(r=translations.ravel(),
                                theta=rotations.ravel(),
                                marker={
                                            "colorscale": plotly.colors.sequential.Viridis,
                                            "showscale": True,
                                            "color": non_eqv_gen_accuracy.transpose().ravel(),
                                            "line_color": None,
                                            "line_width": 0,
                                        },
                                customdata=non_eqv_custom_data,
                                name = 'Non-eqv accuracy'),
                    row=1, col=1)

    # equivariant plot
    fig.add_trace(go.Barpolar(r=translations.ravel(),
                                theta=rotations.ravel(),
                                marker={
                                            "colorscale": plotly.colors.sequential.Viridis,
                                            "showscale": False,
                                            "color": trans_eqv_gen_accuracy.transpose().ravel(),
                                            "line_color": None,
                                            "line_width": 0,
                                        },
                                customdata=trans_eqv_custom_data,
                                name = 'Trans-eqv accuracy'),
                    row=1, col=2)

    if data_name == 'mnist' or data_name == 'mnist-rot':
        fig.update_traces(
                            hovertemplate=
                            "Rotated angle: %{theta:.0f}<br>" +
                            "Translation: %{r:.0f}<br>" +
                            "General accuracy: %{customdata[0]:.1%}<br>"
                            "Digit 0 accuracy: %{customdata[1]:.1%}<br>" +
                            "Digit 1 accuracy: %{customdata[2]:.1%}<br>" +
                            "Digit 2 accuracy: %{customdata[3]:.1%}<br>" +
                            "Digit 3 accuracy: %{customdata[4]:.1%}<br>" +
                            "Digit 4 accuracy: %{customdata[5]:.1%}<br>" +
                            "Digit 5 accuracy: %{customdata[6]:.1%}<br>" +
                            "Digit 6 accuracy: %{customdata[7]:.1%}<br>" +
                            "Digit 7 accuracy: %{customdata[8]:.1%}<br>" +
                            "Digit 8 accuracy: %{customdata[9]:.1%}<br>" +
                            "Digit 9 accuracy: %{customdata[10]:.1%}"
                        )
    else:
        fig.update_traces(
            hovertemplate=
                            "General accuracy: %{z:.1%}<br>" +
                            "Airplane accuracy: %{customdata[0]:.1%}<br>" +
                            "Automobile accuracy: %{customdata[1]:.1%}<br>" +
                            "Bird accuracy: %{customdata[2]:.1%}<br>" +
                            "Cat accuracy: %{customdata[3]:.1%}<br>" +
                            "Deer accuracy: %{customdata[4]:.1%}<br>" +
                            "Dog accuracy: %{customdata[5]:.1%}<br>" +
                            "Frog accuracy: %{customdata[6]:.1%}<br>" +
                            "Horse accuracy: %{customdata[7]:.1%}<br>" +
                            "Ship accuracy: %{customdata[8]:.1%}<br>" +
                            "Truck accuracy: %{customdata[9]:.1%}"
                        )

    fig.update_layout(
        title = plot_title,
        showlegend = False,
        hovermode='x',
        title_x=0.5,
        # polar_bargap=0
    )

    # fig.update_coloraxes(colorbar=dict(title='Accuracy'))
    # fig.show()

    fig.write_html(result_path)
    print(f'\nInteractive polar heatmaps has been saved to {result_path}')
