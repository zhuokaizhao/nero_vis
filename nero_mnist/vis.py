import plotly
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


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


def plot_interactive_line_polar(digit, non_eqv_results, eqv_results, title, result_path):

    # non_eqv_gen_losses = non_eqv_results['all_degree_general_losses']
    # non_eqv_gen_accuracy = non_eqv_results['all_degree_general_accuracy']
    # non_eqv_categorical_losses = non_eqv_results['all_degree_categorical_losses']
    # non_eqv_categorical_accuuracy = non_eqv_results['all_degree_categorical_accuracy']

    # eqv_gen_losses = eqv_results['all_degree_general_losses']
    # eqv_gen_accuracy = eqv_results['all_degree_general_accuracy']
    # eqv_categorical_losses = eqv_results['all_degree_categorical_losses']
    # eqv_categorical_accuuracy = eqv_results['all_degree_categorical_accuracy']
    non_eqv_gen_accuracy = non_eqv_results['categorical_accuracy_heatmap'][:, digit]
    eqv_gen_accuracy = eqv_results['categorical_accuracy_heatmap'][:, digit]
    angles = list(range(len(non_eqv_gen_accuracy)))

    fig = go.Figure()
    # non eqv accuracy
    fig.add_trace(go.Scatterpolar(
        r = non_eqv_gen_accuracy,
        theta = angles,
        # customdata=non_eqv_categorical_accuuracy,
        mode = 'lines',
        name = 'Non-eqv accuracy',
        line_color = 'peru'
    ))

    # eqv accuracy
    fig.add_trace(go.Scatterpolar(
        r = eqv_gen_accuracy,
        theta = angles,
        # customdata=eqv_categorical_accuuracy,
        mode = 'lines',
        name = 'Eqv accuracy',
        line_color = 'darkviolet'
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

    # fig.show()
    fig.write_html(result_path)
    print(f'\nInteractive polar plot has been saved to {result_path}')


def plot_interactive_heatmap(non_eqv_results, eqv_results, plot_title, result_path):

    non_eqv_gen_losses = non_eqv_results['general_loss_heatmap']
    non_eqv_gen_accuuracy = non_eqv_results['general_accuracy_heatmap']
    non_eqv_categorical_losses = non_eqv_results['categorical_loss_heatmap']
    non_eqv_categorical_accuracy = non_eqv_results['categorical_accuracy_heatmap']

    eqv_gen_losses = eqv_results['general_loss_heatmap']
    eqv_gen_accuuracy = eqv_results['general_accuracy_heatmap']
    eqv_categorical_losses = eqv_results['categorical_loss_heatmap']
    eqv_categorical_accuuracy = eqv_results['categorical_accuracy_heatmap']

    vertical_translation = len(non_eqv_gen_accuuracy)
    horizontal_translation = len(non_eqv_gen_accuuracy[0])

    fig = plotly.tools.make_subplots(rows=1, cols=2, horizontal_spacing=0.05)
    # non eqv accuracy
    fig.add_trace(
        go.Heatmap(z = non_eqv_gen_accuuracy,
                    x = list(range(horizontal_translation)),
                    y = list(range(vertical_translation)),
                    customdata=non_eqv_categorical_accuracy,
                    name = 'Non-eqv accuracy',
                    type='heatmap',
                    coloraxis='coloraxis1'),
        row=1, col=1)

    # eqv accuracy
    fig.add_trace(
        go.Heatmap(z = eqv_gen_accuuracy,
                    x = list(range(horizontal_translation)),
                    y = list(range(vertical_translation)),
                    customdata=non_eqv_categorical_accuracy,
                    name = 'Eqv accuracy',
                    type='heatmap',
                    coloraxis='coloraxis1'),
        row=1, col=2)

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

    # set x axes on top
    fig.update_xaxes(side='top', row=1, col=1)
    fig.update_xaxes(side='top', row=1, col=2)
    # reverse both y axes
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=2)

    fig.update_layout(
        title = plot_title,
        showlegend = True,
        hovermode='x',
        title_x=0.5,
        # xaxis=dict(title='x translation'),
        # yaxis=dict(title='y translation'),
        coloraxis=dict(colorscale = 'Viridis')
    )

    fig.update_coloraxes(colorbar=dict(title='Accuracy'))

    # fig.show()
    fig.write_html(result_path)
    print(f'\nInteractive heatmaps has been saved to {result_path}')


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
