import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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


def plot_interactive_line_polar(non_eqv_results, eqv_results, title, result_path):

    non_eqv_gen_losses = non_eqv_results['all_degree_general_losses']
    non_eqv_gen_accuuracy = non_eqv_results['all_degree_general_accuracy']
    non_eqv_categorical_losses = non_eqv_results['all_degree_categorical_losses']
    non_eqv_categorical_accuuracy = non_eqv_results['all_degree_categorical_accuracy']

    eqv_gen_losses = eqv_results['all_degree_general_losses']
    eqv_gen_accuuracy = eqv_results['all_degree_general_accuracy']
    eqv_categorical_losses = eqv_results['all_degree_categorical_losses']
    eqv_categorical_accuuracy = eqv_results['all_degree_categorical_accuracy']
    angles = list(range(len(non_eqv_gen_losses)))

    fig = go.Figure()
    # non eqv accuracy
    fig.add_trace(go.Scatterpolar(
        r = non_eqv_gen_accuuracy,
        theta = angles,
        customdata=non_eqv_categorical_accuuracy,
        mode = 'lines',
        name = 'Non-eqv accuracy',
        line_color = 'peru'
    ))

    # eqv accuracy
    fig.add_trace(go.Scatterpolar(
        r = eqv_gen_accuuracy,
        theta = angles,
        customdata=eqv_categorical_accuuracy,
        mode = 'lines',
        name = 'Eqv accuracy',
        line_color = 'darkviolet'
    ))

    fig.update_traces(
        hovertemplate=
        "Rotated angle: %{theta:.0f}<br>" +
        "General accuracy: %{r:.1%}<br>" +
        "Digit 0 accuracy: %{customdata[0]:.1%}<br>" +
        "Digit 1 accuracy: %{customdata[1]:.1%}<br>" +
        "Digit 2 accuracy: %{customdata[2]:.1%}<br>" +
        "Digit 3 accuracy: %{customdata[3]:.1%}<br>" +
        "Digit 4 accuracy: %{customdata[4]:.1%}<br>" +
        "Digit 5 accuracy: %{customdata[5]:.1%}<br>" +
        "Digit 6 accuracy: %{customdata[6]:.1%}<br>" +
        "Digit 7 accuracy: %{customdata[7]:.1%}<br>" +
        "Digit 8 accuracy: %{customdata[8]:.1%}<br>" +
        "Digit 9 accuracy: %{customdata[9]:.1%}"
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

