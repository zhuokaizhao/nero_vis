import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_interactive_line_polar(digits, all_labels, all_colors, all_results, plot_title, result_path):

    # plot for all digits
    subplot_names = []
    for digit in digits:
        subplot_names.append(f'Digit {digit}')

    # 10 digits, divided into 2*5 subplots
    fig = make_subplots(rows=2, cols=5,
                        subplot_titles=subplot_names,
                        specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
                               [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]]
                        )
    for i, digit in enumerate(digits):

        # subplot position (start with 1)
        row = i // 5 + 1
        col = i % 5 + 1

        for j, cur_result in enumerate(all_results):
            cur_accuracy = cur_result['categorical_accuracy_heatmap'][:, int(digit)]
            angles = list(range(len(cur_accuracy)))

            # non eqv accuracy
            fig.add_trace(go.Scatterpolar(r = cur_accuracy,
                                            theta = angles,
                                            mode = 'lines',
                                            name = all_labels[j],
                                            line_color = all_colors[j]),
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
        if i > (len(all_labels)-1):
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


def plot_interactive_heatmap(name, digits, results, plot_title, result_path):

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
                                    name = name,
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


def plot_interactive_3D_scatter(digits, all_labels, all_colors, all_results, plot_title, result_path):

    # plot for all digits
    subplot_names = []
    for digit in digits:
        subplot_names.append(f'Digit {digit}')

    fig = make_subplots(rows=2, cols=5,
                        subplot_titles=subplot_names,
                        specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
                               [{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}]]
                        )

    for i, digit in enumerate(digits):

        # subplot position (start with 1)
        row = i // 5 + 1
        col = i % 5 + 1

        for j, cur_result in enumerate(all_results):
            cur_accuracy = cur_result['categorical_accuracy_heatmap'][:, :, int(digit)]

            vertical_translations = np.array(list(range(len(cur_accuracy))))-20
            horizontal_translations = np.array(list(range(len(cur_accuracy[0]))))-20

            # since scatter3d takes 1d matrix
            all_x = []
            all_y = []
            all_accuracy = []
            for y in vertical_translations:
                for x in horizontal_translations:
                    all_x.append(x)
                    all_y.append(y)
                    all_accuracy.append(cur_accuracy[y, x])

            # plot accuracy
            fig.add_trace(go.Scatter3d(x = all_x,
                                        y = all_y,
                                        z = all_accuracy,
                                        name = all_labels[j],
                                        mode='markers',
                                        marker=dict(
                                            size=2,
                                            color=all_colors[j],
                                            opacity=0.3
                                        )
                                        ),
                row = row,
                col = col
            )

        # set x axes on top
        # fig.update_xaxes(side='top')
        # fig.update_xaxes(side='top')
        # reverse both y axes
        fig.update_yaxes(autorange='reversed')
        fig.update_yaxes(autorange='reversed')

        # only show one legend
        for i, trace in enumerate(fig['data']):
            if i > (len(all_labels)-1):
                trace['showlegend'] = False

        fig.update_layout(
            title = plot_title,
            hovermode='x',
            # title_x=0.5,
            # xaxis=dict(title='x translation'),
            # yaxis=dict(title='y translation'),
            coloraxis=dict(colorscale = 'Viridis')
        )

        fig.update_coloraxes(colorbar=dict(title='Accuracy'))

        fig.update_annotations(yshift=20)

    # only show one legend
    # layout = Layout(showlegend=True,)

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


def plot_interactive_line(digits, scales, all_labels, all_colors, all_results, plot_title, result_path):


    # plot for all digits
    # fig = go.Figure()
    subplot_names = []
    for digit in digits:
        subplot_names.append(f'Digit {digit}')

    fig = make_subplots(rows=2, cols=5,
                        subplot_titles=subplot_names)
    for i, digit in enumerate(digits):

        # subplot position (start with 1)
        row = i // 5 + 1
        col = i % 5 + 1

        for k, cur_result in enumerate(all_results):
            cur_accuracy = cur_result['categorical_accuracy_heatmap'][:, int(digit)]
            # non eqv accuracy
            fig.add_trace(go.Scatter(x = scales,
                                     y = cur_accuracy,
                                    mode = 'lines',
                                    name = all_labels[k],
                                    line_color = all_colors[k]),
                row = row,
                col = col
            )

        fig.update_traces(
            hovertemplate=
            "Scaled factor: %{theta:.0f}<br>" +
            "Accuracy: %{r:.1%}<br>"
        )

        fig.update_annotations(yshift=20)

    # only show one legend
    for i, trace in enumerate(fig['data']):
        if i > 2:
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

    print(f'\nInteractive line plot has been saved to {result_path}')