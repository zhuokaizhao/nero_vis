import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_interactive_line_polar(digits, all_labels, all_colors, all_styles, all_results, plot_title, subplot_names, result_path):

    # plot for all digits/indices
    num_row = len(digits) // 5
    num_col = len(digits) // 2
    if num_row == 0:
        num_row += 1
    if num_col == 0:
        num_col += 1
    subplot_specs = []
    for i in range(num_row):
        subplot_specs.append([])
        for j in range(num_col):
            subplot_specs[i].append({'type': 'polar'})

    # 10 digits, divided into 2*5 subplots
    fig = make_subplots(rows=num_row, cols=num_col,
                        subplot_titles=subplot_names,
                        specs=subplot_specs
                        )
    for i, digit in enumerate(digits):

        # subplot position (start with 1)
        row = i // 5 + 1
        col = i % 5 + 1

        for j, cur_result in enumerate(all_results):
            cur_value = cur_result[:, int(digit)]
            angles = list(range(len(cur_value)))

            # non eqv accuracy
            fig.add_trace(go.Scatterpolar(r = cur_value,
                                            theta = angles,
                                            mode = 'lines',
                                            name = all_labels[j],
                                            line = {'dash': f'{all_styles[j]}'},
                                            line_color = all_colors[j]),
                row = row,
                col = col
            )

        fig.update_traces(
            hovertemplate=
            "Rotated angle: %{theta:.0f}<br>" +
            # "Accuracy: %{r:.1%}<br>"
            "Loss: %{r:.2}<br>"
        )

        fig.update_annotations(yshift=20)

    # only show one legend
    for i, trace in enumerate(fig['data']):
        if i > (len(all_labels)-1):
            trace['showlegend'] = False

    fig.update_layout(
        title = plot_title,
        hovermode='x',
        polar1 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar2 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar3 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar4 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar5 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar6 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar7 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar8 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar9 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
        polar10 = dict(
            radialaxis = dict(range=[0, 1.1], showticklabels=True, dtick=0.2),
            angularaxis = dict(showticklabels=False, ticks='')
        ),
    )

    # fig.for_each_annotation(lambda a: a.update(text = subplot_names[a.text]))
    if result_path[-4:] == 'html':
        fig.write_html(result_path)
    else:
        fig.write_image(result_path)

    print(f'\nInteractive polar plot has been saved to {result_path}')


def plot_interactive_heatmap(digits, all_labels, all_results, plot_title, result_path):

    # plot for all digits
    # fig = go.Figure()
    subplot_names = []
    for digit in digits:
        subplot_names.append(f'Digit {digit}')

    # rows are different results, columns are different digits
    fig = make_subplots(rows=len(all_results), cols=len(digits),
                        subplot_titles=subplot_names,)

    for j, cur_result in enumerate(all_results):
        for i, digit in enumerate(digits):

            accuracy = cur_result['categorical_accuracy_heatmap'][:, :, int(digit)]

            vertical_translation = len(accuracy)
            horizontal_translation = len(accuracy[0])

            # plot accuracy
            fig.add_trace(go.Heatmap(z = accuracy,
                                        x = np.array(list(range(horizontal_translation)))-20,
                                        y = np.array(list(range(vertical_translation)))-20,
                                        customdata=accuracy,
                                        type='heatmap',
                                        coloraxis='coloraxis1'),
                row = j+1, # the starting cell is (1, 1)
                col = i+1
            )

    # reverse both y axes
    fig.update_yaxes(autorange='reversed')
    fig.update_yaxes(autorange='reversed')

    fig.update_layout(
        title = plot_title,
        showlegend = False,
        hovermode='x',
        # title_x=0.5,
        # xaxis=dict(title='x translation'),
        yaxis1=dict(title=f'{all_labels[0]}'),
        yaxis4=dict(title=f'{all_labels[1]}'),
        yaxis7=dict(title=f'{all_labels[2]}'),
        coloraxis=dict(colorscale = 'Viridis')
    )

    # colorbar title
    fig.update_coloraxes(colorbar=dict(title='Accuracy'))

    fig.update_annotations(yshift=20)

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


def plot_interactive_scatter(dim, digits, all_labels, all_colors, all_styles, all_results, plot_title, result_path):

    # plot for all digits
    subplot_names = []
    for digit in digits:
        subplot_names.append(f'Digit {digit}')

    num_row = len(digits) // 5
    num_col = len(digits) // 2
    if num_row == 0:
        num_row += 1
    if num_col == 0:
        num_col += 1

    if dim == 2:
        fig = make_subplots(rows=num_row, cols=num_col, subplot_titles=subplot_names)
    elif dim == 3:
        fig = make_subplots(rows=num_row, cols=num_col,
                            subplot_titles=subplot_names,
                            specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
                                    [{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}]]
                            )

    for i, digit in enumerate(digits):

        # subplot position (start with 1)
        row = i // 5 + 1
        col = i % 5 + 1

        for j, cur_result in enumerate(all_results):

            cur_value = cur_result[i]

            # plot result
            if dim == 2:
                fig.add_trace(go.Scatter(x = cur_value[:, 0],
                                        y = cur_value[:, 1],
                                        name = all_labels[j],
                                        mode='markers',
                                        marker=dict(
                                            size=6,
                                            color=all_colors[j],
                                            opacity=0.5,
                                            symbol=all_styles[j]
                                        )),
                    row = row,
                    col = col
                )
            elif dim == 3:
                fig.add_trace(go.Scatter3d(x = cur_value[:, 0],
                                            y = cur_value[:, 1],
                                            z = cur_value[:, 2],
                                            name = all_labels[j],
                                            mode='markers',
                                            marker=dict(
                                                size=6,
                                                color=all_colors[j],
                                                opacity=0.5
                                            )),
                    row = row,
                    col = col
                )

        # reverse both y axes
        # fig.update_yaxes(autorange='reversed')
        # fig.update_yaxes(autorange='reversed')

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
        )

        # fig.update_coloraxes(colorbar=dict(title='Accuracy'))

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

    print(f'\nInteractive {dim}D scatter plot has been saved to {result_path}')


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
            cur_value = cur_result[:, int(digit)]
            # non eqv accuracy
            fig.add_trace(go.Scatter(x = scales,
                                     y = cur_value,
                                    mode = 'lines',
                                    name = all_labels[k],
                                    line_color = all_colors[k]),
                row = row,
                col = col
            )

        fig.update_traces(
            hovertemplate=
            "Scaled factor: %{theta:.0f}<br>" +
            # "Accuracy: %{r:.1%}<br>"
            "Loss: %{r:.2}<br>"
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