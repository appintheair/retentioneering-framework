from IPython.display import display, HTML
from datetime import datetime
import networkx as nx
import numpy as np
import os
import requests
import seaborn as sns
from retentioneering.analysis.utils import _check_folder
from retentioneering.utils.export import export_tracks


def _save_graph(graph, graph_name, settings, plot_name=None):
    settings = _check_folder(settings)
    export_folder = settings['export_folder']
    if not plot_name:
        plot_name = datetime.now().strftime('%Y-%m-%dT%H_%M_%S_%f')
    export_filename = os.path.join(export_folder, '{}_{}.png'.format(graph_name, plot_name))
    if isinstance(graph, sns.mpl.axes.Axes):
        graph = graph.get_figure()
    graph.savefig(export_filename)


def plot_graph(df_agg, agg_type, settings, layout=nx.random_layout, save=True, figsize=(20, 10), plot_name=None):
    """
    Visualize trajectories from aggregated tables (with python)

    :param df_agg: table with aggregates (from retentioneering.analysis.get_all_agg function)
    :param agg_type: name of col for weighting graph nodes (column name from df)
    :param settings: experiment config (can be empty dict here)
    :param layout: function that return dictionary of positions keyed by node for NetworkX graph
    :param save: True if the graph should be saved
    :param figsize: width, height in inches. If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8]
    :param plot_name: name of file with graph plot
    :type df_agg: pd.DataFrame
    :type agg_type: str
    :type settings: dict
    :type layout: func
    :type save: bool
    :type figsize: tuple
    :type plot_name: str
    :return: None
    """
    edges = df_agg.loc[:, ['event_name', 'next_event', agg_type]]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges.values)

    width = [G.get_edge_data(i, j)['weight'] for i, j in G.edges()]
    width = np.array(width)

    if len(np.unique(width)) != 1:
        width = (width - width.min()) / (np.mean(width) - width.min())
        width *= 2
        width = np.where(width > 15, 15, width)
        width = np.where(width < 2, 2, width)
    else:
        width = width * 3 / max(width)

    pos = layout(G)
    f = sns.mpl.pyplot.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, edge_color='b', alpha=0.2, width=width)
    nx.draw_networkx_nodes(G, pos, node_color='b', alpha=0.3)
    pos = {k: [pos[k][0], pos[k][1] + 0.03] for k in pos.keys()}
    nx.draw_networkx_labels(G, pos, node_color='b', font_size=16)
    sns.mpl.pyplot.axis('off')
    if save:
        _save_graph(f, 'graphvis', settings, plot_name)


def plot_graph_api(df, settings, users='all', task='lost', order='all', threshold=0.5,
                   start_event=None, end_event=None):
    """
    Visualize trajectories from event clickstream (with Mathematica)

    :param df: data from BQ or your own (clickstream). Should have at least three columns: `event_name`,
            `event_timestamp` and `user_pseudo_id`
    :param settings: experiment config (can be empty dict here)
    :param users: `all` or list of user ids to plot specific group
    :param task: type of task for different visualization (can be `lost` or `prunned_welcome`)
    :param order: depth in sessions for filtering
    :param threshold: threshold for session splitting
    :param start_event: name of start event in trajectory
    :param end_event: name of last event in trajectory

    :param df: pd.DataFrame
    :param settings: dict
    :param users: str or list
    :param task: str
    :param order: int
    :param threshold: float
    :param start_event: str
    :param end_event: str

    :return: None
    """
    export_folder, graph_name, set_name = export_tracks(df, settings, users, task, order, threshold,
                                                        start_event, end_event)
    _api_plot(export_folder, graph_name, set_name, plot_type=task)
    path = os.path.join(export_folder, 'graph_plot.pdf')
    display(HTML("<a href='{href}'> {href} </a>".format(href=path)))
    # try:
    #     img = WImage(filename=path)
    #     return img
    # except:
    print("Please check on path behind")


def _api_plot(export_folder, graph_name, set_name, plot_type='lost', download_path=None):
    if not download_path:
        download_path = export_folder

    url = 'http://35.230.23.217:5001/'
    files = {
        'graph': ('graph.csv', open(os.path.join(export_folder, graph_name), 'rb'), 'multipart/form-data'),
        'settings': ('settings.json', open(os.path.join(export_folder, set_name), 'rb'), 'multipart/form-data')}

    r = requests.post(url, files=files, headers={'plot_type': plot_type}, auth=('admin', 'admin'))
    if r.content == 'File was not proceed':
        print("Can't plot graph for this data")
    else:
        with open(os.path.join(download_path, 'graph_plot.pdf'), 'wb') as f:
            f.write(r.content)


def bars(x, y, settings=dict(), figsize=(8, 5), save=True, plot_name=None):
    """
    Plot bar graph

    :param x: bars names
    :param y: bars values
    :param settings: experiment config (can be empty dict here)
    :param figsize: width, height in inches. If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8]
    :param save: True if the graph should be saved
    :param plot_name: name of file with graph plot

    :type x: list
    :type y: list
    :type settings: dict
    :type figsize: tuple
    :type save: bool
    :type plot_name: str
    :return: None
    """
    sns.mpl.pyplot.figure(figsize=figsize)
    bar = sns.barplot(x, y, palette='YlGnBu')
    bar.set_xticklabels(bar.get_xticklabels(), rotation=90)

    if save:
        _save_graph(bar, 'bar', settings, plot_name)


def heatmap(x, labels=None, settings=dict(), figsize=(10, 15), save=True, plot_name=None):
    """
    Plot heatmap graph

    :param x: data to visualize
    :param labels: list of labels for x ticks
    :param settings: experiment config (can be empty dict here)
    :param figsize: width, height in inches. If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8]
    :param save: True if the graph should be saved
    :param plot_name: name of file with graph plot

    :type x: list[list]
    :type labels: str
    :type settings: dict
    :type figsize: tuple
    :type save: bool
    :type plot_name: str
    :return: None
    """
    sns.mpl.pyplot.figure(figsize=figsize)
    heatmap = sns.heatmap(x, cmap="YlGnBu")
    if labels is not None:
        heatmap.set_xticklabels(labels, rotation=90)

    if save:
        _save_graph(heatmap, 'countmap', settings, plot_name)


def cluster_stats(data, labels=None, settings=dict(), plot_count=2, figsize=(10, 5), save=True, plot_name=None):
    """
    Plot pie chart with different events

    :param data: list of lists with size of each group
    :param labels: list of labels for each group
    :param settings: experiment config (can be empty dict here)
    :param plot_count: number of plots to visualize
    :param figsize: width, height in inches. If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8]
    :param save: True if the graph should be saved
    :param plot_name: name of file with graph plot
    :type data: list
    :type labels: list or tuple
    :type settings: dict
    :type plot_count: int
    :type figsize: tuple
    :type save: bool
    :type plot_name: str
    :return: None
    """
    if plot_count > len(data):
        plot_count = len(data)
    fig, ax = sns.mpl.pyplot.subplots(1 if plot_count <= 2 else ((plot_count - 1) // 2 + 1), (plot_count > 1) + 1)
    fig.set_size_inches(*figsize)
    i = 0
    for i, group_sizes in enumerate(data[:plot_count]):
        if plot_count == 1:
            cur_ax = ax
        elif plot_count == 2:
            cur_ax = ax[i]
        else:
            cur_ax = ax[i // 2][i % 2]
        cur_ax.pie(group_sizes, labels=labels, autopct='%1.1f%%')
        cur_ax.set_title('Class {}'.format(i))
    if plot_count > 1 and i % 2 != 1:
        fig.delaxes(ax[i // 2, 1])

    if save:
        _save_graph(fig, 'clusters', settings, plot_name)
