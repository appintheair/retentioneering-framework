import seaborn as sns
import networkx as nx
import os
from datetime import datetime
import requests
from IPython.display import HTML
from IPython.display import display
import numpy as np
from retentioneering.analysis.utils import _check_folder
from retentioneering.utils.export import export_tracks


def plot_graph(df_agg, agg_type, settings, layout=nx.random_layout, plot_name=None):
    """
    Visualize trajectories from aggregated tables (with python)

    :param df_agg: table with aggregates (from retentioneering.analysis.get_all_agg function)
    :param agg_type: name of col for weighting graph nodes (column name from df)
    :param settings: experiment config (can be empty dict here)
    :param layout: function that return dictionary of positions keyed by node for NetworkX graph.
    :param plot_name: name of file with graph plot
    :type df_agg: pd.DataFrame
    :type agg_type: str
    :type settings: dict
    :type layout: func
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
    f = sns.mpl.pyplot.figure(figsize=(20, 10))
    nx.draw_networkx_edges(G, pos, edge_color='b', alpha=0.2, width=width)
    nx.draw_networkx_nodes(G, pos, node_color='b', alpha=0.3)
    pos = {k: [pos[k][0], pos[k][1] + 0.03] for k in pos.keys()}
    nx.draw_networkx_labels(G, pos, node_color='b', font_size=16)
    sns.mpl.pyplot.axis('off')

    settings = _check_folder(settings)
    export_folder = settings['export_folder']
    if plot_name:
        filename = os.path.join(export_folder, 'graphvis_{}.png'.format(plot_name))
    else:
        filename = os.path.join(export_folder, 'graphvis_{}.png'.format(datetime.now()))
    f.savefig(filename)


def plot_graph_api(df, settings, users='all', task='lost', order='all', threshold=0.5,
                   start_event=None, end_event=None):
    """
    Visualize trajectories from event clickstream (with Mathematica)

    :param df: event clickstream
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
