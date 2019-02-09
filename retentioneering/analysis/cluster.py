import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from retentioneering.visualization import plot


def cluster_users(countmap, n_clusters=None, clusterer=None):
    """
    Cluster users based on input dataframe and return DataFrame with `user_pseudo_id` and `cluster` for each user

    :param countmap: input dataframe, should have user_id in index. All fields will be features in clustering algorithm
    :param n_clusters: supposed number of clusters, could be None
    :param clusterer: clustering algorithm. Should have fit_predict function

    :type countmap: pd.DataFrame
    :type n_clusters: int
    :type clusterer: func

    :return: pd.DataFrame
    """
    users_clusters = pd.DataFrame(countmap.index.values, columns=['user_pseudo_id'])
    if clusterer is None:
        if n_clusters:
            clusterer = KMeans(n_clusters=n_clusters)
        else:
            nn = NearestNeighbors(metric='cosine')
            nn.fit(countmap.values)
            dists = nn.kneighbors(countmap.values, 2)[0][:, 1]
            eps = np.percentile(dists, 99)
            clusterer = DBSCAN(eps=eps, metric='cosine')
    users_clusters['cluster'] = clusterer.fit_predict(countmap)
    return users_clusters


def add_cluster_of_users(data, users_clusters, how='left'):
    """
    Add cluster of each user to clickstream data

    :param data: data from BQ or your own (clickstream). Should have at least one column: `user_pseudo_id`
    :param users_clusters: DataFrame with `user_pseudo_id` and `cluster` for each user
    :param how: argument to pass in pd.merge function

    :type data: pd.DataFrame
    :type users_clusters: pd.DataFrame
    :type how: str

    :return: pd.DataFrame
    """
    df = data.merge(users_clusters, how=how, on=['user_pseudo_id'])
    return df


def calculate_cluster_stats(data, users_clusters, settings, target_events=('lost', 'passed'), make_plot=True,
                            plot_count=2, save=True, plot_name=None, figsize=(10, 5)):
    """
    Plot pie-chart with distribution of target events in clusters

    :param data: data from BQ or your own (clickstream). Should have at least three columns: `event_name`,
            `event_timestamp` and `user_pseudo_id`
    :param users_clusters: DataFrame with `user_pseudo_id` and `cluster` for each user
    :param settings: experiment config (can be empty dict here)
    :param target_events: name of event which signalize target function
            (e.g. for prediction of lost users it'll be `lost`)
    :param make_plot: plot stats or not
    :param plot_count: number of plots for output
    :param save: True if the graph should be saved
    :param plot_name: name of file with graph plot
    :param figsize: width, height in inches. If None, defaults to rcParams["figure.figsize"] = [6.4, 4.8]

    :type data: pd.DataFrame
    :type users_clusters: pd.DataFrame
    :type settings: dict
    :type target_events: list or tuple
    :type make_plot: bool
    :type plot_count: int
    :type plot_name: str
    :type figsize: tuple

    :return: np.array
    """

    data = data.loc[data.event_name.isin(target_events), :]
    data = add_cluster_of_users(data, users_clusters, how='inner')

    main_classes = (users_clusters.groupby('cluster').size()
                    .sort_values(ascending=False)
                    .iloc[:plot_count].index.values)
    groups = (data.groupby(['cluster', 'event_name']).size()
              .unstack('event_name')
              .reindex(index=main_classes, columns=target_events)
              .fillna(0)
              .values
              .astype(int))
    if make_plot:
        plot.cluster_stats(groups, labels=target_events, settings=settings, plot_count=plot_count,
                           figsize=figsize, save=save, plot_name=plot_name)
    return groups
