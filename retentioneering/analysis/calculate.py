import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from retentioneering.analysis.utils import prepare_dataset
from retentioneering.visualization import plot


def calculate_frequency_hist(df, settings, target_events=None,
                             make_plot=True, save=True, plot_name=None, figsize=(8, 5)):
    """
    Calculate frequency of each event from input clickstream and plot a barplot

    :param df: data from BQ or your own (clickstream). Should have at least three columns: `event_name`,
            `event_timestamp` and `user_pseudo_id`
    :param settings: experiment config (can be empty dict here)
    :param target_events: name of event which signalize target function
            (e.g. for prediction of lost users it'll be `lost`)
    :param make_plot: plot stats or not
    :param save: True if the graph should be saved
    :param plot_name: name of file with graph plot
    :param figsize: width, height in inches. If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8]

    :type df: pd.DataFrame
    :type settings: dict
    :type target_events: Union[tuple, list, str, None]
    :type make_plot: bool
    :type save: bool
    :type plot_name: str
    :type figsize: tuple
    :return: pd.DataFrame
    """
    if isinstance(target_events, str):
        target_events = [target_events]

    if target_events is not None:
        users = df.user_pseudo_id[df.event_name.isin(target_events)].unique()
        df = df[df.user_pseudo_id.isin(users)]

    nodes_hist = (df.groupby('event_name', as_index=False)
                  .event_timestamp.count()
                  .sort_values('event_timestamp', ascending=False))
    nodes_hist.event_name = nodes_hist.event_name.apply(lambda x: x.lower())
    if make_plot:
        plot.bars(nodes_hist.event_name.values, nodes_hist.event_timestamp.values, settings,
                  save=save, plot_name=plot_name, figsize=figsize)
    return nodes_hist


def calculate_frequency_map(df, settings, target_events=None, plot_name=None,
                            make_plot=True, save=True, figsize_hist=(8, 5), figsize_heatmap=(10, 15)):
    """
    Calculate frequency of each event for each user from input clickstream and plot a heatmap

    :param df: data from BQ or your own (clickstream). Should have at least three columns: `event_name`,
            `event_timestamp` and `user_pseudo_id`
    :param settings: experiment config (can be empty dict here)
    :param target_events: name of event which signalize target function
            (e.g. for prediction of lost users it'll be `lost`)
    :param plot_name: name of file with graph plot
    :param make_plot: plot stats or not
    :param save: True if the graph should be saved
    :param figsize_hist: width, height in inches for bar plot with events. If None, defaults to rcParams["figure.figsize"] = [6.4, 4.8]
    :param figsize_heatmap: width, height in inches for heatmap. If None, defaults to rcParams["figure.figsize"] = [6.4, 4.8]

    :type df: pd.DataFrame
    :type settings: dict
    :type target_events: Union[tuple, list, str, None]
    :type plot_name: str
    :type make_plot: bool
    :type save: bool
    :type figsize_hist: tuple
    :type figsize_heatmap: tuple
    :return: pd.DataFrame
    """
    if isinstance(target_events, str):
        target_events = [target_events]

    if target_events is not None:
        users = df.user_pseudo_id[df.event_name.isin(target_events)].unique()
        df = df[df.user_pseudo_id.isin(users)]
    data = prepare_dataset(df, target_events)

    cv = CountVectorizer()
    x = cv.fit_transform(data.event_name.values).todense()
    cols = cv.inverse_transform(x[0] + 1)[0]
    x = pd.DataFrame(x, columns=cols, index=data.user_pseudo_id)
    nodes_hist = calculate_frequency_hist(df=df, settings=settings, target_events=target_events,
                                          make_plot=make_plot, save=save, plot_name=plot_name, figsize=figsize_hist)

    sorted_cols = nodes_hist.event_name[~nodes_hist.event_name.isin(target_events or [])].values
    x = x.loc[:, sorted_cols]
    x = x.sort_values(list(sorted_cols), ascending=False)
    if make_plot:
        plot.heatmap(x.values, sorted_cols, settings=settings, save=save, plot_name=plot_name, figsize=figsize_heatmap)
    return x
