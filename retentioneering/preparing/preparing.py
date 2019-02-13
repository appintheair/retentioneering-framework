import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


def drop_duplicated_events(df, duplicate_thr_time=0, settings=None):
    """
    Delete duplicated events (two events with save event names if the time between them less than duplicate_thr_time).

    :param df: input pd.DataFrame
    :param duplicate_thr_time: threshold for time between events
    :param settings: config dict

    :type df: pd.DataFrame
    :type duplicate_thr_time: int
    :type settings: dict
    :return: self
    :rtype: pd.DataFrame
    """
    logging.info('Start. Shape: {}'.format(df.shape))
    df = df.sort_values(['user_pseudo_id', 'event_timestamp'])
    is_first_iter = 1
    if settings is not None:
        duplicate_thr_time = settings.get('events', {}).get('duplicate_thr_time', 0)
    duplicated_rows = None

    while is_first_iter or duplicated_rows.sum() != 0:
        if is_first_iter != 1:
            df = df.loc[~duplicated_rows, :]
        is_first_iter = 0
        df.loc[:, 'prev_timestamp'] = df.event_timestamp.shift(1)
        df.loc[:, 'prev_user'] = df.user_pseudo_id.shift(1)
        df.loc[:, 'prev_event_name'] = df.event_name.shift(1)

        duplicated_rows = (((df.event_timestamp - df.prev_timestamp) <= duplicate_thr_time) &
                           (df.prev_event_name == df.event_name) &
                           (df.prev_user == df.user_pseudo_id))
    logging.info('Done. Shape: {}'.format(df.shape))
    df = df.drop(['prev_timestamp', 'prev_user', 'prev_event_name'], axis=1)
    return df


def filter_users(df, filters=list(), settings=None):
    """
    Apply filters to users from the input table and leave all events for the received users.

    :param df: input pd.DataFrame
    :param filters: list each element of which is a filter dict
    :param settings: config dict

    :type df: pd.DataFrame
    :type filters: list
    :type settings: dict
    :return: pd.DataFrame
    """
    logging.info('Start. Shape: {}'.format(df.shape))
    if settings is not None:
        filters = settings.get('users', {}).get('filters', [])
    conditions = _filter_conditions(df, filters)
    if conditions is not None:
        df = df.loc[df.user_pseudo_id.isin(df.loc[conditions, 'user_pseudo_id']), :].copy()
    logging.info('Done. Shape: {}'.format(df.shape))
    return df


def filter_events(df, filters=list(), settings=None):
    """
    Apply filters to the input table.

    :param df: input pd.DataFrame
    :param filters: list each element of which is a filter dict
    :param settings: config dict

    :type df: pd.DataFrame
    :type filters: list
    :type settings: dict
    :return: self
    :rtype: pd.DataFrame
    """
    logging.info('Start. Shape: {}'.format(df.shape))
    if settings is not None:
        filters = settings.get('events', {}).get('filters', [])
    conditions = _filter_conditions(df, filters)
    if conditions is not None:
        df = df.loc[conditions, :].copy()
    logging.info('Done. Shape: {}'.format(df.shape))
    return df


def _filter_conditions(df, filters):
    if len(filters) == 0:
        return None
    conditions = False
    for i, one_filter in enumerate(filters):
        event_name = one_filter.get('event_name')
        event_value = one_filter.get('event_params_value_string_value')
        is_not = one_filter.get('not', False)
        condition = True
        if event_name:
            condition &= (df.event_name == event_name)
        if event_value:
            condition &= (df.event_params_value_string_value == event_value)
        if is_not:
            condition = ~condition
        conditions |= condition
    return conditions


def add_passed_event(df, positive_event_name=u'passed', filters=None, settings=None):
    """
    Add new events with `positive_event_name` and delete all events after.

    :param df: input pd.DataFrame
    :param positive_event_name: name of the positive event which should be added if filter conditions is True
    :param filters: dict with filter conditions
    :param settings: dict

    :type df: pd.DataFrame
    :type positive_event_name: str
    :type filters: dict
    :type settings: config dict
    :return: self
    :rtype: pd.DataFrame
    """
    logging.info('Start. Shape: {}'.format(df.shape))
    if settings is not None:
        positive_event_name = settings.get('positive_event', {}).get('name', u'passed')
        filters = settings.get('positive_event', {}).get('filters', None)
    if filters is None:
        logging.info('Done. Shape: {}'.format(df.shape))
        return df
    head_match = filters.get('match_up_to_separator', {})
    full_match_list = filters.get('full_match', [])
    df.loc[:, 'target_event'] = 0
    if len(head_match):
        head_match_sep = head_match.get('sep', '_')
        head_match_list = head_match.get('values', [])
        if len(head_match_list):
            df.loc[df.event_name.str.split(head_match_sep, 1).str[0].isin(head_match_list), 'target_event'] = 1
    if len(full_match_list):
        df.loc[df.event_name.isin(full_match_list), 'target_event'] = 1
    if df.target_event.sum() != 0:
        # add the time of the first event "passed"
        first_positive_event = df.loc[df.target_event == 1, :] \
            .groupby('user_pseudo_id').event_timestamp.min() \
            .rename('event_timestamp_passed') \
            .reset_index()
        df = df.merge(first_positive_event, how='left', on=['user_pseudo_id'])

        # leave only events before the "passed" event
        df = df.loc[df.event_timestamp_passed.isnull() | (df.event_timestamp <= df.event_timestamp_passed), :]
        df.loc[df.target_event == 1, 'event_name'] = positive_event_name
        df = df.drop('event_timestamp_passed', axis=1)
    df = df.drop('target_event', axis=1)
    logging.info('Done. Shape: {}'.format(df.shape))
    return df


def add_lost_events(df, positive_event_name=u'passed', negative_event_name=u'lost', settings=None):
    """
    Add new events with `negative_event_name` in input DataFrame.

    :param df: input pd.DataFrame
    :param positive_event_name: positive event name
    :param negative_event_name: negative event name which should be added if there is no positive event in the session
    :param settings: config dict

    :type df: pd.DataFrame
    :type positive_event_name: str
    :type negative_event_name: str
    :type settings: dict
    :return: self
    :rtype: pd.DataFrame
    """
    logging.info('Start. Shape: {}'.format(df.shape))
    if settings is not None:
        positive_event_name = settings.get('positive_event', {}).get('name', u'passed')
        negative_event_name = settings.get('negative_event', {}).get('name', u'lost')

    df = df.sort_values(['user_pseudo_id', 'event_timestamp'])
    last_row = df.groupby('user_pseudo_id', as_index=False).last()
    last_row = last_row.loc[
        last_row.event_name != positive_event_name, ['user_pseudo_id', 'event_name', 'event_timestamp']]
    if len(last_row):
        last_row.loc[:, 'event_name'] = negative_event_name
        last_row.loc[:, 'event_timestamp'] += 1

        df = df.append(last_row, sort=False)
        df = df.sort_values(['user_pseudo_id', 'event_timestamp']).copy()
    logging.info('Done. Shape: {}'.format(df.shape))
    return df


class SessionSplitter(object):
    """
    Class for session splitting processing
    """

    def __init__(self, n_components):
        self.model = None
        self.n_components = n_components
        self.columns_config = None
        self.bics = []
        self.unit = None
        self.delta_unit = None

    def fit(self, df, columns_config, unit=None, delta_unit='s'):
        """
        Fits the gausian mixture model for understanding threshold.

        :param df: DataFrame with columns responding for
                   Event Name, Event Timestamp and User ID
        :param columns_config: Dictionary that maps to required column names:
                               {'event_name_col': Event Name Column,
                               'event_timestamp_col' Event Timestamp Column,
                               'user_id_col': User ID Column}
        :param unit: type of string for pd.datetime parsing
        :param delta_unit: step in timestamp column (e.g. seconds from `01-01-1970`)
        :return: None
        """

        self.columns_config = columns_config
        self.unit = unit
        self.delta_unit = delta_unit
        df = self.add_time_from_prev_event(df.copy(), unit=self.unit, delta_unit=self.delta_unit)
        time_from_prev = np.log(df.loc[df.from_prev_event > 0, 'from_prev_event'].values)
        X = time_from_prev.reshape(-1, 1)
        if isinstance(self.n_components, int):
            self.model = GaussianMixture(n_components=self.n_components)
            self.model.fit(X)
            self.bics.append(self.model.bic(X))
        else:
            best_bic = np.infty
            best_model = None
            for n in self.n_components:
                model = GaussianMixture(n_components=n)
                model.fit(X)
                curr_bic = model.bic(X)
                self.bics.append(curr_bic)

                if curr_bic < best_bic:
                    best_model = model
                    best_bic = curr_bic

            self.model = best_model

    def predict(self, df, thr_prob=0.95, thrs=np.linspace(1, 3000, 11997), sort=True):
        """
        Predicts sessions for passed DataFrame.

        :param df: DataFrame with columns responding for
                   Event Name, Event Timestamp and User ID
        :param thr_prob: Probability threshold for session interruption
        :param thrs: Timedelta values for threshold checking
        :param sort: If sorting by User ID & Event Timestamp is required

        :type df: pd.DataFrame
        :type thr_prob: float
        :type thrs: List[float] or iterable
        :type sort: bool
        :return: self
        :rtype: pd.DataFrame
        """

        df = self.add_time_from_prev_event(df.copy(), unit=self.unit, delta_unit=self.delta_unit)
        thr = self.get_threshold(thr_prob, thrs)
        return self.add_session_column(df, thr, sort)

    def visualize(self, df, figsize=(15, 5), dpi=500, **kwargs):
        """
        Visualize mixture of found distributions.

        :param df: DataFrame with columns responding for
                   Event Name, Event Timestamp and User ID
        :param figsize: size of plot
        :param dpi: dot per inch to saving plot
        :type df: pd.DataFrame
        :type figsize: tuple
        :type dpi: int
        :return: None
        """
        df = self.add_time_from_prev_event(df.copy())
        time_from_prev = np.log(df.loc[df.from_prev_event > 0, 'from_prev_event'].values)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.hist(time_from_prev, **kwargs)
        res = norm.pdf(np.linspace(-10, 20), loc=self.model.means_[:, [0]], scale=self.model.covariances_[:, 0])
        for i, r in enumerate(res):
            plt.plot(np.linspace(-10, 20), r, label=str(i))
        plt.legend()

    def add_time_from_prev_event(self, df, unit=None, delta_unit='s'):
        """
        Adds time from previous event column.

        :param df: DataFrame with columns responding for
                   Event Name, Event Timestamp and User ID
        :param unit: type of string for pd.datetime parsing
        :param delta_unit: step in timestamp column (e.g. seconds from `01-01-1970`)
        :type df: pd.DataFrame
        :type unit: str
        :type delta_unit: str
        :return: input data with column `from_prev_event`
        :rtype: pd.DataFrame
        """
        df[self.columns_config['event_timestamp_col']] = pd.to_datetime(
            df[self.columns_config['event_timestamp_col']],
            unit=unit
        ).dt.tz_localize(None)
        df = df.loc[:, [self.columns_config['event_name_col'],
                        self.columns_config['event_timestamp_col'],
                        self.columns_config['user_id_col']]] \
            .sort_values([self.columns_config['user_id_col'], self.columns_config['event_timestamp_col']]) \
            .reset_index(drop=True)

        df.loc[:, 'from_prev_event'] = (
                (df[self.columns_config['event_timestamp_col']] -
                 df.groupby([self.columns_config['user_id_col']])[self.columns_config['event_timestamp_col']].shift()) /
                np.timedelta64(1, delta_unit))
        return df

    def get_threshold(self, thr=0.95, thrs=np.linspace(1, 3000, 11997)):
        """
        Finds best threshold.

        :param thr: Probability threshold for session interruption
        :param thrs: Timedelta values for threshold checking
        :type thr: float in interval (0, 1)
        :type thrs: List[float] or iterable
        :return: value of threshold
        :rtype: float
        """
        thrs_probs = self.model.predict_proba(np.log(thrs).reshape(-1, 1))[:, np.argmax(self.model.means_)]
        threshold = thrs[np.argmax(thrs_probs >= thr)]
        return threshold

    def add_session_column(self, df, thr, sort):
        """
        Creates columns with session rank.

        :param df: DataFrame with columns responding for
                   Event Name, Event Timestamp and User ID
        :param thr: time threshold from previous step for session interruption
        :param sort: If sorting by User ID & Event Timestamp is required

        :type df: pd.DataFrame
        :type thr: float
        :type sort: bool
        :return: input data with columns `session`
        :rtype: pd.DataFrame
        """
        if sort:
            df = df.loc[:, [self.columns_config['event_name_col'],
                            self.columns_config['event_timestamp_col'],
                            self.columns_config['user_id_col'], 'from_prev_event']] \
                .sort_values([self.columns_config['user_id_col'], self.columns_config['event_timestamp_col']]) \
                .reset_index(drop=True).copy()
        df.loc[:, 'session'] = 0
        df.loc[(df['from_prev_event'] >= thr) | df['from_prev_event'].isnull(), 'session'] = 1
        df['session'] = df['session'].cumsum()
        return df


def add_first_and_last_events(df, first_event_name='fisrt_event', last_event_name='last_event'):
    """
    For every user and session adds first event with `first_event_name` and last event with `last_event_name`.

    :param df: input DataFrame
    :param first_event_name: name of the first event
    :param last_event_name: name of the last event

    :type df: pd.DataFrame
    :type first_event_name: str
    :type last_event_name: str
    :return: self
    :rtype: pd.DataFrame
    """
    df = df.sort_values(['user_pseudo_id', 'session', 'event_timestamp'])
    first_row = df.groupby(['user_pseudo_id', 'session'], as_index=False).first()
    last_row = df.groupby(['user_pseudo_id', 'session'], as_index=False).last()

    first_row = first_row.loc[
        first_row.event_name != first_event_name, ['user_pseudo_id', 'session', 'event_name', 'event_timestamp']]

    if len(first_row):
        first_row.loc[:, 'event_name'] = first_event_name
        first_row.loc[:, 'event_timestamp'] -= 1
        df = df.append(first_row, sort=False)
        df = df.sort_values(['user_pseudo_id', 'session', 'event_timestamp']).copy()

    last_row = last_row.loc[
        last_row.event_name != last_event_name, ['user_pseudo_id', 'session', 'event_name', 'event_timestamp']]

    if len(last_row):
        last_row.loc[:, 'event_name'] = last_event_name
        last_row.loc[:, 'event_timestamp'] += 1
        df = df.append(last_row, sort=False)
        df = df.sort_values(['user_pseudo_id', 'session', 'event_timestamp']).copy()
    return df
