import logging


def drop_duplicated_events(df, duplicate_thr_time=0, settings=None):
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
    logging.info('Start. Shape: {}'.format(df.shape))
    if settings is not None:
        filters = settings.get('users', {}).get('filters', [])
    conditions = filter_conditions(df, filters)
    if conditions is not None:
        df = df.loc[df.user_pseudo_id.isin(df.loc[conditions, 'user_pseudo_id']), :].copy()
    logging.info('Done. Shape: {}'.format(df.shape))
    return df


def filter_events(df, filters=list(), settings=None):
    logging.info('Start. Shape: {}'.format(df.shape))
    if settings is not None:
        filters = settings.get('events', {}).get('filters', [])
    conditions = filter_conditions(df, filters)
    if conditions is not None:
        df = df.loc[conditions, :].copy()
    logging.info('Done. Shape: {}'.format(df.shape))
    return df


def filter_conditions(df, filters):
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
