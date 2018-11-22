from datetime import datetime
import logging
import pandas as pd
from tqdm import tqdm
from retentioneering.utils import queries

logger = logging.getLogger()


def download_bq(client, query, job_config=None, group_name=None, return_dataframe=True, return_only_query=False,
                hide_progress_bar=False, progress_bar_min_interval=4, **params):
    start = datetime.now()
    query = query.format(**params)
    if return_only_query:
        return query
    query_results = client.query(query, job_config=job_config)
    rows = query_results.result()
    total_time = datetime.now() - start
    logging.info('Query complete in {} seconds\n'.format(total_time.total_seconds()))
    dest_table = client.get_table(job_config.destination)
    result = []
    for i, row in tqdm(enumerate(rows), mininterval=progress_bar_min_interval, total=dest_table.num_rows,
                       disable=hide_progress_bar):
        items = list(row.values())
        if group_name is not None:
            items += [group_name]
        result.append(items)
    if return_dataframe:
        col_names = [col.name for col in dest_table.schema]
        if group_name is not None:
            col_names += ['group_name']
        result = pd.DataFrame(result, columns=col_names)
    return result


def prepare_event_filter_query(event_names=None, table_with_events=None):
    if table_with_events is None:
        if event_names is None:
            return ''
        is_in = len(event_names) > 1
        filter_query = '\'' + '\',\''.join(event_names) + '\''
        if is_in:
            filter_query = 'WHERE event_name IN (' + filter_query + ')'
        else:
            filter_query = 'WHERE event_name = ' + filter_query
    else:
        filter_query = "WHERE event_name IN (SELECT string_field_0 FROM {})".format(table_with_events)
    return filter_query


def prepare_app_version_filter_query(app_version=None, is_not_first=True):
    if app_version is None:
        return ''
    app_version_filter = "app_info.version = '{}'".format(app_version)
    if is_not_first:
        app_version_filter = 'AND ' + app_version_filter
    else:
        app_version_filter = 'WHERE ' + app_version_filter
    return app_version_filter


def prepare_drop_duplicates_flag(drop_duplicates=None):
    if drop_duplicates is None:
        return ''
    res = ", ROW_NUMBER() OVER (PARTITION BY {}) AS row_n".format(','.join(['tbl1.' + col for col in drop_duplicates]))
    return res


def download_events_multi(client, job_config, settings=None, return_only_query=False, **kwargs):
    df = pd.DataFrame()
    res = []
    if settings is not None:
        for settings_name, settings_config in settings['sql'].items():
            if return_only_query:
                res.append(download_events(client=client, job_config=job_config, return_dataframe=True,
                                           settings=settings_config, group_name=settings_name, 
                                           return_only_query=return_only_query, **kwargs))
            else:
                job_config.write_disposition = "WRITE_TRUNCATE"
                job_config.destination = client.dataset(settings_config['destination_table']['dataset']) \
                    .table(settings_config['destination_table']['table'])
                df = df.append(
                    download_events(client=client, job_config=job_config, return_dataframe=True,
                                    settings=settings_config, group_name=settings_name, **kwargs), sort=False)
    if return_only_query:
        return res
    return df


def download_events(client, job_config, user_filter_event_names=None, user_filter_event_table=None, dates_users=None,
                    users_app_version=None, event_filter_event_names=None, event_filter_event_table=None,
                    dates_events=None, events_app_version=None, count_events=None, use_last_events=False,
                    random_user_limit=None, random_seed=None, settings=None, group_name=None, drop_duplicates=None,
                    return_dataframe=True, return_only_query=False, hide_progress_bar=False,
                    progress_bar_min_interval=4):
    if settings is not None:
        user_filter_event_names = settings['user_filters'].get('event_names')
        users_app_version = settings['user_filters'].get('app_version')
        dates_users = (settings['user_filters']['date_start'], settings['user_filters']['date_finish'])
        event_filter_event_names = settings['event_filters'].get('event_names')
        events_app_version = settings['event_filters'].get('app_version')
        dates_events = (settings['event_filters']['date_start'], settings['event_filters']['date_finish'])
        count_events = settings['event_filters'].get('count_events', None)
        use_last_events = settings['event_filters'].get('use_last_events', False)
        random_user_limit = settings['user_filters'].get('limit')
        drop_duplicates = settings.get('drop_duplicates_events', None)
        if random_user_limit:
            if 'random_seed' not in settings['user_filters']:
                settings['user_filters']['random_seed'] = 42
            random_seed = settings['user_filters']['random_seed']
    else:
        assert all(v is not None for v in [dates_events, dates_users])

    users_event_filter = prepare_event_filter_query(user_filter_event_names, user_filter_event_table)
    users_app_version_filter = prepare_app_version_filter_query(users_app_version, users_event_filter)
    events_filter = prepare_event_filter_query(event_filter_event_names, event_filter_event_table)
    events_app_version_filter = prepare_app_version_filter_query(events_app_version, events_filter)
    duplicate_event_flag = prepare_drop_duplicates_flag(drop_duplicates)

    new_users_query = queries.query_with_params.format(
        rank_sort_type='DESC' if use_last_events else '',
        date_events_first=dates_events[0],
        date_events_last=dates_events[1],
        date_users_events_first=dates_users[0],
        date_users_events_last=dates_users[1],
        events_filter=events_filter if events_filter else '',
        events_app_version_filter=events_app_version_filter,
        min_event_timestamp=", MIN(event_timestamp) AS min_event_timestamp" if use_last_events else '',
        users_event_filter=users_event_filter if users_event_filter else '',
        users_app_version_filter=users_app_version_filter,
        delete_events_after_target="WHERE tbl1.event_timestamp <= tbl2.min_event_timestamp" if use_last_events else '',
        duplicate_event_flag=duplicate_event_flag,
        duplicate_event_delete='WHERE row_n = 1' if drop_duplicates is not None else '',
        count_events_filter='{} event_rank < {}'.format(
            'AND' if drop_duplicates is not None else 'WHERE',
            count_events) if count_events else '',
        rand_select_start='SELECT user_pseudo_id, RAND({}) AS random_value FROM ('.format(
            random_seed if random_seed else '') if random_user_limit else '',
        rand_select_end='ORDER BY user_pseudo_id) ORDER BY random_value LIMIT {}'.format(
            random_user_limit) if random_user_limit else ''
    )

    result = download_bq(client=client, query=new_users_query, job_config=job_config, group_name=group_name,
                         return_dataframe=return_dataframe, return_only_query=return_only_query,
                         hide_progress_bar=hide_progress_bar, progress_bar_min_interval=progress_bar_min_interval)
    return result
