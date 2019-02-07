import pandas as pd
import numpy as np
from tqdm import tqdm


def _calc_weights(counts, target_mechanics, mechanics_events):
    counts[target_mechanics] = counts.event_name.isin(mechanics_events) * counts.event_count
    return counts


def _diff(x):
    if x.shape[0] == 1:
        return x
    return x.iloc[0] - x.iloc[1]


def _get_mech_events(mechanics_events, mode):
    sub_mech = mechanics_events[mechanics_events['mode'] == mode]
    x = sub_mech.groupby(['mechanics', 'target']).event_name.agg(set).reset_index()
    x = x.sort_values('target', ascending=False)
    mex = x.groupby('mechanics').event_name.agg(_diff).to_dict()
    return mex


def calc_all_norm_mech(data, mechanics_events, mode='session', duration_thresh=1, len_thresh=None):
    """
    Calculates weights of different mechanics in users` sessions
    :param data: clickstream data with columns `session` (rank of user`s session)
    :param mechanics_events: mapping of mechanic and its target events
    :param mode: if `session` then calculates weights over session, if `full` over full users story
    :param duration_thresh: duration in time threshold for technical (ping) session deletion
    :param len_thresh: number of events in session threshold for technical (ping) session deletion
    :return: session description with weights of each mechanic

    :type data: pd.DataFrame
    :type mechanics_events: Dict[str, List[str]]
    :type mode: str
    :type duration_thresh: float
    :type len_thresh: int
    :rtype: pd.DataFrame

    """
    counts = data.groupby(['user_pseudo_id', 'session', 'event_name']).size().rename('event_count')
    counts = counts.reset_index()
    mex = _get_mech_events(mechanics_events=mechanics_events, mode=mode)

    for target_mechanics, mechanics_events in mex.items():
        counts = _calc_weights(counts, target_mechanics, mechanics_events)

    counts = counts.drop('event_name', 1).groupby(['user_pseudo_id', 'session'], as_index=False).sum()

    tmp = data.groupby(['user_pseudo_id', 'session']).event_timestamp.max().rename('session_end').reset_index()
    counts = counts.merge(tmp, on=['user_pseudo_id', 'session'])
    tmp = data.groupby(['user_pseudo_id', 'session']).event_timestamp.min().rename('session_start').reset_index()
    counts = counts.merge(tmp, on=['user_pseudo_id', 'session'])

    
    counts['session_duration'] = (counts.session_end - counts.session_start) / np.timedelta64(1, 's')
    if duration_thresh is not None:
        counts = counts.loc[counts.session_duration >= duration_thresh].copy()
    if len_thresh is not None:
        counts = counts.loc[counts.event_count >= len_thresh].copy()
    norm = (counts[list(mex.keys())].max(axis=1).values + 1e-20).reshape(-1, 1)
    counts[list(mex.keys())] = counts[list(mex.keys())].values / norm
    return counts


def _calc_stats(data, target):
    if type(target) is str:
        data['is_target'] = data.event_name == target
    else:
        data['is_target'] = data.event_name.isin(target)
    grouped = data.groupby('user_ses').agg({'is_target': 'sum', 'event_name': 'count'})
    grouped['freq'] = grouped.is_target / grouped.event_name
    return grouped.reset_index()


def _get_anom(src, data, q=.99, mode='top', q2=.99):
    if mode == 'top':
        thresh = np.quantile(data.freq.values, q=q)
        users = data[data.freq >= thresh].user_ses.values
        return set(src[src.user_ses.isin(users)].event_name)
    else:
        df = data.freq == 0
        if any(df):
            users = data[df == 0].user_ses.values
        else:
            return {}
        tmp = src[src.user_ses.isin(users)]
        return _top_event_loosers(tmp, q2)


def _top_event_loosers(tmp, q=.99):
    if tmp.shape[0] == 0:
        return set()
    nuq = tmp.groupby('user_ses').event_name.count()
    thresh = np.quantile(nuq.values, q=q)
    users = nuq[nuq >= thresh].index.values
    return set(tmp[tmp.user_ses.isin(users)].event_name)


def _get_clean_event_list(data, target, q=.99, q2=.99, session_mode=True):
    if session_mode:
        data['user_ses'] = data['user_pseudo_id'] + data['session'].astype(str)
    else:
        data['user_ses'] = data['user_pseudo_id']
    res = _calc_stats(data.copy(), target)
    top = _get_anom(data, res, q)
    los = _get_anom(data, res, q2=q2, mode='los')
    return list(top - los), top, los


def _get_df_results(top, los, mode):
    res_df_top = pd.DataFrame(list(top), columns=['event_name'])
    res_df_top['target'] = True
    res_df_los = pd.DataFrame(list(los), columns=['event_name'])
    res_df_los['target'] = False
    res_df = res_df_top.append(res_df_los, ignore_index=True)
    res_df['mode'] = mode
    return res_df.reset_index(drop=True)


def _build_df(data, target, mex, q=.99, q2=.99):
    res = pd.DataFrame(columns=['event_name', 'target', 'mode'])
    if 'session' in data.columns:
        good, top, los = _get_clean_event_list(data, target, q=q, q2=q2, session_mode=True)
        res = _get_df_results(top, los, 'session')
    good, top, los = _get_clean_event_list(data, target, q=.99, q2=.99, session_mode=False)
    res = res.append(_get_df_results(top, los, 'full'), ignore_index=True)
    res['mechanics'] = mex
    return res.reset_index(drop=True)


def mechanics_enrichment(data, mechanics, q=.99, q2=.99):
    """
    Enrich list of events specific for mechanic

    :param data: clickstream data with columns `session` (rank of user`s session)
    :param mechanics: table with description in form `['id', 'Events']`,
        where `id` is mechanic name and `Events` contains target events specific for that `mechanics`
    :param q: quantile for frequency of target events
    :param q2: quantile for frequency of target events of other mechanic
    :return: mapping of mechanic and its target events

    :param data: pd.DataFrame
    :param mechanics: pd.DataFrame
    :param q: float in interval (0, 1)
    :param q2: float in interval (0, 1)
    :rtype: Dict[str, List[str]]
    """
    mechanics_map = mechanics.groupby('id').Events.agg(list).to_dict()
    mechanics_events = []
    for key, val in tqdm(mechanics_map.items(), total=len(mechanics_map)):
        mechanics_events.append(_build_df(data, val, key, q=q, q2=q2))

    mechanics_events = pd.concat(mechanics_events)
    return mechanics_events
