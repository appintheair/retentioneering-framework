from __future__ import print_function
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from retentioneering.analysis.utils import _check_folder, get_all_agg
from retentioneering.visualization import plot


def _str_agg(x):
    return ' '.join(x)


def create_filter(data, n_folds=None):
    """
    Creates interested events filter with histogram

    :param data: data from BQ or your own (clickstream). Should have at least three columns: `event_name`,
            `event_timestamp` and `user_pseudo_id`
    :param n_folds: number of folds in histogram (it affects how far technical events should be from users` events)
    :type data: pd.DataFrame
    :type n_folds: int
    :return: set
    """
    all_events = set(data.event_name)
    x = data.groupby('event_name').user_pseudo_id.count()
    if n_folds is None:
        n_folds = (x.shape[0] // 200) + 1
    hist = np.histogram(x, bins=n_folds)
    while (hist[0] == 0).sum() > 0:
        x = x[x < hist[1][n_folds - 2]]
        x = x[x > hist[1][0]]
        hist = np.histogram(x, bins=n_folds)
    return all_events - set(x.index)


class Model:
    """
    Base model for classification
    """

    def __init__(self, data, target_event, settings, event_filter=None,
                 n_start_events=None, emb_type='tf-idf', ngram_range=(1, 3),
                 emb_dims=None, embedder=None):
        """
        Inits users classifier

        :param data: data from BQ or your own (clickstream). Should have at least three columns: `event_name`,
            `event_timestamp` and `user_pseudo_id`
        :param target_event: name of event which signalize target function
            (e.g. for prediction of lost users it'll be `lost`)
        :param settings: experiment config (can be empty dict here)
        :param event_filter: list of events that is wanted to use in analysis
        :param n_start_events: length of users trajectory from start
        :param emb_type: type of embedding (now only `tf-idf` is supported)
        :param ngram_range:
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used.
        :param emb_dims: it is needed only when use emb_type different from `tf-idf`
        :param embedder: pretrained model for event embedding (now only `tf-idf` is supported)

        :type data: pd.DataFrame
        :type target_event: str
        :type settings: dict
        :type event_filter: list or other iterable
        :type n_start_events: int
        :type emb_type: str
        :type ngram_range: tuple[int]
        :type emb_dims: int
        :type embedder: model or None
        """

        self._source_data = data
        self.data = self._prepare_dataset(data, target_event, event_filter, n_start_events)
        self.features = self.data.event_name.values
        self.target = self.data.target.values
        self.users = self.data.user_pseudo_id.values
        self.emb_type = emb_type
        self._embedder = embedder
        self.ngram_range = ngram_range
        self.emb_dims = emb_dims
        self.roc_auc_score = None
        self.average_precision_score = None
        self.roc_c = None
        self.prec_rec = None
        self._check_folder(settings)
        self._model_type = None
        self.model = None

        if embedder is not None:
            self._fit_vec = False
        else:
            self._fit_vec = True

    def _get_vectors(self, sample):
        if self._fit_vec:
            self._fit_vectors(sample)
        return self._embedder.transform(sample)

    def _fit_vectors(self, sample):
        if self.emb_type == 'tf-idf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(ngram_range=self.ngram_range)
            self._embedder = tfidf.fit(sample)
            self._fit_vec = False

    @staticmethod
    def _prepare_dataset(df, target_event=None, event_filter=None, n_start_events=None):
        if event_filter is not None:
            df = df[df.event_name.isin(event_filter)]
        df = df.sort_values('event_timestamp')
        train = df.groupby('user_pseudo_id').event_name.agg(_str_agg)
        train = train.reset_index(None)
        train.event_name = train.event_name.apply(lambda x: x.split())
        if target_event is not None:
            train['target'] = train.event_name.apply(lambda x: x[-1] == target_event)
            train.event_name = train.event_name.apply(lambda x: x[:-1])
        else:
            train.event_name = train.event_name.apply(lambda x: x)
        if n_start_events:
            train.event_name = train.event_name.apply(lambda x: ' '.join(x[:n_start_events]))
        else:
            train.event_name = train.event_name.apply(lambda x: ' '.join(x))
            return train

    def _prepare_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        x_train_vec = self._get_vectors(x_train)
        x_test_vec = self._get_vectors(x_test)
        return x_train_vec, x_test_vec, y_train, y_test

    def _validate(self, x_test_vec, y_test):
        preds = self.model.predict_proba(x_test_vec)
        self.roc_auc_score = roc_auc_score(y_test, preds[:, 1])
        self.average_precision_score = average_precision_score(y_test, preds[:, 1])
        print('ROC-AUC: {:.4f}'.format(self.roc_auc_score))
        print('PR-AUC: {:.4f}'.format(self.average_precision_score))
        self.roc_c = roc_curve(y_test, preds[:, 1])
        self.prec_rec = precision_recall_curve(y_test, preds[:, 1])
        self.plot()

    def fit_model(self, model_type='logit'):
        """
        Fits classifier

        :param model_type: type of model (now only logit is supported)
        :return: None
        :type model_type: str
        """
        self._model_type = model_type
        x_train_vec, x_test_vec, y_train, y_test = self._prepare_data()
        if model_type == 'logit':
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(penalty='l1')
            lr.fit(x_train_vec, y_train)
            self.model = lr
        self._validate(x_test_vec, y_test)

    def predict_proba(self, sample):
        """
        Predicts probability of sample

        :param sample: sample of users vectorized tracks (e.g. with `tf-idf` transform)
        :return: probabilities of different classes as list of
            `[not target_event probability, target_event probability]`
        :type sample: np.ndarray or pd.DataFrame
        :rtype: np.ndarray
        """
        return self.model.predict_proba(sample)

    def build_important_track(self):
        """
        Finds the most important tracks for definition of target_event

        :return: most important edges in graph
        :rtype: pd.DataFrame
        """
        if self._model_type == 'logit':
            imp = self.model.coef_
        else:
            raise NotImplementedError('Sorry, but only logit available in model type')
        if self.emb_type == 'tf-idf':
            imp = self._embedder.inverse_transform(imp)[0]
        edges = []
        for i in imp:
            j = i.split()
            if len(j) == 2:
                edges.append(j)
            elif len(j) > 2:
                for k in range(1, len(j)):
                    edges.append([j[k - 1], j[k]])
            elif len(j) == 1:
                edges.append([j[0], None])
        return pd.DataFrame(edges).drop_duplicates()

    @staticmethod
    def _get_tsne(sample):
        return TSNE().fit_transform(sample.todense())

    def plot_projections(self, sample=None, target=None, ids=None):
        """
        Plots tsne projection of users trajectories

        :param sample: sample of trajectories
        :param target: by which column data should be splitted
            (if target is None then probabilities of `target_event` is highlighted)
        :param ids: list of user_ids for visualization in plot
        :type sample: np.ndarray of pd.DataFrame
        :type target: str
        :type ids: list or other iterable

        :return: None
        """
        if sample is None:
            self._plot_proj_sample(self.features, self.target, self.users)
        else:
            self._plot_proj_sample(sample, target, ids)

    def _plot_proj_sample(self, sample, target=None, ids=None):
        pre_vec = self._get_vectors(sample)
        vec = self._get_tsne(pre_vec)
        self._cached_tsne = vec
        self._tsne_sample = sample
        if ids is not None:
            self._tsne_users = ids

        if ids is not None:
            txt = np.array(['user_id: {}'.format(i) for i in ids])
        else:
            txt = None

        if target is not None:
            figs = []
            for i in np.unique(target):
                figs.append(
                    go.Scatter(
                        x=vec[target == i][:, 0],
                        y=vec[target == i][:, 1],
                        name=str(i),
                        mode='markers',
                        text=list(txt[target == i]) if txt is not None else txt,
                    )
                )
        else:
            probs = self.predict_proba(pre_vec)[:, 1]
            if txt is not None:
                txt = [i + ',\n prob: {}'.format(j) for i, j in zip(txt, probs)]
            else:
                txt = ['prob: {}'.format(i) for i in probs]

            figs = [go.Scatter(
                x=vec[:, 0],
                y=vec[:, 1],
                mode='markers',
                marker=dict(
                    color=probs,
                    colorscale='YlGnBu',
                    showscale=True
                ),
                text=txt, )]

        py.init_notebook_mode()
        py.iplot(figs)
        filename = os.path.join(
            self.export_folder, 'tsne_{}.html'.format(datetime.now().strftime('%Y-%m-%dT%H_%M_%S_%f')))
        py.plot(figs, filename=filename, auto_open=False)

    def _get_data_from_plot(self, bbox):
        bbox = np.array(bbox)
        left = bbox[:, 0].min()
        right = bbox[:, 0].max()
        up = bbox[:, 1].max()
        bot = bbox[:, 1].min()
        res = self._cached_tsne
        fil = (res[:, 1] < up) & (res[:, 1] > bot) & (res[:, 0] < right) & (res[:, 0] > left)
        users = self._tsne_users[fil]
        data = self._source_data[self._source_data.user_pseudo_id.isin(users)]
        return data

    def plot_cluster_track(self, bbox):
        """
        Plots graph for users in selected area

        :param bbox: coordinates of top-left and bottom-right angles of area
        :type bbox: List[List[float]]
        :return: None
        """
        data = self._get_data_from_plot(bbox)
        data_agg = get_all_agg(data, ['trans_count'])
        plot.plot_graph(data_agg, 'trans_count', {'export_folder': self.export_folder})

    def _check_folder(self, settings):
        settings = _check_folder(settings)
        self.export_folder = settings['export_folder']

    def plot(self):
        """
        Plot metrics of model

        :return: None
        """
        f, ax = plt.subplots(1, 2)
        f.set_size_inches(15, 5)
        ax[0].plot(self.roc_c[0], self.roc_c[1])
        ax[0].set_title('ROC curve')
        ax[1].plot(self.prec_rec[1], self.prec_rec[0])
        ax[1].set_title('Precision-Recall curve')
        plt.grid()
        filename = os.path.join(self.export_folder, 'scores {}.png'.format(datetime.now()))
        f.savefig(filename)
