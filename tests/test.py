import pandas as pd
from pandas.testing import assert_frame_equal
import unittest
import warnings
from retentioneering import analysis


class TestAnalysisFunctions(unittest.TestCase):

    def test_calculate_frequency_map(self):
        df_test = pd.DataFrame({
            'event_name': ['A', 'B|C', 'D', 'A', 'E'], 
            'event_timestamp': [0, 1, 4, 2, 3], 
            'user_pseudo_id': ['Z', 'Z', 'Z', 'Z', 'Z']})
        df_result_test = pd.DataFrame(
            {'A': [2], 'B|C': [1], 'E': [1]}, 
            index=pd.Index(['Z'], name='user_pseudo_id'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PendingDeprecationWarning)
            df_result = analysis.calculate.calculate_frequency_map(
                df_test, settings={}, make_plot=False, target_events=['D'])
        assert_frame_equal(df_result, df_result_test)

    def test_clustering(self):
        df = pd.read_csv('../example_datasets/train.csv')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PendingDeprecationWarning)
            countmap = analysis.calculate.calculate_frequency_map(df, settings={}, make_plot=False)
        users_clusters = analysis.cluster.cluster_users(countmap, n_clusters=3)
        self.assertEqual(users_clusters.cluster.nunique(), 3)
        users_clusters = analysis.cluster.cluster_users(countmap, n_clusters=5)
        self.assertEqual(users_clusters.cluster.nunique(), 5)
        cluster_stats = analysis.cluster.calculate_cluster_stats(
            df, users_clusters, {}, target_events=('lost', 'passed'), 
            plot_count=4, figsize=(12, 6), save=False)
        self.assertEqual(cluster_stats.shape, (4, 2))


if __name__ == '__main__':
    unittest.main()