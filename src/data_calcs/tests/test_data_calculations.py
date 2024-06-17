'''
Created on Apr 27, 2024

@author: paepcke
'''

from data_calcs.data_calculations import (
    DataCalcs,
    PerplexitySearchResult)
from data_calcs.daytime_file_selection import (
    DaytimeFileSelector)
from data_calcs.utils import (
    Utils,
    TimeGranularity)
from datetime import (
    datetime)
from logging_service.logging_service import (
    LoggingService)
from pandas.testing import (
    assert_frame_equal)
from sklearn.cluster._kmeans import (
    KMeans)
from sklearn.datasets import (
    make_blobs)
from tempfile import (
    TemporaryDirectory)
from unittest.mock import (
    MagicMock)
import numpy as np
import os
import pandas as pd
import unittest

TEST_ALL = True
#TEST_ALL = False

class DataPrepTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.log = LoggingService()
        cls.log.warn = MagicMock()

    def setUp(self):
        self.create_test_files()


    def tearDown(self):
        self.tmpdir.cleanup()

    # --------------------- Tests ------------------
    
    #------------------------------------
    # test_sort_by_variance
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_sort_by_variance(self):
        
        jumbled = ['HiFtoKnAmp', 'AmpEndLn60ExpC', 'LdgToFcAmp',
                   'HiFtoUpprKnAmp','AmpStartLn60ExpC','MaxSegLnght'] 

        sorted_measures = DataCalcs.sort_by_variance(jumbled)

        expected = ['LdgToFcAmp','HiFtoUpprKnAmp','HiFtoKnAmp',
                    'MaxSegLnght','AmpStartLn60ExpC','AmpEndLn60ExpC']
        
        self.assertListEqual(sorted_measures, expected)

    #------------------------------------
    # test_measures_by_var_rank
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_measures_by_var_rank(self):

        # Want only two of three columns, sorted by
        # variance rank:
        
        df = pd.DataFrame([{'HiFtoKnAmp' : 10,  'LdgToFcAmp' : 20,  'HiFtoUpprKnAmp' : 30},
                           {'HiFtoKnAmp' : 100, 'LdgToFcAmp' : 200, 'HiFtoUpprKnAmp' : 300}])
        df_new = DataCalcs.measures_by_var_rank(df, 2)
        
        expected = pd.DataFrame([{'LdgToFcAmp' : 20,  'HiFtoUpprKnAmp' : 30},
                                 {'LdgToFcAmp' : 200, 'HiFtoUpprKnAmp' : 300}])

        assert_frame_equal(df_new, expected)
        
        # Make minimum rank higher than number
        # of cols in given df:
        df_new = DataCalcs.measures_by_var_rank(df, 15)
        expected_cols = DataCalcs.sort_by_variance(df.columns)
        expected = df[expected_cols]
        assert_frame_equal(df_new, expected)

    #------------------------------------
    # test_constructor
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_constructor(self):
        
        # Measures and inference roots are both the test tmpdir: 
        dp = DataCalcs(self.tmpdir.name, self.tmpdir.name)
        # dp's fid2split_dict should be similar to:
        #   {100: '/tmp/data_prep_6057jcgj/split15.feather',
        #    110: '/tmp/data_prep_6057jcgj/split15.feather',
        #    120: '/tmp/data_prep_6057jcgj/split15.feather',
        #    11: '/tmp/data_prep_6057jcgj/split4.feather',
        #    12: '/tmp/data_prep_6057jcgj/split4.feather',
        #    13: '/tmp/data_prep_6057jcgj/split4.feather'
        #    }
        
        # expected = {100: f'{self.tmpdir.name}/split15.feather',
        #             110: f'{self.tmpdir.name}/split15.feather',
        #             120: f'{self.tmpdir.name}/split15.feather',
        #             11:  f'{self.tmpdir.name}/split4.feather',
        #             12:  f'{self.tmpdir.name}/split4.feather',
        #             13:  f'{self.tmpdir.name}/split4.feather'}
        # self.assertDictEqual(dp.fid2split_dict, expected)
        
        # Check the split_fpaths being a dict mapping
        # a running int (split_id) to the full split path:
        expected = {
            0 : os.path.join(self.tmpdir.name, 'split15.feather'), 
            1 : os.path.join(self.tmpdir.name, 'split4.feather'), 
            }
        self.assertDictEqual(dp.split_fpaths, expected)
        
    #------------------------------------
    # test_measures_from_fid
    #-------------------
    
    #@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test_measures_from_fid(self):
    #
    #     dp = DataCalcs(self.tmpdir.name, self.tmpdir.name)
    #
    #     measures = dp.measures_from_fid(11)
    #     # Ground truth:
    #     df_truth = pd.read_feather(f'{self.tmpdir.name}/split15.feather')
    #     df_truth.index = df_truth.file_id
    #
    #     #    TimeInFile          -0.41427
    #     #    PrecedingIntrvl     -0.41427
    #     #    CallsPerSec         -0.41427
    #     #    file_id            100.00000
    #     #    Name: 100, dtype: float64
    #
    #     expected = df_truth.loc[11]
    #     pd.testing.assert_series_equal(measures, expected)

    #------------------------------------
    # test_add_recording_datetime
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_recording_datetime(self):
        dp = DataCalcs(self.tmpdir.name, self.tmpdir.name)
        df = self.tst_df_large
        new_df = dp.add_recording_datetime(df)
        
        # Since mods are done inplace, the new and old
        # dfs should be the same:
        pd.testing.assert_frame_equal(new_df, df)
        
        all([type(val) == datetime
             for val 
             in new_df.rec_datetime 
             ])
        
        # Check one of the datetimes:
        dt0    = df.rec_datetime.iloc[0]
        
        tz = DaytimeFileSelector().timezone
        # .wav file: barn1_D20220205T192049m784-HiF.wav
        true_dt = datetime(2022, 2, 5, 19, 20, 49, tzinfo=tz)
        self.assertEqual(dt0, true_dt)

        # Check daylight determination: Should be
        # false, since 19:20 in February is after dark:
        is_day0 = df.is_daytime.iloc[0]
        self.assertFalse(is_day0)
        
        # Second chirp's file name:
        #   'barn1_D20220205T140541m567-Myca-Myca.wav', so: afternoon
        is_day1 = df.is_daytime.iloc[1]
        self.assertTrue(is_day1)
        

    #------------------------------------
    # test_feather_input
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_feather_input(self):
        df = pd.read_feather(self.dst_file4)
        pd.testing.assert_frame_equal(df, self.tst_df4)

    #------------------------------------
    # test_run_tsne
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_run_tsne(self):
        dp = DataCalcs(self.tmpdir.name, self.tmpdir.name)
        
        # Get like
        #                  tsne_x        tsne_y
        #     file_id                          
        #     11       -38.101448 -1.185629e-05
        #     12      -972.111694  1.796527e-05
        #     13       897.224304 -5.270238e-07        
        #
        # though the numbers won't be the same.
        
        # All defaults, no key column:
        tsne_df = dp.run_tsne(self.tst_df4) # num_points, num_dims, point_id_col, perplexity, sort_by_bat_variance)
        # We did not asked that any non-measure columns 
        # be retained, so result Tsne df should be like:
        #        tsne_x  tsne_y  PrecedingIntrvl  CallsPerSec
        # 0 -2421.12085     0.0              100         1000
        # 1     0.00000     0.0              200         2000
        # 2  2421.12085     0.0              300         3000
        pd.testing.assert_index_equal(tsne_df.index, pd.RangeIndex(0,3))
        self.assertEqual(len(tsne_df), 3)
        
        # Carry the 'file_id' coumn of the original df in
        # to the final tsne_df: 
        tsne_df = dp.run_tsne(self.tst_df4, cols_to_keep=['file_id']) # num_points, num_dims, point_id_col, perplexity, sort_by_bat_variance)
        
        expected = pd.RangeIndex(0,3)
        pd.testing.assert_index_equal(tsne_df.index, expected)
        self.assertEqual(tsne_df.ndim, 2)
        expected = pd.Index(['tsne_x', 'tsne_y','PrecedingIntrvl','CallsPerSec','file_id'])
        pd.testing.assert_index_equal(tsne_df.columns, expected)
        
        # Keep two columns, and specify a key column to be
        # copied to the index:
        tsne_df = dp.run_tsne(self.tst_df4, cols_to_keep=['file_id', 'TimeInFile']) # num_points, num_dims, point_id_col, perplexity, sort_by_bat_variance)
        self.assertEqual(tsne_df.ndim, 2)
        expected = pd.Index(['tsne_x', 'tsne_y', 'PrecedingIntrvl', 'CallsPerSec', 'file_id', 'TimeInFile'])
        # Columns might be out of order:
        self.assertSetEqual(set(tsne_df.columns), set(expected))
        
        # Now try TSNE on a df with recording time and daylight information:
        df = dp.add_recording_datetime(self.tst_df_large).copy()
        tsne_df = dp.run_tsne(df, cols_to_keep=['rec_datetime', 'is_daytime'])
        
        expected = ['tsne_x', 'tsne_y', 'PrecedingIntrvl', 'CallsPerSec', 'is_daytime', 'rec_datetime']
        self.assertSetEqual(set(tsne_df.columns), set(expected))
        self.assertEqual(len(tsne_df), len(df))
        
        # Check is_daytime col: only second row is True:
        expected = [False]*len(df)
        expected[1] = True
        self.assertListEqual(list(tsne_df['is_daytime']), expected)
        
        # Check one of the datetimes:
        dt0    = tsne_df.rec_datetime.iloc[0]
        
        tz = DaytimeFileSelector().timezone
        # .wav file: barn1_D20220205T192049m784-HiF.wav
        true_dt = datetime(2022, 2, 5, 19, 20, 49, tzinfo=tz)
        self.assertEqual(dt0, true_dt)
        
    #------------------------------------
    # test_cluster_tsne
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_cluster_tsne(self):

        dp = DataCalcs(self.tmpdir.name, self.tmpdir.name)
        
        # Get like
        #                  tsne_x        tsne_y
        #     file_id                          
        #     11       -38.101448 -1.185629e-05
        #     12      -972.111694  1.796527e-05
        #     13       897.224304 -5.270238e-07        
        #
        # though the numbers won't be the same:
        tsne_df = dp.run_tsne(self.tst_df4, cols_to_keep=['file_id']) # num_points, num_dims, point_id_col, perplexity, sort_by_bat_variance)
        
        
        # Provide no specific n_clusters, triggering 
        # a search over multiple n_clusters. But only allow 
        # the range of n_clusters values to be 2:
        cluster_result = dp.cluster_tsne(tsne_df, n_clusters=None, cluster_range=range(2,3))
        self.assertEqual(cluster_result.best_n_clusters, 2)
        # The kmeans entry of result should be a KMeans instance.
        # We test that by checking its n_clusters attribute.
        # An isinstance() would require importing sklearn:
        self.assertEqual(cluster_result.best_kmeans.n_clusters, 2)
        
        # Specify a particular n_clusters:
        cluster_result = dp.cluster_tsne(tsne_df, n_clusters=2)
        # Since we locked down the number of clusters for
        # the kmeans, the two 'best-x' values would not be meaningful:
        self.assertIsNone(cluster_result.best_n_clusters)
        self.assertIsNone(cluster_result.best_silhouette)
        
        self.assertEqual(cluster_result.best_kmeans.n_clusters, 2)

    #------------------------------------
    # test_find_optimal_tsne_clustering
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_find_optimal_tsne_clustering(self):
        
        dp = DataCalcs(self.tmpdir.name, self.tmpdir.name)
        # Use this test df:
        #        TimeInFile  PrecedingIntrvl  CallsPerSec  file_id
        #     0          10              100         1000       11
        #     1          20              200         2000       12
        #     2          30              300         3000       13
        
        
        df = self.tst_df4
        # Should give warning that it does not test 
        # several perplexities, because they are larger than num samples:
        dp.find_optimal_tsne_clustering(df, cols_to_keep=None)
        self.log.warn.assert_called()
        
        # Now use a larger df:
        df = self.tst_df_large.copy()
        
        result = dp.find_optimal_tsne_clustering(df, cols_to_keep=None)
        
        # Optimal perplexity must be 5.0, since that's the
        # only one that the length of our data df supports:
        self.assertEqual(result.optimal_perplexity, 5.0)
        expected_tsne_data_pts = len(df)
        expected_num_cols      = 4
        expected_cell_types    = np.float64
        
        # Only one tsne_df available, due to limited data df.
        # Get that df:
        tsne_df = result.tsne_df(5.0)
    
        # Number of points    
        self.assertEqual(len(tsne_df), expected_tsne_data_pts)
        self.assertEqual(len(tsne_df.columns), expected_num_cols)

        # All tsne df of float type:
        self.assertTrue(all([type(cell) == expected_cell_types
                             for cell
                             in tsne_df.to_numpy().flatten()]))
        
        # Silhouette df expected like:
        #
        #                     N_CLUSTERS
        #              2            4               10
        # PERPLEXITY
        #     5.0   silhouette  silhouette      silhouette
        #    10.0   silhouette  silhouette      silhouette
        #
        # Though only one row:
        sils_df = result.silhouettes_df
        self.assertEqual(sils_df.index.name, 'Perplexities')
        # One row:
        self.assertEqual(len(sils_df), 1)
        # ... therefore one index entry, for the one perplexity we have:
        self.assertEqual(sils_df.index[0], 5.0)
        # One column that's n_clusters of 2:
        self.assertEqual(sils_df.columns[0], 2)
        
        
        # Clustering:
        perp5_clusters = result.clustering_result(5.0)
        # Sum of cluster populations should equal
        # number of points:
        pop_all = sum(perp5_clusters.cluster_pops)
        self.assertEqual(pop_all, len(df))
        
        # Number of KMeans objects in the cluster result for
        # perplexity 5.0 should be the same number of columns
        # in the silhouettes df. That's b/c each column there is for one
        # n_clusters:
        
        num_n_clusters  = len(sils_df.columns)
        num_kmeans_objs = len(perp5_clusters._kmeans_objs)
        self.assertEqual(num_n_clusters, num_kmeans_objs)

    #------------------------------------
    # test_PerplexitySearchResult_json
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_PerplexitySearchResult_json(self):
        dp = DataCalcs(self.tmpdir.name, self.tmpdir.name)

        df = self.tst_df_large.copy()
        # Should give warning that it does not test 
        # several perplexities, because they are larger than num samples:
        perpl_srch_res = dp.find_optimal_tsne_clustering(df, cols_to_keep=None)
        # Get like:
        #    {'optimal_perplexity': 5.0,
        #     'optimal_n_clusters': 2,
        #     'cluster_populations': [5, 4],
        #     'cluster_centers': [[55.93270, -52.38911], [41.893125, 119.783]],
        #     'cluster_labels': [1, 1, 1, 1, 0, 0, 0, 0, 0],
        #     'best_silhouette': 0.517001748085022,
        #     'tsne_df': '{"tsne_x":{"0":72.0819854736,"1":17.3076343536,...}}'
        
        jstr = perpl_srch_res.to_json()
        
        # Turn the JSON into a dict that will have the
        # most important state of the original PerplexitySearchResult
        jdict = PerplexitySearchResult.read_json(jstr)
        self.verify_perplexity_summary(jdict, df)
        
        # Same for saving the JSON to a file, and retrieving it:
        jdest_file = os.path.join(self.tmpdir.name, 'perpl_srch_res.json')
        with open(jdest_file, 'w') as fd:
            perpl_srch_res.to_json(fd)
        # Get it back:
        jdict = PerplexitySearchResult.read_json(jdest_file)
        self.verify_perplexity_summary(jdict, df)        


    #------------------------------------
    # test_make_chirp_sample_file
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_make_chirp_sample_file(self):
        
        # We'll use three test dfs, two with 4 rows,
        # and one with 2 rows:
        
        tst_dfs    = {0 : self.split_df1, 
                      1 : self.split_df2, 
                      2 : self.split_df3}
        total_rows = sum([len(df) for df in [self.split_df1,self.split_df2,self.split_df3]])
        cols       = self.split_df1.columns
        
        dp = DataCalcs(self.tmpdir.name, self.tmpdir.name)
        
        # Get fewer samples than are available in the dfs in total:
        
        samples_wanted = 5
        samples = dp.make_chirp_sample_file(samples_wanted, unittests=tst_dfs)
        
        # Got correct number of samples?
        self.assertEqual(len(samples['df']), samples_wanted)
        # Same cols as the originals, plus augmentations:
        new_cols_expected = cols.append(pd.Index(['rec_datetime', 'is_daytime', 'species', 'sin_hr', 'cos_hr',
       'sin_day', 'cos_day', 'sin_month', 'cos_month', 'sin_year', 'cos_year']))
        
        pd.testing.assert_index_equal(samples['df'].columns, new_cols_expected)

        # Get exactly as many samles as are in the the dfs:

        samples_wanted = total_rows 
        samples = dp.make_chirp_sample_file(samples_wanted, unittests=tst_dfs)
        
        # Got correct number of samples?
        self.assertEqual(len(samples['df']), samples_wanted)
        # Same cols as the originals:
        pd.testing.assert_index_equal(samples['df'].columns, new_cols_expected)
                        
        # Ask for more samples than are available:
        samples_wanted = 11
        with self.assertRaises(ValueError):
            samples = dp.make_chirp_sample_file(samples_wanted, unittests=tst_dfs)


    #------------------------------------
    # test__sin_cos_cache
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__sin_cos_cache(self):
        
        dc = DataCalcs(self.tmpdir.name, self.tmpdir.name)
        dt = datetime(2024, 1, 1, 1, 1, 1)
        # The numeric, rounded tuples in the asserts below
        # come from the following _trig result. They were
        # manually transferred. The _trig is otherwise
        # unused:
        _trig = dc._sin_cos_cache(dt)
        
        
        self.assertEqual(tuple(map(lambda trig: round(trig, 4), 
                                   Utils.cycle_time(1, TimeGranularity.HOURS))),
                         (0.2588, 0.9659))
        self.assertEqual(tuple(map(lambda trig: round(trig, 4),
                                   Utils.cycle_time(1, TimeGranularity.DAYS))),
                                   (0.2079, 0.9781))
        self.assertEqual(tuple(map(lambda trig: round(trig, 4),
                                   Utils.cycle_time(1, TimeGranularity.MONTHS))),                           
                                   (0.5, 0.8660))
        self.assertEqual(tuple(map(lambda trig: round(trig, 4),
                                   Utils.cycle_time(1, TimeGranularity.YEARS))),
                                   (0.5878, 0.8090))
        
    #------------------------------------
    # test__add_trig_cols
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__add_trig_cols(self):
        
        dc = DataCalcs(self.tmpdir.name, self.tmpdir.name)
        df_start = pd.DataFrame(
            {'foo' : [1,2,3,4],#Yr  Mn Dy Hr mn  Sec
             'dt'  : [datetime(2023, 1, 1, 1, 1, 1),
                      datetime(2023, 1, 1, 18, 1, 1),
                      datetime(2023, 1, 23, 1, 1, 1),
                      datetime(2023, 9, 1, 1, 1, 1),
                      ]
             })
        df = dc._add_trig_cols(df_start, 'dt')
        # Got all the cols?
        cols = ['foo', 'dt',
                'sin_hr', 'cos_hr', 
                'sin_day', 'cos_day', 
                'sin_month', 'cos_month', 
                'sin_year', 'cos_year']
        # The 'list()' is needed b/c df.columns is of
        # type pd.Index, not list:               
        self.assertListEqual(list(df.columns), cols)
        self.assertEqual(len(df), len(df_start))
        
        # Just test one dt, since similar methods are
        # tested elsewhere:
        dt_check = df_start.dt.iloc[0]
        self.assertEqual((round(df.sin_hr[0], 4), 
                          round(df.cos_hr[0], 4)), 
                         tuple(map(lambda trig: round(trig, 4), 
                                   Utils.cycle_time(dt_check.hour, TimeGranularity.HOURS))))
         
        self.assertEqual((round(df.sin_day[0], 4),
                          round(df.cos_day[0], 4)),
                         tuple(map(lambda trig: round(trig, 4), 
                                   Utils.cycle_time(dt_check.day, TimeGranularity.DAYS))))
        
        self.assertEqual((round(df.sin_month[0], 4),
                          round(df.cos_month[0], 4)),
                         tuple(map(lambda trig: round(trig, 4), 
                                   Utils.cycle_time(dt_check.month, TimeGranularity.MONTHS))))

        self.assertEqual((round(df.sin_year[0], 4),
                          round(df.cos_year[0], 4)), 
                         tuple(map(lambda trig: round(trig, 4),
                                   Utils.cycle_time(dt_check.year, TimeGranularity.YEARS))))

    #------------------------------------
    # test_correlate_all_against_one
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_correlate_all_against_one(self):
        
        # First case: continuous vars against one continuous var:
        
         
        # Use self.tst_df4, corr all against CallsPerSec
        #    TimeInFile  PrecedingIntrvl  CallsPerSec  file_id
        # 0          10              100         1000       11
        # 1          20              200         2000       12
        # 2          30              300         3000       13
        
        corrs = DataCalcs.correlate_all_against_one(self.tst_df4, 'CallsPerSec')
        self.assertEqual(list(corrs.columns), ['Corr_all_against_CallsPerSec', 'p_value'])
        corr_col = corrs.Corr_all_against_CallsPerSec
        expected = pd.Series({'TimeInFile' : 1.0,
                              'PrecedingIntrvl' : 1.0,
                              'file_id'         : 1.0,
                              'chirp_idx'       : 0.3273
                              }, name='Corr_all_against_CallsPerSec')
        pd.testing.assert_series_equal(corr_col.round(4), expected)
        
        # Now, continuous vars against a dichotomous var:
        df = self.tst_df4.copy()
        df.file_id = [0, 1, 0]
        
        corrs = DataCalcs.correlate_all_against_one(df, 'file_id')
        self.assertEqual(list(corrs.columns), ['Corr_all_against_file_id', 'p_value'])
        
        corr_col = corrs.Corr_all_against_file_id.round(4)
        expected = pd.Series([0.0, 0.0, 0.0, 0.9449], 
                             name='Corr_all_against_file_id',
                             index=['TimeInFile', 'PrecedingIntrvl', 'CallsPerSec', 'chirp_idx']
                             )
        pd.testing.assert_series_equal(corr_col, expected)
        p_values = corrs.p_value.round(4)
        expected = pd.Series([1.0, 1.0, 1.0, 0.2123], name='p_value',
                             index=['TimeInFile', 'PrecedingIntrvl', 'CallsPerSec', 'chirp_idx']
                             )
        pd.testing.assert_series_equal(p_values, expected)

        # Now, continuous vars against a dichotomous var that is not 0 and 1:
        df = self.tst_df4.copy()
        df.file_id = [False, True, False]
        
        corrs = DataCalcs.correlate_all_against_one(df, 'file_id')
        self.assertEqual(list(corrs.columns), ['Corr_all_against_file_id', 'p_value'])
        
        corr_col = corrs.Corr_all_against_file_id.round(4)
        expected = pd.Series([0.0, 0.0, 0.0, 0.9449], 
                             name='Corr_all_against_file_id',
                             index=['TimeInFile', 'PrecedingIntrvl', 'CallsPerSec', 'chirp_idx']
                             )
        pd.testing.assert_series_equal(corr_col, expected)
        p_values = corrs.p_value.round(4)
        expected = pd.Series([1.0, 1.0, 1.0, 0.2123], name='p_value',
                             index=['TimeInFile', 'PrecedingIntrvl', 'CallsPerSec', 'chirp_idx']
                             )
        pd.testing.assert_series_equal(p_values, expected)
        
    #------------------------------------
    # test_find_optimal_k
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_find_optimal_k(self):

        n_features = 3
        X, _y = make_blobs(random_state=42, n_features=n_features)
        X_df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
        
        res_df = DataCalcs.find_optimal_k(X_df, range(2,5))
        expected_first_2_cols = pd.DataFrame({'k' : [2,3,4],
                                              'silhouette_score' : [0.804257,
                                                                    0.739392,
                                                                    0.597150
                                                                    ]
                                              })
        pd.testing.assert_frame_equal(res_df[['k', 'silhouette_score']], expected_first_2_cols)
        for row_idx in range(n_features):
            self.assertTrue(isinstance(res_df.iloc[row_idx, 2], KMeans)) 


    # ----------------------- Utilities ----------------

    #------------------------------------
    # verify_perplexity_summary
    #-------------------
    
    def verify_perplexity_summary(self, jdict, df):
        '''
        Given the result of PerplexitySearchResult.read_json()
        of a previously json-saved PerplexitySearchResult, verify
        that the result is OK. The original PerplexitySearchResult is
        assumed to have come from df.
        
        Throws assertion error if inconsistencies found:
        
        :param jdict: dict returned from PerplexitySearchResult.read_json(json_source)
        :type jdict: dict[str : any]
        :param df: dataframe from which PerplexitySearchResult was originally computed
        :type df: pd.DataFrame
        '''

        # The df used above only allows for one perplexity: 5.0:
        self.assertEqual(jdict['optimal_perplexity'], 5.0)
        # n_clusters should be an int between 2 and 8:
        n_clusters = jdict['optimal_n_clusters']
        self.assertIn(n_clusters, range(2,9))
        # Cluster populations should be as many ints as the number of clusters:
        self.assertEqual(len(jdict['cluster_populations']), n_clusters)
        # All should be ints:
        self.assertTrue(all([type(pop_num) == int for pop_num in jdict['cluster_populations']]))
        # There should be as many cluster centers as clusters:
        self.assertEqual(len(jdict['cluster_centers']), n_clusters)
        # Should have as many cluster labels as data points (i.e. rows in df)
        self.assertEqual(len(jdict['cluster_labels']), len(df))
        # Best silhouette should be a single float:
        self.assertEqual(type(jdict['best_silhouette']), float)
        # Tsne df should have as many rows as df:
        self.assertEqual(len(jdict['tsne_df']), len(df))
        

    #------------------------------------
    # test_distances
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_distances(self):
        
        # Series with Series:
        
        obj1 = pd.Series([10, 20, 30])
        obj2 = pd.Series([10, 20, 30])
        dist = DataCalcs.distances(obj1, obj2, metric='euclidean')
        expected = pd.Series([0.0])
        pd.testing.assert_series_equal(dist, expected)        

        obj1 = pd.Series([20, 30, 40])
        obj2 = pd.Series([10, 20, 30])
        
        dist = DataCalcs.distances(obj1, obj2, metric='euclidean')
        expected = pd.Series(np.sqrt(10**2 + 10**2 + 10**2))
        pd.testing.assert_series_equal(dist, expected)        

        # Series with df: want distance of series with each df row: 
        obj1   = pd.Series([20, 30, 40])
        obj2df = pd.DataFrame([[20, 30, 40], [10, 20, 30]])
        dist   = DataCalcs.distances(obj1, obj2df, metric='euclidean')
        expected = pd.Series([0, 17.320508])
        pd.testing.assert_series_equal(dist, expected)        
        
        # df with Series:
        obj1df = pd.DataFrame([[20, 30, 40], [10, 20, 30]])
        obj2   = pd.Series([20, 30, 40])
        dist   = DataCalcs.distances(obj1df, obj2, metric='euclidean')
        expected = pd.Series([0, 17.320508])
        pd.testing.assert_series_equal(dist, expected)        
              
        # df with df:
        obj1df = pd.DataFrame([[20, 30, 40], 
                               [10, 20, 30]])
        obj2df = pd.DataFrame([[20, 30, 40], 
                               [20, 30, 40]])
        dist   = DataCalcs.distances(obj1df, obj2df, metric='euclidean')
        expected = pd.Series([0, 17.320508])
        pd.testing.assert_series_equal(dist, expected)        

# ------------------------- Utilities ---------

    #------------------------------------
    # create_test_files
    #-------------------

    
    def create_test_files(self):
        '''
        Create two dfs:
            self.tst_df4
               TimeInFile  PrecedingIntrvl  CallsPerSec  file_id
            0          10              100         1000       11
            1          20              200         2000       12
            2          30              300         3000       13
            
            self.tst_df15
              TimeInFile PrecedingIntrvl CallsPerSec  file_id
            0       foo1            bar1        fum1      100
            1       foo2            bar2        fum2      110
            2       foo3            bar3        fum3      120

        and put both into separate files: 
            <tmpdir>/split4.feather
            <tmpdir>/split15.feather
        
        '''
        
        self.tmpdir = TemporaryDirectory(dir='/tmp', prefix='data_prep_')
        # self.feather_fd = NamedTemporaryFile(dir=self.tmpdir.name, 
        #                                      prefix='feather_test', 
        #                                      suffix='.feather',
        #                                      delete=False)

        # Build a small df, which will be split4:
        c1 = [10,20,30]
        c2 = [100,200,300]
        c3 = [1000,2000,3000]
        file_id = [11,12,13]
        chirp_idx = [1,4,2]
        
        self.tst_df4 = pd.DataFrame(
            {'TimeInFile'      : c1,
             'PrecedingIntrvl' : c2,
             'CallsPerSec'     : c3,
             'file_id'         : file_id,
             'chirp_idx'       : chirp_idx 
             })
        
        self.dst_file4 = os.path.join(self.tmpdir.name, 'split4.feather')
        self.tst_df4.to_feather(self.dst_file4)

        # Build another small df, which will be split15:
        c1 = ['foo1','foo2','foo3']
        c2 = ['bar1','bar2','bar3']
        c3 = ['fum1','fum2','fum3']
        file_id = [14,15,16]
        chirp_idx = [1,4,2]
        self.tst_df15 = pd.DataFrame(
            {'TimeInFile'      : c1,
             'PrecedingIntrvl' : c2,
             'CallsPerSec'     : c3,
             'file_id'         : file_id,
             'chirp_idx'       : chirp_idx 
             })
        self.dst_file15 = os.path.join(self.tmpdir.name, 'split15.feather')
        self.tst_df15.to_feather(self.dst_file15)
        
        self.tst_df_large = self.tst_df4.copy()
        # Add some rows to the df so that there are enough samples
        # for perplexities searches to make sense:
        for i in range(4, 10):
            # Last value is the file_id:
            #       TmInFil PredIntv ClsPSec  FID  ChirpIDX  
            new_row = [i*10, i*100,   i*1000, 10+i, i]
            self.tst_df_large.loc[i] = new_row 
        
        # Ensure the index is 0-10:
        self.tst_df_large.reset_index(drop=True, inplace=True)        
        
        # The file ID to .wav file recording name map:
        split_file_name_to_id = ('Filename,file_id\n'
                                 'barn1_D20220205T192049m784-HiF.wav,11\n'
                                 'barn1_D20220205T140541m567-Myca-Myca.wav,12\n'
                                 'barn1_D20220205T202248m042-HiF.wav,13\n'
                                 'barn1_D20220205T205824m469-Tabr-Tabr-Tabr.wav,14\n'
                                 'barn1_D20220205T211937m700-HiF.wav,15\n'
                                 'barn1_D20220205T231442m354-Myca-Myca.wav,16\n'
                                 'barn1_D20220205T235354m889-Tabr-Tabr-Tabr.wav,17\n'
                                 'barn1_D20220206T001144m425-Tabr-Tabr.wav,18\n'
                                 'barn1_D20220206T012049m898.wav,19')
        self.fname_to_id_file = os.path.join(self.tmpdir.name, 'split_filename_to_id.csv')
        with open(self.fname_to_id_file, 'w') as fd:
            fd.write(split_file_name_to_id)
            
        # For easy testing: a dict fid --> .wav file name:
        fname_id_pairs = [fname_comma_fid.split(',')
                          for fname_comma_fid
                          in split_file_name_to_id.split('\n')]
        
        self.fid_wav_fname_dict = {int(fid) : fname
                                   for fname, fid 
                                   in fname_id_pairs[1:] # Skip header 
                                   }
            
        # ________________________ Example split file exerpts -------------------
        
        cols = ['TimeInFile','PrecedingIntrvl','CallsPerSec','CallDuration','Fc','HiFreq','LowFreq','Bndwdth','FreqMaxPwr','PrcntMaxAmpDur','TimeFromMaxToFc','FreqKnee','PrcntKneeDur','StartF','EndF','DominantSlope','SlopeAtFc','StartSlope','EndSlope','SteepestSlope','LowestSlope','TotalSlope','HiFtoKnSlope','KneeToFcSlope','CummNmlzdSlp','HiFtoFcExpAmp','HiFtoFcDmp','KnToFcExpAmp','KnToFcDmp','HiFtoKnExpAmp','HiFtoKnDmp','FreqLedge','LedgeDuration','FreqCtr','FBak32dB','FFwd32dB','FBak20dB','FFwd20dB','FBak15dB','FFwd15dB','FBak5dB','FFwd5dB','Bndw32dB','Bndw20dB','Bndw15dB','Bndw5dB','DurOf32dB','DurOf20dB','DurOf15dB','DurOf5dB','Amp1stQrtl','Amp2ndQrtl','Amp3rdQrtl','Amp4thQrtl','Amp1stMean','Amp2ndMean','Amp3rdMean','Amp4thMean','LnExpA_StartAmp','LnExpB_StartAmp','AmpStartLn60ExpC','LnExpA_EndAmp','LnExpB_EndAmp','AmpEndLn60ExpC','AmpK@start','AmpK@end','AmpKurtosis','AmpSkew','AmpVariance','AmpMoment','AmpGausR2','Quality','HiFminusStartF','FcMinusEndF','RelPwr2ndTo1st','RelPwr3rdTo1st','LdgToFcSlp','KnToFcDur','UpprKnFreq','HiFtoUpprKnSlp','HiFtoUpprKnAmp','HiFtoUpprKnExp','HiFtoKnAmp','HiFtoKnExp','HiFtoFcAmp','HiFtoFcExp','UpprKnToKnAmp','UpprKnToKnExp','KnToFcAmp','KnToFcExp','LdgToFcAmp','LdgToFcExp','PreFc250','PreFc500','PreFc1000','PreFc250Residue','PreFc500Residue','PreFc1000Residue','PreFc3000','PreFc3000Residue','KneeToFcResidue','Kn-FcCurviness','meanKn-FcCurviness','Kn-FcCurvinessTrndSlp','MinAccpQuality','Max#CallsConsidered','file_id','chirp_idx','split'] 
        chirp1 = np.array([-1.4778419574815187,0.03484375289834824,-0.9386650336394794,-0.43812578654755646,2.60661056616406,1.2181594248813217,2.451151735281947,-0.1676646838408213,2.211552067418199,0.2687795603108202,-0.4408463944173755,2.263420036296655,-0.8926046755135605,1.2218585537810098,2.418928557876674,-0.5412090681924409,1.5298984228256614,-0.09417874417143986,-0.5061167647087946,-0.15120058705123002,0.5959721458904789,-0.28209050864201557,0.40258092664320005,0.02040398057783987,-0.1594333507700763,0.1862401343717595,-0.09225318304979092,1.4456012768024276,-0.5448245862140565,-0.9200971972274773,3.08491779259281,2.5297352515620815,-0.23752964961754472,2.330555666320157,2.8055732169639693,0.8701517109420618,2.8381460790407926,1.457067537016766,2.7987639971724443,1.5991591467077078,2.287474317502067,1.9169777496059595,-0.5783120466404935,-0.6955089928005129,-0.728480679770921,-0.24049666350521637,-0.4248532815871985,-0.17070396921258635,-0.306746352565768,-0.22733330725598816,0.11169110380982263,-0.6683709763375529,0.820751867687758,-1.498451435877687,0.0783709734588271,-0.11075725440861527,0.9544235448464778,-3.1136670474903543,2.533110945661222,-2.678690909362776,1.7763568394002505e-15,0.28483890889706376,0.3016571396193126,1.7763568394002505e-15,-0.030285805664782724,-0.3110459899792552,1.0635374174500771,-1.3794493428039327,3.157189584747198,3.1529477767011453,0.4620266836320838,-0.11075725440861527,0.9544235448464778,-3.1136670474903543,2.533110945661222,-2.678690909362776,1.6551343845921032,0.03794221813637991,1.6079184963317337,0.42969086939196915,-0.0019181673211083902,-0.12432599907621264,-0.002007109361103796,0.061498453627758416,-0.0018787346567131064,0.8804629361519999,-0.0019483823618756588,0.8224771507334433,-0.003385246700998527,0.5889654609352651,-0.0019094826663048627,-0.5496447634128939,-1.6184324598225466,-1.4447235497043414,-0.6027143854779639,-0.12549852365819306,0.023954127754194948,0.8644858999285274,-0.041098300197053,0.1392886846330207,0.10331499993423351,-1.2102054256013768,-0.6024109225764916,0.03498823619000142,0.0,0.0,0.0,0.0,0.0])
        chirp2 = np.array([-1.4160167128584713,0.010954099874630028,-0.9386650336394794,-0.03507620045485489,2.471389430942075,2.956883381326471,2.8742938687013835,1.9043398559895242,2.569421762318163,0.26540740605915514,0.012875796427263237,2.8974285304309,-0.448454510870412,2.9476782618518205,2.840753809148612,-0.06368657106119173,-0.35833040473729366,-2.3054405640088866,0.09940853981974482,1.2863144053296296,1.559888843505793,0.5970693375184611,1.815263456842474,0.3725371262091592,0.6327069306388495,0.3842845529093269,-0.13912330123689465,1.9020950669268422,-0.5911707160962512,0.5261200498562999,0.4719118132016026,2.2447307190674928,-0.2821298322882572,2.9515842547110034,3.1575080820938055,1.951763981042534,3.072967744391617,2.516259932699239,2.998517872618593,2.6564429810593966,2.726140957899917,2.2042436822448046,0.4874320228493959,0.7959942391472349,0.9574811018087477,-0.4988550388987855,0.7530915501133518,0.0822116726500705,0.04624273422435759,-0.4117043477381769,-1.0105166754972084,-0.8087248121136126,0.630519952760365,-0.7924157768331962,-1.0363108902657188,-0.4500969723932673,0.4391956079427542,-1.4939610209211442,2.3433541987260393,-1.54234838907265,1.7763568394002505e-15,-0.2704617771531219,0.06943241708499232,1.7763568394002505e-15,0.08863743547448949,-0.36722698923648023,-0.034891115299376066,-0.1123785147222785,1.6945605333174483,1.69782231601198,0.15577067980828235,-0.4500969723932673,0.4391956079427542,-1.4939610209211442,2.3433541987260393,-1.54234838907265,0.07006836026338514,0.46313599641480313,3.0692647781330304,2.295016089912671,-0.0019082199503833365,-1.1131173150557203,-0.0019879859205045236,-0.6972789341550952,-0.0018657409087607319,0.3243531641812425,-0.0018209211379833629,-0.49529331081632105,-0.0032966767952197383,0.3987091125724432,-0.001909834837956907,0.304737036872381,-0.15205539876230437,-0.20992151943481507,-0.6719258488247425,-0.10685034656345527,-0.2047484504418538,0.30092308053782313,-0.325902452362958,-0.3504920692494859,-0.2507344971918824,0.8902209249571663,-0.054092579484527704,0.06964200141756625,0.0,0.0,0.0,1.0,0.0])
        chirp3 = np.array([-1.613301763116285,-0.07317902599150809,0.5801944568105295,-0.38241107130235674,-0.11300411329124142,0.2523438269280249,0.18286286166354185,0.20943869968685347,0.01602452380851442,0.15285846762524574,-0.16236174266637815,-0.6128428409646229,0.961235433932059,0.26321092766913234,0.1576989413899062,0.5910466735736979,-0.19491429308448252,-0.03893596012236676,0.4001927243659518,-0.022469768041775407,0.8596446944955147,0.4824897399144324,0.32512742001648776,-0.2300280874402452,0.6417761860576805,0.4395996810583273,-0.08783709762690387,-0.8556161218990977,0.6497446688186002,1.3566169061074234,-0.6630906306951809,0.16364141528966708,0.7216391203408762,0.1696223696139039,0.08654562881084872,-0.05670277613964727,0.00999514078451578,0.3641209059499379,-0.004722050882182245,0.38401817075122124,0.6816878453902853,0.35700785014988273,-0.11563118205645702,0.5899962419041026,0.7138375062752733,-0.6376383276170599,0.6811864736241825,0.0374834788738564,-0.12109737457426248,-0.6716546755080871,1.1412874156167652,0.3291829778843615,1.4700855665483228,0.8843336182196967,0.6312127684404769,0.4903541876943056,-0.2124825566126116,0.5801226128398153,-1.042804289352111,0.6888289024996226,1.7763568394002505e-15,-0.6188911406251408,0.7083871819839463,1.7763568394002505e-15,-0.43613998684113414,-0.5329577432580296,1.0270482126121252,-0.6166485540229387,-0.9533113808178493,-0.95401441941939,-1.1104451204465622,0.4903541876943056,-0.2124825566126116,0.5801226128398153,-1.042804289352111,0.6888289024996226,0.3923181927210794,-0.6617028882172824,0.4160812414057949,0.2820665281225908,-0.0019190574445980787,-0.31283220742324863,-0.002008747608090748,-0.544837798245992,-0.0018795634598059612,-0.5656208307033699,-0.0019067485238537614,-0.6281733585582693,-0.0034146686586574862,0.00901622972554248,-0.0019108415644296913,-0.42539793161126993,-0.09301889242455247,0.14788950669080841,0.13873461607579068,-0.12442337841842607,-0.37778987637579026,-0.40207933467396806,0.25305397512746397,-0.5950518765123096,-0.5109375888638765,0.4003184593517802,0.7321367020439867,-0.04046976385518611,0.0,0.0,7500.0,0.0,5.0])
        chirp4 = np.array([-1.5181325663369876,-0.4595681966359942,0.5801944568105295,0.05369258864955311,0.4445633793087929,-0.01897934639674612,-0.33659293559916426,0.22687168968568658,-0.6485421603706965,0.946784336216713,-1.1506559206177442,-0.3830293291349461,0.3549807582372725,-0.006098572579289573,-0.30768775647868185,-0.07032474523341596,0.2708886942700014,0.07151437388773126,0.85108281638782,-0.5259655235506968,1.0970698169498203,-0.0477470596700525,-0.08778219379293346,-0.32749681309278367,-0.19991603908032946,-0.0861236871045636,-0.00434817036230662,-0.9542656893942312,1.826665680755909,0.519580431549319,-0.648943492930526,0.45434157157116484,-0.08108899790265192,-0.19527504799084416,-0.14780391330556295,-0.2902028849973298,-0.26191982724103247,0.11484255431824401,-0.30629455275038897,-0.15941005256332647,-0.4757598056546459,-0.21565683105399938,-0.2555643947455715,0.4756233491004868,0.10847198702550097,0.5335145842671039,0.9665802094480267,0.43664673220451744,0.14057515640848123,0.2563329295319487,-0.27847826991769925,-1.1438254919074,0.48479569596944355,-0.23139594171623265,-0.016543333413386426,-0.5805278719301251,0.8976962006512164,0.18848298854822104,0.2948904744916521,-0.924152485767734,1.7763568394002505e-15,0.4752211540317221,0.42162851499674026,1.7763568394002505e-15,0.27312179360097044,-0.24321878843051925,-0.5986420333156872,1.2649018242620416,-0.530568165269705,-0.5292971867985357,0.28257502490075653,-0.5805278719301251,0.8976962006512164,0.18848298854822104,0.2948904744916521,-0.924152485767734,0.2806953222555668,-0.8827895362243929,0.189338101755927,-0.19879208896849068,-0.0019193784123849855,0.1799190788536267,-0.002009602492152727,-0.044992859249919935,-0.001880887099443425,0.1088573559845042,-0.0018978325166453631,-0.8915940961681579,-0.0034170447365881257,0.2726374743255811,-0.0019108866320486006,-0.18468717907053928,-0.32842937652992105,0.021822047916315042,0.20600366389094765,-0.1272860291891389,-0.392792370503074,-0.5327678999126862,0.3206312208673833,-0.6485424096428158,-0.5614941118913708,-0.6236537175255488,-0.12863119968138687,-0.1854188904626692,0.0,0.0,7500.0,1.0,5.0])
        chirp5 = np.array([ -0.8929334634072966,-0.5156569472134196,-0.5451986437295693,-0.47754837177790643,-0.8345610838112374,-0.31165727716226294,-1.0094085106736654,0.3302056609099423,0.011523355059565744,-0.710526034389105,-0.05047049142087538,-0.6480873021626843,0.42536655149450303,-0.29660434214677034,-1.030861902590041,0.15960035593406996,1.3172428913936862,0.04914214998061458,-0.6813016468187559,0.0350581369658152,2.0729860211984827,0.5989309054315416,0.1456905707054233,1.035854967142936,0.82554861194628,0.9243332153937948,-0.09298128847042145,-0.0012457229445362594,0.13301412929832276,0.9194855113470299,-0.5111377443313756,-0.4234743156871685,0.2152654318953021,-0.3691781265534074,-0.5397844383015857,-0.5081079532831023,-0.6017083525956634,-0.32485672290227735,-0.5581776494458444,-0.24239804651601565,-0.008442775291470467,-0.14999988342244508,-0.2875244603002738,0.12113484423181688,0.28631618932529246,-0.3766325182478442,-0.4112482987007608,-0.2105164569785358,-0.3174242514976529,-0.592122766748172,0.9203521000389596,0.17659641868329803,-1.6903112191056773,-1.4481775301842845,0.9973357825317907,0.7712717319280735,-2.4255070700676287,-1.6171095093829921,-1.3034098813883552,0.2130373568143922,1.7763568394002505e-15,-0.04763723387287925,0.3757537881569451,1.7763568394002505e-15,-0.4822708032156961,-0.33863917130527,-0.2810176467186135,0.22770034838253178,0.3355348068804094,0.33313134132128497,-0.23214711965176493,0.7712717319280735,-2.4255070700676287,-1.6171095093829921,-1.3034098813883552,0.2130373568143922,1.50827345469556,-0.5565811561000752,-1.139471672665801,0.19557787258735185,-0.0019186289789717413,-0.636882368002207,-0.002008225930643929,-0.4677976542919294,-0.0018771086869519376,-1.0285015289157835,-0.0019483823618756588,0.8224771507334433,-0.003310064083332595,-1.7709769977282868,-0.0019090638526017849,-1.9424961075048859,-1.3394531899611029,-0.984873682553075,-1.1066128736042198,-0.11985628377961502,-0.38603175289634867,-0.34252617523411755,-1.1013132209615133,-0.4625733534062698,-0.49066928738250837,-0.23422950963992328,-0.058857605433185335,0.11134126725762275,0.0,0.0,7501.0,4.0,5.0])
        chirp6 = np.array([-1.6181644228057381,0.34644792277293385,-1.0398672693871795,-0.9325949116384182,1.7729284213012801,-0.29997705837358174,1.151044648394611,-1.2773880935889064,0.6271889294892179,0.5460953237484902,-0.9627320233903872,0.5262196039394125,-0.41206639411882745,-0.2850108101744507,1.1228676003566092,-0.4812392276621628,-0.00587059806957153,0.4972973138989185,0.0942751657192535,-1.1780471058533406,0.30539506272189965,-0.8775787497777545,-0.5916485703952687,-0.775425709916216,-0.36330596965888245,-1.392474893326616,2.181714951377437,-1.0456541926050134,2.049715800403681,-1.1554310063176028,2.1715757439253482,1.604486333900881,-0.15687417091800596,0.7303517800244852,0.06593290127781627,-0.3126741641909761,-0.021884880449118526,-0.054025076031992525,-0.0036705954099695466,0.08354812728921866,0.8035627108524948,0.47527776304737185,-0.40352008512967547,-0.06525334526415538,0.15877565839261037,-0.6078336747419135,-0.54651225861797,-1.118783818287787,-0.3494814803343243,-0.29427761454646395,0.15519759418243873,-0.6967259508980315,0.21799787175135468,0.8891067166757939,0.18669068408373732,-1.009376791900078,0.468057548061986,0.0009132523253659904,-0.2598396058373336,-0.18661264314554588,1.7763568394002505e-15,-0.6855364090667306,0.567661479694355,1.7763568394002505e-15,-0.12202520140809332,-0.5328687734699502,-0.44612169166992127,2.2240805127993726,-0.9245731720761526,-0.9298077377674242,-0.4539302069475164,-1.009376791900078,0.468057548061986,0.0009132523253659904,-0.2598396058373336,-0.18661264314554588,-0.23228059789877423,-0.9036622196271692,-0.03135863218209135,-0.4763845891962944,-0.0019190692246182637,0.4679798042781618,-0.0020092907124187155,0.6731485577336347,-0.0018806522842982167,1.1075710536244894,-0.0019483823618756588,0.8224771507334433,-0.0034122433993871945,0.9286651728786846,-0.001910795563976953,0.41432721844459175,0.14024835329548221,0.4912789181952285,0.6792269809206942,-0.1257995298176396,-0.3964990466021614,-0.5345002296084922,0.7960226441658333,-0.6492514482554689,-0.5614719055077148,-0.7136912301759467,-0.32025331461669143,-0.6211842508209481,0.0,0.0,12000.0,0.0,8.0])
        chirp7 = np.array([-0.8825134783584685,-0.33284916755366273,-1.0398672693871795,0.7494646122083971,0.4437359291008508,0.39791870902188614,0.4525367830560954,0.20687403794583864,0.13417387647712753,-0.023033086333475027,0.04863710537799894,0.09395837570180547,-0.01574569276139347,0.4077053999436886,0.4265335968604918,0.1694353547754491,-1.092242076511064,-0.23466669124409517,0.3509210665299507,-0.7811948920428401,-0.5807587806894353,-0.6283014167486856,-0.20051116944317093,-0.8135241900696218,-0.7733952229294885,-0.5756477114689732,-0.2602824158747634,-0.35444129577648226,-0.5146992104503368,0.6407488783742638,-0.7560571678783244,0.2750694786061688,-0.1670867788889346,0.06645961164477615,0.4329277682623963,0.07782011615500728,0.36333246437250477,0.39414238435184407,0.28622048922317583,0.28095224294535676,0.21145444171579314,-0.04705661940879575,-0.15531878887843864,0.2535880804736756,0.1418566439006039,-0.6145967766437499,1.8425409338316654,1.0158868897511273,0.9060790396637556,0.1789844288526543,-0.09527850132934441,-0.4814737952900727,0.014739754301643109,-0.48594694486172657,-0.11434665660567349,-0.6520878675571414,-0.5206002196314179,0.526341261483207,-1.539603309256055,0.8266462748679487,1.7763568394002505e-15,-1.3850131789474815,0.44521191411810707,1.7763568394002505e-15,-0.5177471779308901,-0.5450378158462164,-0.478129845820338,0.6152389378818774,-0.7259046855574681,-0.7234745941625729,-0.33728135264435366,-0.6520878675571414,-0.5206002196314179,0.526341261483207,-1.539603309256055,0.8266462748679487,-1.1566807056202488,0.2633837614511605,0.587760621907099,-0.2908001513223215,-0.0019192748544415565,0.385066121443296,-0.002009433728423191,0.26190353397885013,-0.001881024753918185,0.7361424565766209,-0.0019086149090953813,0.046883954910840527,-0.003422307772473007,0.8215338232800884,-0.0019114531154395423,1.0077167392868125,1.0310121899420073,1.126731408882367,1.0143194801909314,-0.1262665011757574,-0.35805874802357773,-0.34982165914761343,0.8303737857295216,-0.2577727234007223,-0.1969704460973017,0.3183062955233548,-0.1541581244063386,0.09270626238028325,0.0,0.0,12000.0,8.0,8.0])        
        chirp8 = np.array([0.9465412305458385,1.135845153121884,0.5212842227853761,-0.022697701822991367,0.2290039583621305,-0.24399502255732947,0.32474114598139353,-0.5794431238411843,-0.19787957511935028,0.32019370723090734,-0.33179786625370034,-0.48008301186620406,0.7593718754731419,-0.2294442549708605,0.29913567908552474,-0.6659374855965227,-0.26617862512842544,-0.06963543683888969,0.34443900917098436,0.25906870645022356,-0.9297545692955558,-0.5133685716702654,-0.2762091490761669,-0.7279169868742379,-0.5338712797560131,-0.4370383800501288,-0.1244440434796213,-0.8033851778669707,0.1947110337243062,0.745311609144782,-0.8786463684810656,0.1275906688559692,-0.3280918125641311,-0.24575637473568965,0.23453458165468083,-0.461733785451234,0.10138467551489236,-0.20700532922254972,0.05731396102299348,-0.13374932191427089,-0.19047926681188593,-0.43368652156877013,-0.6739611951831896,-0.45236714741045947,-0.32172555827739313,-0.7038708895618296,-0.5470771568151295,0.19934149967247458,0.10172752181593732,0.2113735830745145,-0.2876670867195671,0.011002812440226816,1.4808612167694288,-0.25140793414427987,-1.309534548357737,-0.800330861459268,0.4878002432575523,-0.3111037878383911,0.3301694954374173,-0.2040008209861121,1.7763568394002505e-15,0.1512421681616372,-0.14742367339456702,1.7763568394002505e-15,-0.21749041200417932,-0.37714084709663726,-0.5712315765407504,0.5271398913835226,0.31960384333881675,0.3214995592287819,1.1047519095132246,-0.800330861459268,0.4878002432575523,-0.3111037878383911,0.3301694954374173,-0.2040008209861121,-0.5214646199703387,-0.6639080203907626,0.07117969733452503,-0.8974592011027825,-0.0019197801806871695,1.0248978984045924,-0.002010085373300498,0.124504028178425,-0.0018818157414638658,0.4900524148075891,-0.0019113274211678914,-0.14531633186814522,-0.003427097766719188,0.654288120224778,-0.0019112344773127875,0.44429991830024246,0.3766231760385662,0.32833451654456997,0.6226057548250027,-0.12303699585674688,-0.26230865259349573,-0.46862621835206036,0.6931464128632931,-0.6103706933687677,-0.5342161348201492,-0.5760470366495749,-0.2691994651667879,0.11104292358347816,0.0,0.0,13499.0,23.0,8.0])
        chirp9 = np.array([0.6617283058778682,-0.4979993775871931,0.5212842227853761,-0.7332503697965014,0.5190340330513981,0.03238377360086113,0.1966504833474588,-0.10327509004136605,1.2333810851339797,-1.9304129087348136,0.011205543055589611,-0.3440218041039083,0.8768979441307807,0.04488334690901704,0.17144365391536784,0.9375776172174671,0.5583500184119758,-0.18122858770483222,-1.5790915786316377,-0.2492563670470417,1.4458675372764453,0.7447749098542095,0.41581967010512166,-0.10065012409365105,0.8293273005335096,-0.07296654139750945,0.37943308275552096,-0.9907897175715408,2.911704206304575,0.5812603694688622,-0.2870009557702084,0.5036837252696003,-0.24090355125806032,0.2814373589264165,0.45501384193366523,-0.2152120936790843,0.36333246437250477,0.14798728216482504,0.33643520499815105,0.27368346965801527,0.6761214026081424,0.6034649843579603,-0.510891265980824,-0.15264525352630356,0.06252145478384817,0.02625887730603546,-0.5124047734402268,-0.30871371948563137,-0.3636228250630947,-0.47603146687088527,2.511684325155599,2.9047581284483996,-0.6762370318853537,0.22033139443190278,0.6718522113887677,0.9086080900222161,-1.5585207124794722,-0.03356250575060579,-0.35147322770383943,0.6592288427160629,1.7763568394002505e-15,0.777321874658749,-3.7853215727740146,1.7763568394002505e-15,-0.5156709437338447,1.828849649715312,-0.5103127175206632,0.17640147856203978,-0.3161769630988604,-0.3208682266004404,-1.3649950718310184,0.9086080900222161,-1.5585207124794722,-0.03356250575060579,-0.35147322770383943,0.6592288427160629,0.5404840741055417,-0.9268055151107606,-0.8525458768544549,0.4417557906089755,-0.0019193093450440872,-0.8448409231104257,-0.002009871469673649,-0.6825479929874082,-0.001881270769253488,-0.7359071501183585,-0.0019483823618756588,0.8224771507334433,-0.0034254661171590813,0.057411895058459374,-0.0019110463623189635,-0.38120818112408905,-0.4771195477744671,-0.06773633009984617,0.1157267600990566,-0.1270797664680176,-0.3969273411658254,-0.5347003947622184,0.22994070702293007,-0.6493333753985775,-0.5602100026507447,-0.36128825693925487,0.9064005081663239,-1.332458519495065,0.0,0.0,13499.0,19.0,8.0])
        chirp10 = np.array([0.6978509207138059,-0.2320971526275467,0.5212842227853761,-0.8263019297213668,0.6424321362551967,-0.0713697903890889,0.6032066882154291,-0.5514321873911191,0.7404916302561272,-0.7945742939936252,-0.3304401404325958,0.5939529813846879,-0.5461539888166895,-0.05810019159500909,0.5767345945473736,0.7333158724175753,2.154590048562692,0.10652299537260072,0.06050293364086054,0.03621223981041826,0.8101624555265322,0.7440782873239252,0.20121163081741228,1.3194187332884357,0.9236683015775176,0.14179841647610616,0.3701041393722492,0.6825952478994151,-0.11442491947814427,-0.769857495173853,1.9519833046261064,0.6852152690785004,-0.13711499462642662,0.3784268212265632,0.6299478354094011,-0.250705567528011,0.5771497965938714,0.0680152984166631,0.5667741709495541,0.17510611364684353,0.5491575525969531,0.5004699922825967,-0.6520043835048306,-0.5181783204137573,-0.4216049128528433,0.04860119974254858,-0.5469412018385743,-0.4409545812460132,-0.5526740655448414,-0.47430505117923155,0.7310186756928237,1.2140877339118905,0.1001608736691926,1.3079895900993148,0.8637509801396169,1.4849396555693948,-0.44785308847746885,0.7373831910435001,-0.9197333099920688,0.7994305639510949,1.7763568394002505e-15,0.9899804500443867,0.07682983546858932,1.7763568394002505e-15,-0.4477685424618731,0.44855496271704615,1.277973922049656,-0.9219977513868528,-1.0215125646070202,-1.0248530202710304,-0.9008899869399359,1.4849396555693948,-0.44785308847746885,0.7373831910435001,-0.9197333099920688,0.7994305639510949,0.42958067466127053,-0.4740634731707612,0.03255671944625824,0.2461761199257433,-0.0019192795513632659,-0.40769741210818505,-0.0020097994104299227,-0.23112767513893198,-0.0018805833881613057,-0.6856009182381174,-0.0019483823618756588,0.8224771507334433,-0.003397242694860048,-0.9866564716760984,-0.0019110060210220355,-0.2538242067375735,-0.4867392070146658,-1.1436234685111226,-1.4512640855650616,2.319858253450094,1.6426282915936379,0.3299630116387457,-1.282873317101257,-0.29726308575466015,-0.23059162256363533,-1.0549299525548874,-0.655847285000724,0.17543925817191983,0.0,0.0,13499.0,20.0,8.0])
        
        # Fix the file_ids to match self.fid_to_fname_dict:
        self.split_df1 = pd.DataFrame([chirp1,chirp2,chirp3,chirp4], columns=cols)
        self.split_df1.file_id = pd.Series([11,12,13,14])
        
        self.split_df2 = pd.DataFrame([chirp5,chirp6,chirp7,chirp8], columns=cols)
        self.split_df2.file_id = pd.Series([15,16,17,18])
        
        self.split_df3 = pd.DataFrame([chirp9,chirp10], columns=cols)
        self.split_df3.file_id = pd.Series([15,19])
        
        
# ----------------------------- Main ------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()