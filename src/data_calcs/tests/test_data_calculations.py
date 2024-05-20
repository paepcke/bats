'''
Created on Apr 27, 2024

@author: paepcke
'''

from data_calcs.data_calculations import DataCalcs, PerplexitySearchResult
from data_calcs.daytime_file_selection import DaytimeFileSelector
from logging_service.logging_service import LoggingService
from pandas.testing import assert_frame_equal
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
from datetime import datetime
import os
import pandas as pd
import unittest
import numpy as np

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
        split_file_name_to_id = ('barn1_D20220205T192049m784-HiF.wav,11\n'
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
                                   in fname_id_pairs 
                                   }
            
        

# ----------------------------- Main ------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()