'''
Created on Apr 28, 2024

@author: paepcke
'''
from data_calcs.data_calculations import DataCalcs
from data_calcs.data_viz import DataViz
import os
import pandas as pd
import tempfile
import unittest

TEST_ALL = True
#TEST_ALL = False

class DataVizTester(unittest.TestCase):


    def setUp(self):
        self.create_test_files()


    def tearDown(self):
        self.tmpdir.cleanup()

    # ---------------------- Tests ---------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_run_tsne(self):
        
        # Use just columns 'LdgToFcAmp', 'HiFtoUpprKnAmp', 'HiFtoKnAmp'
        num_dims   = 3
        
        # Get something like:
        #     tsne_df
        #
        #         tsne_x     tsne_y  LdgToFcAmp  HiFtoUpprKnAmp  HiFtoKnAmp  file_id
        # m1  -54.806255  28.564016           1               4           7       65
        # m2  -22.131073  53.253307           2               5           8       66
        # m3  -23.132006 -30.249628           3               6           9       67
        # m4   11.317867  37.579723           4               7          10       68
        # m5   42.992317 -21.148355           5               8          11       69
        # m6    9.180456  -3.989465           6               9          12       70
        # m7   43.702076  19.802553           7              10          13       71
        # m8   10.346558 -45.884659           8              11          14       72
        # m9  -20.996885  11.334363           9              12          15       73
        # m10 -55.498295 -12.432767          10              13          16       74
        
        # Ask for the whole df by making num_points be
        # the length of the df we'll use: 
        # Initially, don't exclude any columns:
        num_points = len(self.small_df)
        tsne_df = DataCalcs().run_tsne(self.small_df, num_points, num_dims, sort_by_bat_variance=True)
        self.assertEqual(tsne_df.ndim, 2)
        self.assertEqual(len(tsne_df), len(self.small_df))
        # Since num_dims is 3, the file_id column will be cut off:
        expected_cols = ['tsne_x', 'tsne_y', 'LdgToFcAmp', 'HiFtoUpprKnAmp', 'HiFtoKnAmp']
        self.assertListEqual(list(tsne_df.columns), expected_cols)

        # Make a df that does not use SonoBat measure names.
        # Things should still work:
        non_sono_df = self.small_df.copy()
        non_sono_df.columns = ['Col1', 'Col2', 'Col3', 'file_id']
        # We'll set num_dims to include all the columns this time:
        num_dims = len(non_sono_df.columns)
        tsne_df = DataCalcs().run_tsne(non_sono_df, num_points, num_dims, sort_by_bat_variance=False)
        self.assertEqual(tsne_df.ndim, 2)
        self.assertEqual(len(tsne_df), len(self.small_df))
        expected_cols = ['tsne_x', 'tsne_y', 'Col1', 'Col2', 'Col3', 'file_id']
        self.assertListEqual(list(tsne_df.columns), expected_cols)

        # If setting sort_by_variance to True, should get ValueError,
        # because there is no overlap between the SonoBat measure names
        # and the Col1, Col2, ... columns we are now using:
        with self.assertRaises(ValueError):
            tsne_df = DataCalcs().run_tsne(non_sono_df, num_points, num_dims, sort_by_bat_variance=True)

        # Tsne on just 5 points
        num_points = 5
        tsne_df = DataCalcs().run_tsne(self.small_df, num_points, num_dims, sort_by_bat_variance=True)
        self.assertEqual(tsne_df.ndim, 2)
        self.assertEqual(len(tsne_df), num_points)
        
    #------------------------------------
    # test_cluster_tsne 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_cluster_tsne(self):
        
        # Test the clustering of a Tsne result embedding:
        
        # Get a Tsne df:
        data_calcs = DataCalcs()
        tsne_df = data_calcs.run_tsne(self.small_df, sort_by_bat_variance=False)
        
        cluster_res = data_calcs.cluster_tsne(tsne_df, n_clusters=3)
        kmeans = cluster_res.best_kmeans
        labels = kmeans.labels_
        # The cluster assignments vary with each
        # run. But at least ensure that the number
        # of clusters is 3, as requested in the viz.cluster_tsne()
        # call:
        self.assertEqual(len(set(labels)), 3)

        # Do another clustering of the same Tsne result, but
        # have cluster_tsne find the best n_clusters:
        cluster_res = data_calcs.cluster_tsne(tsne_df, n_clusters=None)
        kmeans = cluster_res.best_kmeans
        labels = kmeans.labels_
        #n_clusters = len(set(labels))
        
        # Test how many add_silhouette were computed. Since
        # we did not provide a cluster_range in the call to 
        # cluster_tsne(), we don't know the range (the default
        # of the current implementation might change):
        self.assertTrue(len(data_calcs.n_cluster_range) > 2)

        # Do one more clustering of the same Tsne result, but
        # have cluster_tsne find the best n_clusters within a
        # range we provide:
        cluster_res = data_calcs.cluster_tsne(tsne_df, n_clusters=None, cluster_range=range(2,4))
        kmeans = cluster_res.best_kmeans
        labels = kmeans.labels_
        #n_clusters = len(set(labels))
        # We expect two silhouette coefficients: one for n_clusters==2, and 
        # one for n_clusters==3. Like:
        #    {2: 0.21597628, 3: 0.20873506}
        self.assertListEqual(list(cluster_res.silhouettes_dict().keys()), [2,3])

        
    #------------------------------------
    # test_plot_tsne
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_plot_tsne(self):
        
        data_calcs = DataCalcs()
        #****viz = DataViz(self.df_file)
        tsne_df = data_calcs.run_tsne(self.small_df, sort_by_bat_variance=False)
        #*****kmeans = viz.cluster_tsne(tsne_df, n_clusters=3)
        cluster_res = data_calcs.cluster_tsne(tsne_df, n_clusters=None)
        kmeans = cluster_res.best_kmeans
        
        # The sum of cluster populations should be equal
        # to the number of rows in the tsne_df, and also
        # to data_calcs.effective_num_points:
        
        self.assertEqual(len(tsne_df), data_calcs.effective_num_points)
        sum_cluster_population = sum(data_calcs.cluster_populations)
        self.assertEqual(sum_cluster_population, len(tsne_df))
        
        #viz.plot_tsne(tsne_df)
        fig = DataViz.plot_tsne(tsne_df, cluster_ids=kmeans.labels_, title="Test TSNE", show_plot=False)
        # Check that the figure has as many
        # points as there are rows in tsne_df:
        ax = fig.gca()
        collection = ax.collections[0]
        num_points = len(collection.get_offsets().data)
        expected = len(tsne_df)
        self.assertEqual(num_points, expected) 
        
    #------------------------------------
    # test_simple_chart 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_simple_chart(self):
        
        fig = DataViz.simple_chart(self.small_df, 'SonoBat Measure Name', 'Magnitudes')
        ax = fig.gca()
        self.assertEqual(ax.get_title(), '')
        self.assertEqual(ax.get_xlabel(), 'SonoBat Measure Name')
        
        legend   = ax.legend()
        txt_objs = legend.get_texts()
        txts = [txt_obj.get_text() for txt_obj in txt_objs]
        self.assertListEqual(list(self.small_df.columns), txts)
        
        # Test stacking of charts:
        fig = DataViz.simple_chart(self.small_df, 'SonoBat Measure Name', 'Magnitudes', title='My Chart', stacked=True)
        
        axes = fig.axes
        self.assertEqual(len(axes), len(self.small_df.columns))
        self.assertEqual(fig.get_suptitle(), 'My Chart')
        
        # Each chart has a col name as its title:
        for ax, title in zip(axes, self.small_df.columns):
            self.assertEqual(ax.get_title(), title)
            
        # Test plotting a series instead of a df:
        one_col = self.small_df.LdgToFcAmp
        fig = DataViz.simple_chart(one_col, xlabel='', ylabel='LdgToFcAmp')
        
        ax = fig.gca()
        self.assertEqual(ax.get_ylabel(), 'LdgToFcAmp')
    
        
    # ---------------------- Utilities ---------------

    def create_test_files(self):

        self.tmpdir = tempfile.TemporaryDirectory(dir='/tmp', 
                                                  prefix='dataviz_',
                                                  delete=True)
        df = pd.DataFrame(
            {
            'LdgToFcAmp'     : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'HiFtoUpprKnAmp' : [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'HiFtoKnAmp'     : [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'file_id'        : [65,66,67,68,69,70,71,72,73,74]
            },
            index=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10']
        )
        self.small_df = df
        self.df_file  = os.path.join(self.tmpdir.name, 'tsne_test.feather')
        self.small_df.to_feather(self.df_file)
        
# ---------------------- Main ---------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()