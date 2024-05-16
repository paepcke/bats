'''
Created on Apr 28, 2024

@author: paepcke
'''
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
        
        viz = DataViz(self.df_file)
        num_points = 10
        # Use just Col1, Col2, and Col3, not file_id:
        num_dims   = 3
        
        # Get something like:
        #     tsne_df
        #                 tsne_x     tsne_y
        #     file_id                      
        #     65        5.611321   2.916003
        #     66       46.984352   2.345862
        #     67      -52.424259  44.120220
        #     68      -12.179502  44.626827
        #     69      -31.309053   3.476619
        #     70       26.676809 -37.502068
        #     71       28.054359  42.841827
        #     72      -53.905338 -36.210114
        #     73      -72.722496   4.274879
        #     74      -13.652823 -38.519417
        
        # The test df is not measures from SonoBat. So sorting
        # by SonoBat measure variance makes no sense:
        tsne_df = viz.run_tsne(num_points, num_dims, sort_by_bat_variance=False)
        self.assertEqual(tsne_df.ndim, 2)
        self.assertEqual(len(tsne_df), len(self.small_df))
        # Tsne index should be:
        #    Index([65, 66, 67, 68, 69, 70, 71, 72, 73, 74], dtype='int64', name='file_id')
        self.assertEqual(tsne_df.index.name, 'file_id')

    #------------------------------------
    # test_cluster_tsne 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_cluster_tsne(self):
        
        # Test the clustering of a Tsne result embedding:
        
        # Get a Tsne df:
        viz = DataViz(self.df_file)
        tsne_df = viz.run_tsne(sort_by_bat_variance=False)
        
        kmeans = viz.cluster_tsne(tsne_df, n_clusters=3)
        labels = kmeans.labels_
        # The cluster assignments vary with each
        # run. But at least ensure that the number
        # of clusters is 3, as requested in the viz.cluster_tsne()
        # call:
        self.assertEqual(len(set(labels)), 3)

        # Do another clustering of the same Tsne result, but
        # have cluster_tsne find the best n_clusters:
        kmeans = viz.cluster_tsne(tsne_df, n_clusters=None)
        labels = kmeans.labels_
        #n_clusters = len(set(labels))
        
        # Test how many add_silhouette were computed. Since
        # we did not provide a cluster_range in the call to 
        # cluster_tsne(), we don't know the range (the default
        # of the current implementation might change):
        self.assertTrue(len(viz.n_cluster_range) > 2)

        # Do one more clustering of the same Tsne result, but
        # have cluster_tsne find the best n_clusters within a
        # range we provide:
        kmeans = viz.cluster_tsne(tsne_df, n_clusters=None, cluster_range=range(2,4))
        labels = kmeans.labels_
        #n_clusters = len(set(labels))
        # We expect two add_silhouette: one for n_clusters==2, and 
        # one for n_clusters==3
        self.assertEqual(len(viz.add_silhouette), 2)

        
    #------------------------------------
    # test_plot_tsne
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_plot_tsne(self):
        
        #**********
        path = '/Users/paepcke/quintus/home/vdesai/bats_data/inference_files/measures/split80.feather'
        #***********
        viz = DataViz(path)
        #****viz = DataViz(self.df_file)
        tsne_df = viz.run_tsne(sort_by_bat_variance=False)
        #*****kmeans = viz.cluster_tsne(tsne_df, n_clusters=3)
        kmeans = viz.cluster_tsne(tsne_df, n_clusters=None)
        
        # The sum of cluster populations should be equal
        # to the number of rows in the tsne_df, and also
        # to viz.effective_num_points:
        
        self.assertEqual(len(tsne_df), viz.effective_num_points)
        sum_cluster_population = sum(viz.cluster_populations)
        self.assertEqual(sum_cluster_population, len(tsne_df))
        
        #viz.plot_tsne(tsne_df)
        viz.plot_tsne(tsne_df, kmeans)
        
        print()
        
        
        
    # ---------------------- Utilities ---------------

    def create_test_files(self):

        self.tmpdir = tempfile.TemporaryDirectory(dir='/tmp', 
                                                  prefix='dataviz_',
                                                  delete=True)
        df = pd.DataFrame(
            {
            'Col1':    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Col2':    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'Col3':    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'file_id': [65,66,67,68,69,70,71,72,73,74]
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