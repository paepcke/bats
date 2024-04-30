'''
Created on Apr 27, 2024

@author: paepcke
'''

from data_calcs.data_prep import DataPrep
from pandas.testing import assert_frame_equal
from tempfile import TemporaryDirectory, NamedTemporaryFile
import os
import pandas as pd
import unittest

TEST_ALL = True
#TEST_ALL = False

class DataPrepTester(unittest.TestCase):

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

        sorted_measures = DataPrep.sort_by_variance(jumbled)

        expected = ['LdgToFcAmp','HiFtoUpprKnAmp','HiFtoKnAmp',
                    'MaxSegLnght','AmpStartLn60ExpC','AmpEndLn60ExpC']
        
        self.assertListEqual(sorted_measures, expected)
        
        with self.assertRaises(ValueError):
            DataPrep.sort_by_variance(['Bluebell'])

    #------------------------------------
    # test_measures_by_var_rank
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_measures_by_var_rank(self):

        # Want only two of three columns, sorted by
        # variance rank:
        
        df = pd.DataFrame([{'HiFtoKnAmp' : 10,  'LdgToFcAmp' : 20,  'HiFtoUpprKnAmp' : 30},
                           {'HiFtoKnAmp' : 100, 'LdgToFcAmp' : 200, 'HiFtoUpprKnAmp' : 300}])
        df_new = DataPrep.measures_by_var_rank(df, 2)
        
        expected = pd.DataFrame([{'LdgToFcAmp' : 20,  'HiFtoUpprKnAmp' : 30},
                                 {'LdgToFcAmp' : 200, 'HiFtoUpprKnAmp' : 300}])

        assert_frame_equal(df_new, expected)
        
        # Make minimum rank higher than number
        # of cols in given df:
        df_new = DataPrep.measures_by_var_rank(df, 15)
        expected_cols = DataPrep.sort_by_variance(df.columns)
        expected = df[expected_cols]
        assert_frame_equal(df_new, expected)

    #------------------------------------
    # test_constructor
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_constructor(self):
        
        # Measures and inference roots are both the test tmpdir: 
        dp = DataPrep(self.tmpdir.name, self.tmpdir.name)
        # dp's fid2split_dict should be similar to:
        #   {100: '/tmp/data_prep_6057jcgj/split15.feather',
        #    110: '/tmp/data_prep_6057jcgj/split15.feather',
        #    120: '/tmp/data_prep_6057jcgj/split15.feather',
        #    11: '/tmp/data_prep_6057jcgj/split4.feather',
        #    12: '/tmp/data_prep_6057jcgj/split4.feather',
        #    13: '/tmp/data_prep_6057jcgj/split4.feather'
        #    }
        
        expected = {100: f'{self.tmpdir.name}/split15.feather',
                    110: f'{self.tmpdir.name}/split15.feather',
                    120: f'{self.tmpdir.name}/split15.feather',
                    11:  f'{self.tmpdir.name}/split4.feather',
                    12:  f'{self.tmpdir.name}/split4.feather',
                    13:  f'{self.tmpdir.name}/split4.feather'}
        self.assertDictEqual(dp.fid2split_dict, expected)
        
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
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_measures_from_fid(self):
        
        dp = DataPrep(self.tmpdir.name, self.tmpdir.name)
        
        measures = dp.measures_from_fid(100)
        # Ground truth:
        df_truth = pd.read_feather(f'{self.tmpdir.name}/split15.feather')
        df_truth.index = df_truth.file_id
        
        #    TimeInFile          -0.41427
        #    PrecedingIntrvl     -0.41427
        #    CallsPerSec         -0.41427
        #    file_id            100.00000
        #    Name: 100, dtype: float64
        
        expected = df_truth.loc[100]
        pd.testing.assert_series_equal(measures, expected)


    #------------------------------------
    # test_feather_input
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_feather_input(self):
        df = pd.read_feather(self.dst_file4)
        pd.testing.assert_frame_equal(df, self.tst_df4)
        
    # ----------------------- Utilities ----------------
    
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
        self.tst_df4 = pd.DataFrame(
            {'TimeInFile'      : c1,
             'PrecedingIntrvl' : c2,
             'CallsPerSec'     : c3,
             'file_id'         : file_id
             })
        
        self.dst_file4 = os.path.join(self.tmpdir.name, 'split4.feather')
        self.tst_df4.to_feather(self.dst_file4)

        # Build another small df, which will be split15:
        c1 = ['foo1','foo2','foo3']
        c2 = ['bar1','bar2','bar3']
        c3 = ['fum1','fum2','fum3']
        file_id = [100,110,120]
        self.tst_df15 = pd.DataFrame(
            {'TimeInFile'      : c1,
             'PrecedingIntrvl' : c2,
             'CallsPerSec'     : c3,
             'file_id'         : file_id
             })
        self.dst_file15 = os.path.join(self.tmpdir.name, 'split15.feather')
        self.tst_df15.to_feather(self.dst_file15)

# ----------------------------- Main ------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()