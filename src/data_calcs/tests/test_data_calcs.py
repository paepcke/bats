'''
Created on Feb 22, 2024

@author: paepcke
'''
from data_calcs.data_calcs import DataCalcs
from tempfile import TemporaryDirectory
import pandas as pd
import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#******TEST_ALL = True
TEST_ALL = False


class BatCalcTester(unittest.TestCase):


    def setUp(self):
        self.tmp_dir  = TemporaryDirectory(prefix='data_calc_', dir='/tmp')

        # Create a simple df with column values representative of
        # SonoBat data types. Store the df as a raw .csv, i.e
        # without an index:
        self.sono_raw_path = os.path.join(self.tmp_dir.name, 'sono_raw.csv')
        self.df_simple = self.make_simple_df(self.sono_raw_path)
        
        # Save the same df to .csv as a dataframe:
        self.sono_df_path = os.path.join(self.tmp_dir.name, 'sono_df.csv')
        self.df_simple.to_csv(self.sono_df_path, index=True)
        
        # Create a df with enough data to test stats analysis:
        self.sono_complex_path = os.path.join(self.tmp_dir.name, 'sono_complex.csv')
        self.df_stats = self.make_complex_df(self.sono_complex_path)
        
        # Path where stats dataframe will be placed:
        self.sono_stats_path = os.path.join(self.tmp_dir.name, 'sono_stats.csv')

        self.admin_cols = ['strings']

    def tearDown(self):
        # Deleting the tmp_dir object deletes 
        # the dir and its content:
        self.tmp_dir.cleanup()

    # -------------- Tests ---------------

    #------------------------------------
    # test_cleaning_sonobat_raw
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_cleaning_sonobat_raw(self):

        cleaner = DataCalcs(self.sono_raw_path, unittesting=True)
        
        new_df = cleaner.load_sonobat_data(
            self.sono_raw_path,
            exclude_cols=self.admin_cols,
            remove_nans=True)
        expected = pd.DataFrame(
            [[5,  '0.5 sec',  '30 kHz anti-katydid',          1],
             [3,  '4.0 sec',  '10 kHz cutoff',          2],
             [1,  '4.0 sec',  '10 kHz cutoff',          3]
             ],
            columns=['numbers','mixed', 'enum', 'TimeIndex'],            
            )
        expected.index.name = 'TimeIndex'
        pd.testing.assert_frame_equal(new_df, expected)

    #------------------------------------
    # test_make_numeric
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_make_numeric(self):
        
        # Get:
        #               numbers    mixed           enum         TimeIndex
        #     TimeIndex                                                  
        #     0                5  0.5 sec  30 kHz anti-katydid          1
        #     1                3  4.0 sec        10 kHz cutoff          2
        #     2                1  4.0 sec        15 kHz cutoff          3        

        cleaner = DataCalcs(self.sono_raw_path, unittesting=True)
        new_df = cleaner.load_sonobat_data(
            self.sono_raw_path,
            exclude_cols=self.admin_cols,
            remove_nans=True)

        # To match SonoBat column names, change two names:
        conversion = {
            'mixed': 'MaxSegLnght',
            'enum' : 'Filter'
            }
        new_df.rename(conversion, axis='columns', inplace=True)
        # Again, to match Sonobat data, add a 'Preemphasis' 
        # column set to 'low', 'medium', or 'high':
        new_df['Preemphasis'] = ['medium', 'low', 'high']
        
        # We now have:
        #               numbers MaxSegLnght    Filter        TimeIndex Preemphasis
        #    TimeIndex                                                                
        #    0                5     0.5 sec  30 kHz anti-katydid  1      medium
        #    1                3     4.0 sec        10 kHz cutoff  2         low
        #    2                1     4.0 sec        10 kHz cutoff  3        high        
        
        # Turn the 'MaxSegLnght' column into just the numbers.
        # And turn the 'Filter', and 'Preemphasis' columns 
        # into values [0,1], as if from an enum:

        df_numeric = cleaner.make_numeric(new_df)
        expected = pd.DataFrame(
            [[5,  0.5,  1,          1, 1],
             [3,  4.0,  2,          2, 0],
             [1,  4.0,  2,          3, 2]
             ],
            columns=['numbers','MaxSegLnght', 'Filter', 'TimeIndex', 'Preemphasis'],
            )
        expected.index.name = 'TimeIndex'
        
        # DataCalc makes columns that were modified
        # from string to int or float into type 'object'.
        # To make the expected frame equal to the one that
        # was returned, must turn the respective cols in
        # expected to be dtype object as well: 
        expected['MaxSegLnght'] = expected['MaxSegLnght'].astype(object)
        expected['Filter'] = expected['Filter'].astype(object)
        expected['Preemphasis'] = expected['Preemphasis'].astype(object)
        
        # Can't get 
        pd.testing.assert_frame_equal(df_numeric, expected)

    #------------------------------------
    # test__normalize_frame
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__normalize_frame(self):
        
        df_raw = pd.read_csv(self.sono_raw_path, index_col=False)
        df_raw_normal = DataCalcs._normalize_frame(df_raw)
        
        # Same for file saved from df with index info:
        df_df = pd.read_csv(self.sono_df_path, index_col=False)
        df_df_normal = DataCalcs._normalize_frame(df_df)
        
        pd.testing.assert_frame_equal(df_raw_normal, df_df_normal)

    #------------------------------------
    # test_loading
    #-------------------
    
    unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_loading(self):
        # Test whether we can load either
        # a raw sonobat csv file, or a csv file
        # stored from a dataframe by DataCalc, and
        # end up with the same data:
        cleaner = DataCalcs(self.sono_df_path, unittesting=True)
        
        df_df = cleaner.load_sonobat_data(
            self.sono_df_path,
            exclude_cols=self.admin_cols,
            remove_nans=True)
        
        df_raw = cleaner.load_sonobat_data(
            self.sono_raw_path,
            exclude_cols=self.admin_cols,
            remove_nans=True)
        
        pd.testing.assert_frame_equal(df_df, df_raw)

    #------------------------------------
    # test_stats_computation
    #-------------------
    
    #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_stats_computation(self):
        
        cleaner = DataCalcs(self.sono_complex_path, 
                            compute_stats=True,
                            outfile=False,
                            admin_cols=['Filename']
                            )
                
        stats_df = cleaner.compute_stats(cleaner.df)
        
        # In order to compare the floats in the 
        # columns of stats_df to the expected values
        # below, independent of a machine's architecture,
        # truncate all values in stats_df to six digits:
        
        stats_df_trunc = stats_df.apply(BatCalcTester.truncate_series)
        
        pre_int = pd.Series([
            1025.345088,
              32.021010,
             151.056000,
              85.536000,
             103.068000,
              87.840000
            ], name='PrecedingIntrvl'
            )
        lnExpB_start_amp = pd.Series([
            1.751163,
            1.323315,
            -0.876543,
            -3.604043,
            -1.964261,
            -1.688229
            ], name='LnExpB_StartAmp')
        
        lnExpA_end_amp = pd.Series([
            0.003518,
            0.059315,
            0.111705,
            -0.027919,
            0.028525,
            0.015158
            ], name='LnExpA_EndAmp')
        
        amd_2nd_mean = pd.Series([
            0.003516,
            0.059298,
            0.855096,
            0.726407,
            0.813731,
            0.836711
            ], name='Amp2ndMean')
        
        
        expected = pd.concat([
            pre_int,
            lnExpB_start_amp,
            lnExpA_end_amp,
            amd_2nd_mean            
            ], 
            axis='columns',
            )
        expected_idx = pd.Index(['variance','stdev', 'max', 'min','mean', 'median'], name='Stats')
        expected.index = expected_idx

        pd.testing.assert_frame_equal(stats_df_trunc, expected)
        
    # ---------------- Utilities ------------

    #------------------------------------
    # make_simple_df
    #-------------------

    def make_simple_df(self, sono_raw_path):
        '''
        Create a dataframe that includes columns with:
           o straight numbers
           o string that can be turned into number by editing
           o purely string typed
           o time index-like
    
        :param sono_raw_path: where to store the 
            new df (as raw data, i.e. not as a df)
        :type sono_raw_path: str
        :return: the new df
        :rtype: pd.DataFrame
        '''
        
        # from SonoBat: 
        df = pd.DataFrame([
            [5, '0.5 sec','car','30 kHz anti-katydid',19],
            [3, '4.0 sec','bike','10 kHz cutoff',20],
            [1, '4.0 sec','bike','10 kHz cutoff',21]
            ])
        df.columns = ['numbers','mixed','strings', 'enum', 'TimeIndex']
        df.index.name = 'TimeIndex'
        df.set_index('TimeIndex', drop=False, inplace=True)

        df.to_csv(sono_raw_path, index=False)
        return df

    #------------------------------------
    # make_complex_df
    #-------------------
        
    def make_complex_df(self, save_path):
        '''
        Create a df of 4 rows, and five columns:
        
            'Filename',
            'PrecedingIntrvl',
            'LnExpB_StartAmp',
            'Amp2ndMean',
            'LnExpA_EndAmp',
            'TimeIndex'
        
        Datatypes include enough numbers to make
        sense for stats analysis testing
        
        :param save_path: where to store the df
            (as raw data, i.e. not as a df)
        :type save_path: str
        :return: a dataframe
        :rtype: pd.DataFrame
        '''
        
        col_Filename = [
            'barn1_D20220205T192049m784-HiF.wav',
            'barn1_D20220205T192049m784-HiF.wav',
            'barn1_D20220205T192049m784-HiF.wav',
            'barn1_D20220205T192049m784-HiF.wav',
            ]
        col_PrecedingIntrvl = [
            88.848,
            85.536,
            86.832,
            151.056,
            ]
        col_LnExpB_StartAmp = [
            -3.604043,
            -2.472156,
            -0.876543,
            -0.904301,
            ]

        col_Amp2ndMean = [
            0.855096,
            0.827872,
            0.845549,
            0.726407,
            ]
        col_LnExpA_EndAmp = [
            0.008351,
            -0.027919,
            0.111705,
            0.021965           
            ]

        col_TimeIndex = [
            19,
            20,
            21,
            22,
            ]
        sono_cols = [
            'Filename',
            'PrecedingIntrvl',
            'LnExpB_StartAmp',
            'Amp2ndMean',
            'LnExpA_EndAmp',
            'TimeIndex'
            ] 
        
        rows = [[Filename, PrecedingIntrvl,
                 LnExpB_StartAmp, Amp2ndMean,
                 LnExpA_EndAmp, TimeIndex]
                for Filename, PrecedingIntrvl,
                    LnExpB_StartAmp, Amp2ndMean,
                    LnExpA_EndAmp, TimeIndex
                in
                zip(col_Filename,
                    col_PrecedingIntrvl,
                    col_LnExpB_StartAmp,
                    col_Amp2ndMean,
                    col_LnExpA_EndAmp,
                    col_TimeIndex
                    )
                ]

        df_complex = pd.DataFrame(
            rows,
            columns=sono_cols,
            )
        df_complex.set_index('TimeIndex', drop=False, inplace=True)
        
        df_complex.to_csv(save_path, index=False)
        return df_complex

    #------------------------------------
    # truncate
    #-------------------
    
    @staticmethod
    def truncate(num, decimals):
        '''
        Given a float, such as 0.003518287137, 
        truncate at given number of decimals. I.e.
        do not round. Just truncate.
        
        Ex.:  truncate(0.003518987137, 6)
          returns 0.003518
        
        :param num: number to truncate
        :type num: float
        :param decimals: number of decimals to leave standing
        :type decimals: int
        :returns truncated float
        :rtype: float
        '''
        res = int(num*10**decimals)/10**decimals
        return res 
    
    #------------------------------------
    # truncate_series
    #-------------------
    
    @staticmethod 
    def truncate_series(float_ser):
        '''
        Applies the truncate() method to each
        element of a Pandas series. Used for
        df.apply(func), which passes a column
        at a time to func.
        
        :param float_ser:
        :type float_ser:
        :return: a Series in which each float is
            truncated to 6 digits.
        :rtype: pd.Series
        '''
        return float_ser.apply(lambda x: BatCalcTester.truncate(x,6))


# ----------------- Main -------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()