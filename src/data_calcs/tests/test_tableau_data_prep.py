'''
Created on Jul 9, 2024

@author: paepcke
'''
import unittest

import pandas as pd
from data_calcs.tableau_data_prep import TableauPrepper

TEST_ALL = True
#TEST_ALL = False


class TableauDataPrepTester(unittest.TestCase):


    def setUp(self):
        self.create_test_df()


    def tearDown(self):
        pass

    # ----------------------- Tests ----------------

    #------------------------------------
    # test_add_level 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_level(self):

        # Amputate the number of measure types and
        # values in the class being tested:
        TableauPrepper.measure_types = {
            'amplitudes' : ['PrcntMaxAmpDur', 'Amp1stQrtl'],
            'bandwidths' : ['Bndwdth', 'Bndw32dB']
            }
        
        prepper = TableauPrepper(self.df)
        new_df = prepper.df
        
        expected = pd.DataFrame({'Measure_Type' : [
		       'amplitudes',
		       'amplitudes',
		       'amplitudes',
		       'amplitudes',
		       'amplitudes',
		       'amplitudes',
		       'bandwidths',
		       'bandwidths',
		       'bandwidths',
		       'bandwidths',
		       'bandwidths',
		       'bandwidths'],
		   'Measure_Name' : [
		       'PrcntMaxAmpDur',
		       'PrcntMaxAmpDur',
		       'PrcntMaxAmpDur',
		       'Amp1stQrtl',
		       'Amp1stQrtl',
		       'Amp1stQrtl',
		       'Bndwdth',
		       'Bndwdth',
		       'Bndwdth',
		       'Bndw32dB',
		       'Bndw32dB',
		       'Bndw32dB'],
		   'Measure_Value' : [
		       10,
		       20,
		       30,
		       100,
		       200,
		       300,
		       1000,
		       2000,
		       3000,
		       10000,
		       20000,
		       30000]})
        expected.index.name = 'chirp_num'
        
        pd.testing.assert_frame_equal(new_df, expected)
        
        
    # ------------------- Utilities --------------
    
    #------------------------------------
    # create_test_df
    #-------------------
    
    def create_test_df(self):
        


        df_amplitudes = pd.DataFrame({'PrcntMaxAmpDur' : [10,20,30],
                                      'Amp1stQrtl'     : [100,200,300] 
                                      })
    
        df_bandwidths = pd.DataFrame({'Bndwdth'        : [1000,2000,3000], 
                                      'Bndw32dB'       : [10000,20000,30000] 
                                      })
        
        self.df = pd.concat([df_amplitudes, df_bandwidths], axis='columns')
        return self.df
# -------------------- Main ----------------        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()