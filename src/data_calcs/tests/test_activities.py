'''
Created on May 24, 2024

@author: paepcke
'''
from data_calcs.data_calculations import Activities
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pandas as pd
import unittest


TEST_ALL = True
#TEST_ALL = False


class ActivitiesTester(unittest.TestCase):


    def setUp(self):
        self.create_tst_files()


    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls)->None:
        super(ActivitiesTester, cls).tearDownClass()
        cls.tmpdir.cleanup()

# -------------------------- Tests --------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_extract_column(self):
        
        activities = Activities(self.tmpdir, data_dir=self.tst_df_nm)
        col_name = 'col2'
        ser, save_path = activities._extract_column(self.tst_df_nm, col_name, self.tmpdir.name, prefix='col_extract_tst').values()
        ser_copy = pd.read_csv(save_path, usecols=[col_name])[col_name]
        pd.testing.assert_series_equal(ser, ser_copy)

    # ------------------------ Utilities ---------------
    
    #------------------------------------
    # create_tst_files
    #-------------------
    
    def create_tst_files(self):
        
        ActivitiesTester.tmpdir = TemporaryDirectory(dir='/tmp', prefix='activities_tsts_', delete=False)

        df = pd.DataFrame({'col1' : [1,2,3,4,5],
                           'col2' : [6,7,8,9,10]})
        df.index.name = 'Idx'
        
        with NamedTemporaryFile(dir=self.tmpdir.name, 
                                prefix='col_extraction_', 
                                suffix='.csv', 
                                delete=False) as fd:
            self.tst_df_nm = fd.name  
            df.to_csv(fd)
            
            
# --------------------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()