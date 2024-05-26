'''
Created on May 24, 2024

@author: paepcke
'''
from data_calcs.data_calculations import Activities, FileType, Localization
from tempfile import NamedTemporaryFile, TemporaryDirectory
import os
import pandas as pd
import unittest
#from data_calcs.universal_fd import UniversalFd


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

    #------------------------------------
    # test__extract_column
    #-------------------
    

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__extract_column(self):
        
        activities = Activities(self.tmpdir, data_dir=self.tst_df_nm, fid_map_file=self.fid_map_file)
        col_name = 'col2'
        ser, save_path = activities._extract_column(self.tst_df_nm, col_name, self.tmpdir.name, prefix='col_extract_tst').values()
        ser_copy = pd.read_csv(save_path, usecols=[col_name])[col_name]
        pd.testing.assert_series_equal(ser, ser_copy)

    #------------------------------------
    # test__concat_files
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__concat_files(self):
        activities    = Activities(self.tmpdir, data_dir=None, fid_map_file=self.fid_map_file)
        df_paths      = [self.tst_df_nm, self.tst_df_nm_feather]
        idx_cols      = ['Idx', None]
        df, save_path = activities._concat_files(df_paths,
                                                 idx_columns=idx_cols, 
                                                 dst_dir=self.tmpdir.name, 
                                                 out_file_type=FileType.CSV, 
                                                 augment=False).values()
        df_recovered = pd.read_csv(save_path, index_col=0)
        # Or: df_recovered = UniversalFd(save_path, 'r').asdf(index_col=0)
        # NOTE: on reading csv, everything is turned into
        #       strings. Whereas the concatenation result, 'df', 
        #       preserved the ints. To test equivalence, we
        #       need to make the df cols into strings as well,
        #       though this is just to get the test comparison
        #       to pass:
        df.col1 = df.col1.astype(str) 
        df.col2 = df.col2.astype(str) 
        
        
        pd.testing.assert_frame_equal(df, df_recovered)
        
        # If we output as .feather, and read back we should not
        # have to do the string shenanigans. But: fail because
        # result df has mixed types:
        with self.assertRaises(TypeError):
            df, save_path = activities._concat_files(df_paths,
                                                     idx_columns=idx_cols, 
                                                     dst_dir=self.tmpdir.name, 
                                                     out_file_type=FileType.FEATHER, 
                                                     augment=False).values()
                                                     
        # Try .feather with like-typed dfs:
        
        df_paths      = [self.tst_df_nm, self.tst_df_nm]
        idx_cols      = ['Idx', 'Idx']
        df, save_path = activities._concat_files(df_paths,
                                                 idx_columns=idx_cols, 
                                                 dst_dir=self.tmpdir.name, 
                                                 out_file_type=FileType.FEATHER, 
                                                 augment=False).values()
        df_recovered = pd.read_feather(save_path)   
        pd.testing.assert_frame_equal(df, df_recovered)
        
        # Now with augmentation of rec_datetime, and sin/cos cols:
        df_paths      = [self.tst_split_nm, self.tst_split_nm]
        df, save_path = activities._concat_files(df_paths,
                                                 idx_columns=[0, 0],   # Indicate that left-col of csv is just the index 
                                                 dst_dir=self.tmpdir.name, 
                                                 out_file_type=FileType.CSV, 
                                                 augment=True).values()
        df_recovered = pd.read_csv(save_path, index_col=0)
        self.assertListEqual(list(df.index), list(df_recovered.index))
        self.assertListEqual(list(df.columns), list(df_recovered.columns))
        self.assertListEqual(list(df_recovered.Knee), [10.0]*len(df))
        expected_col_nms = ['Knee', 'file_id', 'rec_datetime', 'is_daytime', 'sin_hr', 'cos_hr',
                            'sin_day', 'cos_day', 'sin_month', 'cos_month', 'sin_year', 'cos_year']
        self.assertListEqual(list(df.columns), expected_col_nms)
        
    
    # ------------------------ Utilities ---------------
    
    #------------------------------------
    # create_tst_files
    #-------------------
    
    def create_tst_files(self):
        '''
        Writes two dfs, one .csv, the other .feather.
        Puts respective temp file names into self.tst_df_nm,
        and self.tst_df_nm_feather, respectively 
        '''
        
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
    
        # Now a second df, this time in .feather format:
        df_feather = pd.DataFrame({'col1' : ['foo', 'bar', 'fum', 'baz', 'buz'],
                           'col2' : ['blue', 'green', 'red', 'purple', 'white']})
        df_feather.index.name = 'Idx'
        
        with NamedTemporaryFile(dir=self.tmpdir.name, 
                                prefix='col_extraction_', 
                                suffix='.feather', 
                                delete=False) as fd:
            self.tst_df_nm_feather = fd.name  
            df_feather.to_feather(fd.name)
        
        # For testing concat with augmentation:
        
        # A df that looks somewhat like a bat measures file
        
        df_split = pd.DataFrame({'Knee'    : 10.0,
                                 'file_id' : 0
                                 }, index=pd.RangeIndex(2))
        with NamedTemporaryFile(dir=self.tmpdir.name, 
                                prefix='col_extraction_', 
                                suffix='.csv', 
                                delete=False) as fd:
            self.tst_split_nm = fd.name  
            df_split.to_csv(fd)
        # We also need the file_id --> .wav file map:
        mapping_content = ('Filename,file_id\n' 
        				   'barn1_D20220205T192049m784-HiF.wav,0\n'
    					   'barn1_D20220205T200541m567-Myca-Myca.wav,1\n'
    					   'barn1_D20220205T202248m042-HiF.wav,2\n')
        self.fid_map_file = os.path.join(self.tmpdir.name, Localization.measures_root, 'split_filename_to_id_testing.csv')
        with open(self.fid_map_file, 'w') as fd:
            fd.write(mapping_content)
        
        
        
        
# --------------------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()