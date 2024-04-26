'''
Created on Apr 24, 2024

@author: paepcke
'''
from data_calcs.pull_daytime_chirps import DaytimeChirpPuller
from pathlib import Path
from pyarrow import feather
import os
import pandas as pd
import tempfile
import unittest


TEST_ALL = True
#TEST_ALL = False


class PullDaychirpsTester(unittest.TestCase):


    def setUp(self):
        self.create_test_files()


    def tearDown(self):
        # Remove the source temp dir
        try:
            for fname in os.listdir(self.tmp_src_dir.name):
                path = os.path.join(self.tmp_src_dir.name, fname)
                os.remove(path)
            self.tmp_src_dir.cleanup()
        except:
            pass
        
        # Remove the destination temp dir
        try:
            for fname in os.listdir(self.tmp_dst_dir.name):
                path = os.path.join(self.tmp_dst_dir.name, fname)
                os.remove(path)
            self.tmp_dst_dir.cleanup()
        except:
            pass

    # ------------------- Tests ----------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_pulling(self):
        _puller = DaytimeChirpPuller(
            self.tmp_dst_dir.name,
            (self.csv_fname1, self.csv_fname2)
            )        

        dst_fnames = os.listdir(self.tmp_dst_dir.name)

        # Expect two .feather files, corresponding to
        # the two source .csv files:

        self.assertEqual(len(dst_fnames), 2)

        # Go through each destination .feather file,
        # and see whether it's correct:
        for dst_fname in dst_fnames:
            dst_path = os.path.join(self.tmp_dst_dir.name, dst_fname)
            self.assertEqual(Path(dst_path).suffix, '.feather')
            
            # Is the content correct?
            df = feather.read_feather(dst_path)
            
            self.assertTrue(df.equals(self.df_expected))
        
    # ------------------- Utilities ----------------
    
    #------------------------------------
    # create_test_files
    #-------------------

    def create_test_files(self):
        '''
        Create two temp directories: tmp_src_dir and tmp_dst_dir.
        Place two .csv files into tmp_src_dir. Each will contain
        three three-column rows. The columns will be called
        
                Filename, Col1, Col2
         
        Two rows will be daytime records, one will be night time.
        Both .csv files will have the same content.
        
        Initializes
          
            o self.tmp_src_dir, which is cleaned out in self.tear_down().
            o self.tmp_dst_dir, which is cleaned out in self.tear_down().
            
            o self.csv_fname1, first file with CSV contents
            o self.csv_fname2, second file with CSV contents
            o self.csv_expected: the contents of the .csv files
                 as rows of strings, including the col names.
            
        '''
        self.tmp_src_dir = tempfile.TemporaryDirectory(dir='/tmp', 
                                                       prefix='daytime_src_',
                                                       delete=True)
        self.tmp_dst_dir = tempfile.TemporaryDirectory(dir='/tmp', 
                                                       prefix='daytime_dst_',
                                                       delete=True)

        tmpfile1 = tempfile.NamedTemporaryFile('w',
                                               suffix='.csv', 
                                               prefix='csv_', 
                                               dir=self.tmp_src_dir.name, 
                                               delete=False)
        
        tmpfile2 = tempfile.NamedTemporaryFile('w',
                                               suffix='.csv', 
                                               prefix='csv_', 
                                               dir=self.tmp_src_dir.name, 
                                               delete=False)

        
        fname_post_sunset = '\\barn1\\grouped_audio\\20220205_to_20220208\\20220205\\barn1_D20220205T222049m784-HiF.wav'
        fname_pre_sunset1 = '\\barn1\\grouped_audio\\20220205_to_20220208\\20220205\\barn1_D20220205T152049m784-HiF.wav'
        fname_pre_sunset2 = '\\barn1\\grouped_audio\\20220205_to_20220208\\20220205\\barn1_D20220205T112049m784-HiF.wav'
        # The .csv file contents
        str_lines = [
                'Filename,Col1,Col2\n',
                f'{fname_pre_sunset1},10,foo\n',
                f'{fname_post_sunset},20,bar\n',
                f'{fname_pre_sunset2},30,fum\n'
                ]
        tmpfile1.writelines(str_lines)
        tmpfile2.writelines(str_lines)
        
        self.csv_fname1 = tmpfile1.name
        self.csv_fname2 = tmpfile2.name
        
        tmpfile1.close()
        tmpfile2.close()

        csv_expected_all = [
            row.strip().split(',')
            for row
            in str_lines
            ]
        # The destinations, if .csv formatted should
        # not have the second column:
        self.csv_expected = csv_expected_all.pop(1) 
        
        # Create a .feather file with same content:
        
        df = pd.DataFrame(
                {'Filename' : [fname_pre_sunset1, fname_post_sunset, fname_pre_sunset2],
                 'Col1'     : [10,20,30],
                 'Col2'     : ['foo', 'bar', 'fum'],
                 }
            )
        #self.feather_fname = os.path.join(self.tmp_src_dir.name, 'df.feather')
        #feather.write_feather(df, self.feather_fname)
        # We expect the middle row to not be copied
        # to the new location, because it was recorded
        # after sunset:
        df_expected_tmp = df.drop(1)
        # Reindex , b/c the two remaining rows are now labeled
        # 0 and 2. We want the two rows to be 0 and 1:
        self.df_expected = df_expected_tmp.reset_index(drop=True) 
        
    
# --------------- Main -----------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()