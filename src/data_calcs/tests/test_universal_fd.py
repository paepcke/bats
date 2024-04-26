'''
Created on Apr 24, 2024

@author: paepcke
'''

from data_calcs.universal_fd import UniversalFd
import gzip
import os
import shutil
import tempfile
import unittest
import pandas as pd
from pyarrow import feather

TEST_ALL = True
#TEST_ALL = False

class UniversalFdTester(unittest.TestCase):

    def setUp(self):
        self.create_test_files()

    def tearDown(self):
        try:
            for fname in os.listdir(self.tmpdir.name):
                path = os.path.join(self.tmpdir.name, fname)
                os.remove(path)
            self.tmpdir.cleanup()
        except:
            pass

    #------------------------------------
    # test_universal_fd_writing
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_universal_fd_writing(self):
        
        # Writing regular .csv:
        
        out_path = os.path.join(self.tmpdir.name, 'fd_test.csv')
        csv_fd = UniversalFd(out_path, mode='w')
        for row_arr in self.csv_expected:
            csv_fd.write(row_arr)
        csv_fd.close()
        
        with open(out_path, 'r') as in_fd:
            for i, row in enumerate(in_fd):
                row_arr = row.strip().split(',')
                self.assertEqual(row_arr, self.csv_expected[i])

        # Test .gz output:
                        
        out_path = os.path.join(self.tmpdir.name, 'fd_test.csv.gz')
        gz_fd = UniversalFd(out_path, mode='w')
        for row_arr in self.csv_expected:
            gz_fd.write(row_arr)
        gz_fd.close()
        
        in_fd = UniversalFd(out_path, 'r')
        for i, row in enumerate(in_fd):
            self.assertEqual(row, self.csv_expected[i])

        # Test .feather output:
        out_path = os.path.join(self.tmpdir.name, 'fd_test.feather')
        feather_fd = UniversalFd(out_path, mode='w')
        for row_arr in self.csv_expected:
            feather_fd.write(row_arr)
        feather_fd.close()
        

        in_fd = UniversalFd(out_path, 'r')
        for i, row in enumerate(in_fd):
            self.assertEqual(row, self.csv_expected[i])

    #------------------------------------
    # test_universal_fd_reading
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_universal_fd_reading(self):
        
        # Test csv:
        
        in_fd = UniversalFd(self.csv_fname, 'r')
        for i, row in enumerate(in_fd):
            self.assertEqual(row, self.csv_expected[i])
        in_fd.close()
                        
        # Test .gz:
        
        in_fd = UniversalFd(self.gz_fname, 'r')
        for i, row in enumerate(in_fd):
            self.assertEqual(row, self.csv_expected[i])
        in_fd.close()
           
        # Test .feather:
        in_fd = UniversalFd(self.feather_fname, 'r')
        for i, row in enumerate(in_fd):
            row_strs = [str(el) for el in row]
            self.assertEqual(row_strs, self.csv_expected[i])

    #------------------------------------
    # test_context_manager
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_context_manager(self):
        
        # CSV:
        out_path = os.path.join(self.tmpdir.name, 'fd_test.csv')
        with UniversalFd(out_path, 'w') as fd:
            for row in self.csv_expected:
                fd.write(row)
        # Read back:
        with UniversalFd(out_path, 'r') as fd:
            for i, row in enumerate(fd):
                self.assertEqual(row, self.csv_expected[i])
                
        # .gz:
        out_path = os.path.join(self.tmpdir.name, 'fd_test.csv.gz')
        with UniversalFd(out_path, 'w') as fd:
            for row in self.csv_expected:
                fd.write(row)
        # Read back:
        with UniversalFd(out_path, 'r') as fd:
            for i, row in enumerate(fd):
                self.assertEqual(row, self.csv_expected[i])
            
        # .feather:
        out_path = os.path.join(self.tmpdir.name, 'fd_test.feather')

# ------------------------ Utilities ------------

    #------------------------------------
    # create_test_files
    #-------------------

    def create_test_files(self):
        '''
        Create a temp directory, and place three files into it:

            o csv_*.csv
            o gz_*.csv.gz
            o df.feather

        Each file has three cols and three rows. The third
        column is a file name that includes a recording time.
        The name's format matches SonoBat classification outputs.
        
        Initializes
          
            o self.tmpdir, which is cleaned out in self.tear_down().
            o self.csv_fname, file with CSV contents
            o self.gz_fname, file with CSV contents compressed
            o self.feather_fname, file in .feather content that loads into dataframe

            o self.csv_expected: the contents of the .csv and .csv.gz files
                 as rows of strings
            o self.df_expected: the contents of df.feather as a dataframe
            
        '''
        self.tmpdir = tempfile.TemporaryDirectory(dir='/tmp', 
                                                  prefix='daytime_',
                                                  delete=True)
        tmpfile = tempfile.NamedTemporaryFile('w',
                                              suffix='.csv', 
                                              prefix='csv_', 
                                              dir=self.tmpdir.name, 
                                              delete=False)
        fname_post_sunset = '\\barn1\\grouped_audio\\20220205_to_20220208\\20220205\\barn1_D20220205T222049m784-HiF.wav'
        fname_pre_sunset1 = '\\barn1\\grouped_audio\\20220205_to_20220208\\20220205\\barn1_D20220205T152049m784-HiF.wav'
        fname_pre_sunset2 = '\\barn1\\grouped_audio\\20220205_to_20220208\\20220205\\barn1_D20220205T112049m784-HiF.wav'
        # The .csv file contents
        str_lines = [
                'Col1,Col2,filePath\n',
                f'10,foo,{fname_pre_sunset1}\n',
                f'20,bar,{fname_post_sunset}\n',
                f'30,fum,{fname_pre_sunset2}\n'
                ]
        tmpfile.writelines(str_lines)
        self.csv_fname = tmpfile.name
        tmpfile.close()
        self.csv_expected = [
            row.strip().split(',')
            for row
            in str_lines
            ]
        
        # Create a .csv.gz file with same content:
        
        self.gz_fname = tmpfile.name + '.gz'
        with open(tmpfile.name, 'rb') as f_in:
            # Create a new gzipped file in write mode
            with gzip.open(self.gz_fname, 'wb') as f_out:
                # Copy the contents of the existing file to the gzipped file
                shutil.copyfileobj(f_in, f_out)
        f_in.close()
        f_out.close()
        
        # Create a .feather file with same content:
        df = pd.DataFrame(
                {'Col1' : [10,20,30],
                 'Col2' : ['foo', 'bar', 'fum'],
                 'filePath' : [fname_pre_sunset1, fname_post_sunset, fname_pre_sunset2]
                 }
            )
        self.feather_fname = os.path.join(self.tmpdir.name, 'df.feather')
        feather.write_feather(df, self.feather_fname)
        self.df_expected = df

# ------------------------ Main ------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
