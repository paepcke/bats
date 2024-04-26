'''
Created on Apr 20, 2024

@author: paepcke
'''
from _datetime import datetime #timezone, timedelta
from data_calcs.daytime_file_selection import DaytimeFileSelector
from pyarrow import feather
import csv
import gzip
import os
import pandas as pd
import shutil
import tempfile
import unittest

TEST_ALL = True
#TEST_ALL = False

class DaytimeTimeSelectionTester(unittest.TestCase):

    def setUp(self):
        self.selector = DaytimeFileSelector()
        self.tzinfo = self.selector.timezone
        self.create_test_files()

    def tearDown(self):
        try:
            for fname in os.listdir(self.tmpdir.name):
                path = os.path.join(self.tmpdir.name, fname)
                os.remove(path)
            self.tmpdir.cleanup()
        except:
            pass

    # ----------------- Tests --------------

    #------------------------------------
    # test_time_from_fname 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_time_from_fname(self):
        
        fname = '/foo/bar/barn1_D20220205T192049m784-HiF.wav'
        dt = self.selector.time_from_fname(fname)
        expected = datetime(2022, 2, 5, 
                            hour=19, 
                            minute=20, 
                            second=49, 
                            tzinfo=self.tzinfo)
        self.assertEqual(dt, expected)
        
        # Bad fname:
        fname = '/foo/bar/barn1_DD0220205T192049m784-HiF.wav'
        with self.assertRaises(ValueError):
            dt = self.selector.time_from_fname(fname)

    #------------------------------------
    # test_sunset_time
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_sunset_time(self):
        
        the_date = datetime(2022, 2, 5).date()
        sunrise, sunset = self.selector.sunrise_sunset_times(
            the_date, round_to_minute=True)

        expected_sunrise = datetime(2022, 2, 5, 
                                    hour=8, minute=9, 
                                    tzinfo=self.tzinfo
                                    )
        expected_sunset = datetime(2022, 2, 5, 
                                   hour=18, minute=38, 
                                   tzinfo=self.tzinfo
                                   )

        self.assertEqual(sunrise, expected_sunrise)
        self.assertEqual(sunset, expected_sunset)

    #------------------------------------
    # test_open_file
    #-------------------
    
    def test_open_file(self):
        selector = DaytimeFileSelector()

        # Test csv opening:
        
        expected = self.csv_expected
        
        reader, csv_fd = selector.open_file(self.csv_fname)
        for i, row in enumerate(reader):
            self.assertEqual(row, expected[i])
        csv_fd.close()
        
        # Test .gz opening:
        
        reader, gz_fd = selector.open_file(self.gz_fname)
        for i, row in enumerate(reader):
            self.assertEqual(row, expected[i])
        gz_fd.close()

        # Test .feather opening:
            
        expected = self.df_expected
                
        reader, feather_fd = selector.open_file(self.feather_fname)
        col_names = next(reader)
        self.assertListEqual(col_names, list(expected.columns))
        for i, row in enumerate(reader):
            self.assertTrue(row == list(expected.iloc[i]))
        if feather_fd is not None:
            feather_fd.close()
        
    #------------------------------------
    # test_daytime_recordings
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_daytime_recordings(self):
    
        out_fpath = os.path.join(self.tmpdir.name, 'out.csv')
         
        # Place the pre-sunset files into out_fpath:
        self.selector.daytime_recordings(self.csv_fname, out_fpath, 'filePath')
        with open(out_fpath, 'r') as fd:
            reader = csv.reader(fd)
            content = list(reader)

        # Col names:
        self.assertListEqual(content[0], self.csv_expected[0])
        self.assertListEqual(content[1], self.csv_expected[1])
        # The 22:30:49 file should have been filtered out:
        self.assertListEqual(content[2], self.csv_expected[3])
        
    #------------------------------------
    # test_is_daytime_recording 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_is_daytime_recording(self):
        
        fname = '/foo/bar/barn1_D20220205T222049m784-HiF.wav'
        decision = self.selector.is_daytime_recording(fname)
        self.assertFalse(decision)
        
        fname = 'barn1_D20220205T152049m784-HiF.csv'
        decision = self.selector.is_daytime_recording(fname)
        self.assertTrue(decision)
        
        with self.assertRaises(ValueError):
            self.selector.is_daytime_recording('33')
            
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