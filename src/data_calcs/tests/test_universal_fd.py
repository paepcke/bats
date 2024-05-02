'''
Created on Apr 24, 2024

@author: paepcke
'''

from data_calcs.universal_fd import UniversalFd
from pyarrow import feather
import ast
import gzip
import os
import pandas as pd
import shutil
import tempfile
import unittest

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
            if i == 0:
                self.assertListEqual(row, list(self.df_expected.columns))
            else:
                self.assertEqual(row, list(self.df_expected.iloc[i-1]))

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
        
        
    #------------------------------------
    # test_type_mapping
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_type_mapping(self):
        
        # Work with table:
        
        #     Col1,Col2,filePath,Number
        #     10,foo,\barn1\...\foo.wav,100.3
        #     20,bar,\barn1\...\bar.wav,'200.4'
        #     30,fum,\barn1\...\fum.wav,300        
        
        # Write ints/floats as csv, load and get strings 
        # instead (without type mapping):
        
        with UniversalFd(self.csv_fname, 'r') as fd:
            _cols = fd.readline()
            row = fd.readline()
            self.assertEqual(type(row[0]), str)
        
        # Convert from column names: 
        with UniversalFd(self.csv_fname, 'r', type_map={'Col1' : int}) as fd:
            # Read past the col name line:
            fd.readline()
            for row in fd:
                self.assertEqual(type(row[0]), int)
                
        # Convert from column indices:
        with UniversalFd(self.csv_fname, 'r', type_map={0 : int}) as fd:
            # Read past the col name line:
            fd.readline()
            for row in fd:
                self.assertEqual(type(row[0]), int)

        # Error from trying to convert "100.3" to an int:
        # i.e. int('100.3') fails:
        with UniversalFd(self.csv_fname, 'r', type_map={'Number' : int}) as fd:
            # Read past the col name line:
            fd.readline()
            with self.assertRaises(TypeError):
                fd.readline()
                
        # But the following does work:
        #   int(float('100.3')
        conv_func = lambda str_num: int(float(str_num))
        with UniversalFd(self.csv_fname, 'r', type_map={'Number' : conv_func}) as fd:
            # Read past the col name line:
            fd.readline()
            row = fd.readline()
            self.assertEqual(row[-1], 100)
           
        # This also works:
        #   int(ast.literal_eval('100.3'))
        
        conv_func = lambda str_num: int(ast.literal_eval(str_num))
        with UniversalFd(self.csv_fname, 'r', type_map={'Number' : conv_func}) as fd:
            # Read past the col name line:
            fd.readline()
            row = fd.readline()
            self.assertEqual(row[-1], 100)

        
        # Convert multiple columns:
        with UniversalFd(self.csv_fname, 
                         'r', 
                         type_map={
                             'Col1' : float,
                             'Number' : str
                             }) as fd:
            # Read past the col name line:
            fd.readline()
            for row in fd:
                self.assertEqual(type(row[0]), float)
                self.assertEqual(type(row[-1]), str)
                
                
        # Dataframe with type conversion:
        #
        #    Col1 Col2     filePath  Number
        # 0    10  foo  \barn1\...   100.3
        # 1    20  bar  \barn1\...   200.4
        # 2    30  fum  \barn1\...   300.0        
        with UniversalFd(self.feather_fname, 
                         'r', 
                         type_map={
                             'Col1' : float,
                             'Number' : str
                             }) as fd:
            # Read past the col name line:
            col_names = None
            for row in fd:
                if col_names is None:
                    col_names = fd.readline()
                    self.assertListEqual(col_names, ['Col1', 'Col2', 'filePath', 'Number'])
                    continue
                self.assertEqual(type(row[0]), float)
                self.assertEqual(type(row[-1]), str)
        
    #------------------------------------
    # test_asdf_feather
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_asdf_feather(self):
        #    Col1 Col2     filePath  Number
        # 0    10  foo  \barn1\...   100.3
        # 1    20  bar  \barn1\...   200.4
        # 2    30  fum  \barn1\...   300.0        
        
        type_map = {'Col1' : float,
                    'Number' : str
                    }
        
        df_expected = self.df_expected.astype(type_map)
        
        with UniversalFd(self.feather_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf()
            pd.testing.assert_frame_equal(df, df_expected)

        n_rows = 2
        df_excerpt = df_expected.iloc[0:n_rows]            
        with UniversalFd(self.feather_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf(n_rows=n_rows)
            pd.testing.assert_frame_equal(df, df_excerpt)
            
        n_rows = 0
        df_excerpt = df_expected.iloc[0:n_rows]            
        with UniversalFd(self.feather_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf(n_rows=n_rows)
            pd.testing.assert_frame_equal(df, df_excerpt)

    #------------------------------------
    # test_asdf_csv
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_asdf_csv(self):

        df_expected = pd.read_csv(self.csv_fname)
        with UniversalFd(self.csv_fname, 'r') as fd: 
            df = fd.asdf()
            pd.testing.assert_frame_equal(df, df_expected)

        type_map = {'Col1' : float,
                    'Number' : str
                    }
        df_expected.Col1   = df_expected.Col1.astype(float)
        df_expected.Number = df_expected.Number.astype(str)
        with UniversalFd(self.csv_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf()
            pd.testing.assert_frame_equal(df, df_expected)

        n_rows = 2
        df_excerpt = df_expected.iloc[0:n_rows]            
        with UniversalFd(self.csv_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf(n_rows=n_rows)
            pd.testing.assert_frame_equal(df, df_excerpt)
            
        n_rows = 0
        df_excerpt = df_expected.iloc[0:n_rows]            
        with UniversalFd(self.csv_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf(n_rows=n_rows)
            pd.testing.assert_frame_equal(df, df_excerpt)


    #------------------------------------
    # test_asdf_csvgz
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_asdf_csvgz(self):

        with gzip.open(self.gz_fname, 'rt') as fd:
            df_expected = pd.read_csv(fd)

        with UniversalFd(self.csv_fname, 'r') as fd: 
            df = fd.asdf()
            pd.testing.assert_frame_equal(df, df_expected)
            
        type_map = {'Col1' : float,
                    'Number' : str
                    }
        df_expected.Col1   = df_expected.Col1.astype(float)
        df_expected.Number = df_expected.Number.astype(str)
        with UniversalFd(self.gz_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf()
            pd.testing.assert_frame_equal(df, df_expected)
            
        n_rows = 2
        df_excerpt = df_expected.iloc[0:n_rows]            
        with UniversalFd(self.gz_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf(n_rows=n_rows)
            pd.testing.assert_frame_equal(df, df_excerpt)
            
        n_rows = 0
        df_excerpt = df_expected.iloc[0:n_rows]            
        with UniversalFd(self.gz_fname, 
                         'r', 
                         type_map=type_map) as fd:
            df = fd.asdf(n_rows=n_rows)
            pd.testing.assert_frame_equal(df, df_excerpt)
            
        
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
                'Col1,Col2,filePath,Number\n',
                f'10,foo,{fname_pre_sunset1},100.3\n',
                f"20,bar,{fname_post_sunset},'200.4'\n",
                f'30,fum,{fname_pre_sunset2},300\n'
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
                 'filePath' : [fname_pre_sunset1, fname_post_sunset, fname_pre_sunset2],
                 'Number'   : [100.3, 200.4, 300]
                 }
            )
        self.feather_fname = os.path.join(self.tmpdir.name, 'df.feather')
        feather.write_feather(df, self.feather_fname)
        self.df_expected = df

# ------------------------ Main ------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
