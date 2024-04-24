'''
Created on Apr 24, 2024

@author: paepcke
'''
from pathlib import Path, PosixPath
import csv
import gzip
import pandas as pd

# --------------------- Class UniversalFd -----------

class UniversalFd:
    '''
    Instances behave like csv readers and writers. But
    they handle .csv, .csv.gz, and .feather files.  
    The input or output filename extension determines the
    implementations of reading and writing.
    
    A UniversalFd can thus write to:
    
        o .csv files
        o .csv.gz files
        o .feather files
        
    Use the write() method to write one row, either as
    a string (for csv/.gz files) or an array (.feather) files.
    The write() method accepts string-formated rows, or
    lists. However, when writing to .feather files, string
    inputs will be turned into array elements naively: splitting
    by comma.
    
    Don't forget to call the close() method. This is particularly 
    important when writing to .feather files. Alternatively, use
    the 'with' context managager:

        path = my_path.csv
    or  path = my_path.csv.gz
    or  path = my_path.csv.feather
        
        with UniversalFd(path, 'w') as fd:
            for row in my_python_arrays:
                fd.write(row)
        
        with UniversalFd(path, 'r') as fd:
            for i, row in enumerate(fd):
                self.assertEqual(row, my_python_arrays[i])
    '''
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, fname, mode):

        if mode == 'w':
            self._initialize_writing(fname)
            
        elif mode == 'r':
            self._initialize_reading(fname)
            
        else:
            raise ValueError(f"Mode argument must be 'w' or 'r', not {mode}")
    
    
    #------------------------------------
    # _initialize_writing
    #-------------------
    
    def _initialize_writing(self, out_fname):
        
        self.direction = 'write'
        
        if type(out_fname) == str:
            self.out_name = Path(out_fname)
        elif type(out_fname) == PosixPath:
            self.out_name = out_fname
        else:
            raise TypeError(f"Outfile name must be string or a pathlib.Path, not {out_fname}")
           
        # Now output file is a Path instance: 
        extension = self.out_name.suffix
        if extension == '.csv':
            self.filetype = 'csv'
            self.fd = open(self.out_name, 'wt')
            self.writer = csv.writer(self.fd)
        elif extension == '.gz':
            self.filetype = 'gz'
            self.fd = gzip.open(self.out_name, 'wt')
            self.writer = csv.writer(self.fd)
        elif extension == '.feather':
            self.filetype = 'feather'
            # Fill a table with row lists in self.write(),
            # then convert to df, and write to disk in call to close() 
            self.table = []
        else:
            raise TypeError(f"UniversalFd is for .csv, .csv.gz, or .feather files, not {out_fname}")

    #------------------------------------
    # _initialize_reading
    #-------------------
    
    def _initialize_reading(self, in_fname):
        
        self.direction = 'read'
        
        if type(in_fname) == str:
            self.in_name = Path(in_fname)
        elif type(in_fname) == PosixPath:
            self.in_name = in_fname
        else:
            raise TypeError(f"Infile name must be string or a pathlib.Path, not {in_fname}")
           
        # Now input file is a Path instance: 
        extension = self.in_name.suffix
        if extension == '.csv':
            self.filetype = 'csv'
            self.fd = open(self.in_name, 'rt')
            self.reader = csv.reader(self.fd)
        elif extension == '.gz':
            self.filetype = 'gz'
            self.fd = gzip.open(self.in_name, 'rt')
            self.reader = csv.reader(self.fd)
        elif extension == '.feather':
            self.filetype = 'feather'
            # Read the file into a dataframe:
            df = pd.read_feather(self.in_name)
            # Construct an array of arrays, like:
            #   [['col1', 'col2'], [10, 'foo'], [20, 'bar']]
            # Python array from df row values:            
            rows = df.to_numpy().tolist()
            cols = list(df.columns)
            # Get like
            #   [['col1', 'col2'], [10, 'foo'], [20, 'bar']]
            self.content = [cols] + rows
            # Scan pointer:
            self.scan_pos = 0
        else:
            raise TypeError(f"UniversalFd is for .csv, .csv.gz, or .feather files, not {in_fname}")


    #------------------------------------
    # write
    #-------------------

    def write(self, str_or_row):
        
        if self.filetype in ('csv', 'gz'):
            if type(str_or_row) == str:
                row_arr = str_or_row.split(',') 
                self.writer.writerow(row_arr)
            elif type(str_or_row) == list:
                self.writer.writerow(str_or_row)
            else:
                raise TypeError(f"Arg to write() must be string or list of strings, not {str_or_row}")                                   
        # Must be .feather destination:
        else:
            if type(str_or_row) == str:
                new_row = self.table.append(str_or_row.split(','))
            else:
                new_row = str_or_row
            self.table.append(new_row)
                
    #------------------------------------
    # readline
    #-------------------
    
    def readline(self, **kwargs):
        
        if self.filetype in ('csv', 'gz'):
            res = next(self.reader)
            return res
        
        # We must be reading from a .feather file
        # So we created self.content, which is
        #
        #   [['col1', 'col2'], [10, 'foo'], [20, 'bar']]
        #
        # We also have self.scan_pos pointing into the
        # outer array:
        
        row = self.content[self.scan_pos]
        self.scan_pos += 1
        return row  
    
    #------------------------------------
    # close
    #-------------------
    
    def close(self):
        '''
        If this instance represents a .csv or .gz file,
        the self.fd's close() method is called. If
        the instance represents a .feather file, the 
        self.table python array of arrays is turned into
        a df, which is then written to disk as a feather file. 
        '''
        
        if self.filetype in ['csv', 'gz']:
            self.fd.close()
            return
        
        if self.direction == 'write':
            # Must be a feather:
            df = pd.DataFrame(self.table[1:], columns=self.table[0])
            df.to_feather(self.out_name)

    #------------------------------------
    # __iter__
    #-------------------
    
    def __iter__(self):
        if self.direction == 'read':
            if self.filetype in ('csv', 'gz'):
                return self
            else:
                # .feather:
                # Row by row, starting with column headers
                return iter(self.content)
        else:
            raise TypeError("Cannot iterate over a writer")
    
    
    #------------------------------------
    # __next__
    #-------------------
    
    def __next__(self):
        row = self.readline()
        return row
        
    #------------------------------------
    # __enter__ and __exit__
    #-------------------
    
    # Context handler for UniversalFd
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return None
    

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        '''
        Depending on whether the instance is for writing 
        or reading, representation will be like:
        
              <UniversalFd type=out-csv out.csv at 0xab4233>
              <UniversalFd type=in-csv in.csv at 0xab4233>
        '''
        try:
            filetype = self.filetype
        except AttributeError:
            # Instance is not fully instantiated yet:
            filetype = 'pending'
        
        if self.direction == 'read':
            direction = 'in'
            try:
                fname = self.in_name.name
            except AttributeError:
                # Instance is not fully instantiated yet:
                fname = 'fname-pending'

        
        elif self.direction == 'write':
            direction = 'out'
            try:
                fname = self.out_name.name
            except AttributeError:
                # Instance is not fully instantiated yet:
                fname = 'fname-pending'
        else:
            direction = 'dir-pending'

        summary = f"<UniversalFd type={direction}-{filetype} {fname} at {hex(id(self))}>"
        return summary
