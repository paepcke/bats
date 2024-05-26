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
    
    Instances of the class also support type mapping: converting
    .csv values to types other than strings. Example: 
    
    given .csv file:
    
         Col1,Col2
          10 , '20'

    would read as:
         "Col1,Col2"
         "10,'20'"
    
    And converting the first row to an array via split(',') would yield:
       ['10', "'20'\n"]
       
    Using UniversalFd, one can instead obtain:
       [10, 20]

    Examples:
    
    prints arrays:
        with UniversalFd('/myfile.csv', 'r') as fd:
            for row in fd:
                print(row)
        
    Col2 is converted to integers:
        with UniversalFd('/myfile.csv', 'r', type_map={'Col2' : int) as fd:
            for row in fd:
                print(row)
       
    Print .feather file row by row:
        with UniversalFd('/myfile.feather', 'r') as fd:
            for row in fd:
                print(row)

    Obtain dataframe from a feather file:
        with UniversalFd('/myfile.csv', 'r', type_map={'Col2' : int) as fd:
            df = fd.asdf()
    

    A UniversalFd can thus write and read:
    
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

    # Fieldnames: empty tuple: no col names
    def __init__(self, fname, mode, type_map=None):

        self.fname = fname 
        
        self._initialize_type_map(type_map)

        if mode == 'w':
            self._initialize_writing(fname)
            
        elif mode == 'r':
            self._initialize_reading(fname)
            
        else:
            raise ValueError(f"Mode argument must be 'w' or 'r', not {mode}")
        
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
    # def write_df
    #-------------------
    
    def write_df(self, df):
        '''
        Write the given df according to self.filetype: as
        .csv, .csv.gz, or .feather
        
        :param df: dataframe to write
        :type df: pd.DataFrame
        '''
        if self.filetype in ('csv', 'gz'):
            df.to_csv(self.fname, compression='infer')
        elif self.filetype == 'feather':
            try:
                df.to_feather(self.fname)
            except Exception as e:
                msg = f"Feather write error. Tip: columns written to .feather must have the same type. Ensure that this is true for all df columns. ({e})"
                raise TypeError(msg)
        else:
            raise TypeError(f"Can only write df to .csv, .csv.gz, or .feather, not {self.filetype}")                
                
    #------------------------------------
    # readline
    #-------------------
    
    def readline(self, **kwargs):
        
        if self.filetype in ('csv', 'gz'):
            row_raw = next(self.reader)
            if self.conversion_dict is not None:
                row = self._convert_types(row_raw)
                return row
            return row_raw
        
        # We must be reading from a .feather file
        # So we created self.content, which is
        #
        #   [['col1', 'col2'], [10, 'foo'], [20, 'bar']]
        #
        # We also have self.line_num pointing into the
        # outer array:
        
        row_raw = self.content[self.line_num]
        # For .csv and .csv.gz files: convert field value
        # types on the fly. For .feather files (i.e. dataframes),
        # the conversion was done during initialization:
        if self.conversion_dict is not None and self.filetype != 'feather':
            row = self._convert_types(row_raw)
        else:
            row = row_raw
        self.line_num += 1
        return row  
    
    #------------------------------------
    # read
    #-------------------
    
    def read(self):
        '''
        Read entire file
        '''
        big_str = self.fd.read()
        self.fd.close()
        if self.conversion_dict is None:
            return big_str
        
        # Must perform the conversion on all 
        # the lines:
        big_arr = big_str.strip().split('\n')
        converted_arr = [self._convert_types(row)
                         for row
                         in big_arr
                         ] 
        return '\n'.join(converted_arr)
    
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
            if len(self.table) > 0:
                df = pd.DataFrame(self.table[1:], columns=self.table[0])
                df.to_feather(self.out_name)

    #------------------------------------
    # line_num (property)
    #-------------------
    
    @property
    def line_num(self):
        if self.filetype == 'feather':
            return self._line_num
        else:
            return self.reader.line_num

    @line_num.setter
    def line_num(self, new_val):
        if self.filetype == 'feather':
            self._line_num = new_val
        else:
            raise AttributeError(f"Cannot set line number for .csv or .csv.gz files")
    
    #------------------------------------
    # asdf
    #-------------------
    
    def asdf(self, n_rows=None, index_col=None):
        '''
        Have the UniversalFd read the file straight into a
        dataframe. Because of the initializations, this
        method can handle all the supported file types
        
        If idex_column is provided, it must be the name of a
        column, or the index to a column. Used only for 
        .csv and .csv.gz. This information is needed for a 
        following .csv file:
           
              Idx  Col1   Col2
               0   'foo'  'bar'
               1   'blue' 'green'
               
        where the intention for the resulting dataframe is:
        
                   Col1    Col2
            Idx    
             0     'foo'   'bar'
             1     'blue'  'green'
    
        Similarly, when the original df's index did not have a name,
        the .csv often looks like this:
        
              ,Col1   Col2
               0   'foo'  'bar'
               1   'blue' 'green'
        
        in which case the df can look like:
        
                  Unnamed   Col1    Col2
                    
             0      0    	'foo'   'bar'
             1      1    	'blue'  'green'
        
        In this case, pass index_col=0, since no name is
        available to designate as the index name:
        
        :param n_rows: how many rows to read. None means all.
        :type n_rows: union[None | int]
        :param index_col: which of the columns, if any, to use
            as the name of the dataframe index, rather than a
            regular column
        :type index_col: union[None | str]
        :return the dataframe that was read from the disk
        :rtype pd.DataFrame
        '''
        
        if n_rows is not None and type(n_rows) != int:
            raise TypeError(f"Argument n_rows must be None or integer, not {n_rows}")
        
        if self.filetype == 'feather':
            if n_rows is not None:
                df_excerpt = self.df.iloc[0:n_rows]
            else:
                df_excerpt = self.df
            return df_excerpt

        elif self.filetype == 'csv':
            with open(self.fname, 'rt') as fd:
                try:
                    new_df = pd.read_csv(fd, header=0, index_col=index_col, engine='c')
                except ValueError:
                    msg = f"Could not read {fd.name} with index_col={index_col}. Maybe '{index_col}' not present in file?"
                    raise ValueError(msg)
            # Make any necessary column type adjustments:
            df_typed = self._type_map_df(new_df)
            if n_rows is not None:
                df_excerpt = df_typed.iloc[0:n_rows]
            else:
                df_excerpt = df_typed
            return df_excerpt
            
        elif self.filetype == 'gz':
            with gzip.open(self.fname, 'rt') as fd:
                new_df = pd.read_csv(fd, header=0, index_col=index_col, engine='c')
            # Make any necessary column type adjustments:                
            df_typed = self._type_map_df(new_df)
            if n_rows is not None:
                df_excerpt = df_typed.iloc[0:n_rows]
            else:
                df_excerpt = df_typed
            return df_excerpt

        else:
            raise NotImplementedError("The asdf() method currently works only with .feather, .csv, and .gz files")
    
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
            cols = list(df.columns)
            if len(self.conversion_dict) > 0:
                df = self._type_map_df(df)
            # Save the df; it can be retrieved with 
            #   fd.asdf()
            self.df = df
            # To feed out row by row via readline(),
            # Construct an array of arrays, like:
            #   [['col1', 'col2'], [10, 'foo'], [20, 'bar']]
            # Python array from df row values:            
            rows = df.to_numpy().tolist()
            
            # Get like
            #   [['col1', 'col2'], [10, 'foo'], [20, 'bar']]
            self.content = [cols] + rows
            # Scan pointer:
            self.line_num = 0
        else:
            raise TypeError(f"UniversalFd is for .csv, .csv.gz, or .feather files, not {in_fname}")


    #------------------------------------
    # _type_map_df
    #-------------------


    def _type_map_df(self, df):
        '''
        Given a DataFrame, see whether we have a 
        self.conversion_dict for converting the types
        of some dataframe columns. If not, return the
        df that was passed in. Else, do the conversions,
        and return the modified df.
        
        :param df: dataframe to type-adjust
        :type df: pd.DataFrame
        :return: a df with type-adjusted columns
        :rtype: pd.DataFrame
        '''
        
        # If there is no conversion dict, not
        # type adjustments are needed:
        if self.conversion_dict in (None, {}):
            return df
        
        # Must convert the integer keys into
        # column names for the df.astype() below
        # to work:
        tmp_cdict = {df.columns[int_key] : conv_func
                     for int_key, conv_func
                     in self.conversion_dict.items()
                     if type(int_key) == int
                     }
        self.conversion_dict = tmp_cdict
        df = df.astype(self.conversion_dict)
        return df

    #------------------------------------
    # _initialize_type_map
    #-------------------
    
    def _initialize_type_map(self, type_map):
        
        # For speed, create a dict <column-number : <type-convert-func>
        self.conversion_dict = {}

        if type_map in (None, {}):
            return
            
        if type(type_map) != dict:
            err_msg = f"Type map must be None, or a dict of col name:function, not {type_map}"
            raise TypeError(err_msg)

        # If all type_map keys are integers, we interpret
        # them as column indices. Else, we must assume that
        # the first row is col names. There is no peek() in 
        # csv, so just get the first line:
        
        keys_are_idxs = all([type(key) == int for key in type_map.keys()])
        
        if keys_are_idxs:
            self.col_names = list(type_map.keys())
        else:
            # Read col names from file:        
            with UniversalFd(self.fname, 'r') as fd:
                # If file is empty, col_names will be empty string:
                self.col_names = fd.readline()
                # Ensure that all the type map keys
                # correspond to columns:
                for key in type_map.keys():
                    if key not in self.col_names:
                        err_msg = f"Keys in type_map must be column names or column numbers, not {key}"
                        raise ValueError(err_msg) 

        # Initialize dict col-idx : conversion-func:
        for type_map_key, conv_func in type_map.items():
            # We already checked above that the .index() will succeed:
            col_idx = self.col_names.index(type_map_key)
            self.conversion_dict[col_idx] = conv_func

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
    # _convert_types
    #-------------------
        
    def _convert_types(self, row):
        
        for col_idx, conv_func in self.conversion_dict.items():
            try:
                row[col_idx] = conv_func(row[col_idx])
            except ValueError:
                # If working on first row, tolerate conversion 
                # error, because the row may be the col names:
                line_num = self.reader.line_num
                if line_num == 1:
                    return row
                # All subsequent rows must be convertible:
                col = self.col_names[col_idx]
                raise TypeError(f"Cannot convert---row {line_num}, col '{col}', val {row[col_idx]}, conversion: {conv_func}")
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
