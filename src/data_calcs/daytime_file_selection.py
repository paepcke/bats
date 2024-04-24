'''
Created on Apr 20, 2024

@author: paepcke
'''
from astral import geocoder, sun
from data_calcs.utils import Utils
from datetime import datetime, timezone, timedelta
from pathlib import Path, PosixPath
import csv
import gzip
import os
import pyarrow
import re
import sys
import pandas as pd

class DaytimeFileSelector:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, *dirs):
        '''
        Constructor
        '''
        # Jasper Ridge Research Station:
        self.lat = 37.408 
        self.lon = -122.2275
        # Pacific Daylight Time offset from UTC
        utc_offset = -7
        self.timezone = timezone(timedelta(hours=utc_offset), name='PDT')
        
        self.geo_db = geocoder.database()
        geocoder.add_locations(
            (f"{'Jasper Ridge Biological Preserve'},"
             f"{'California'},"
             f"{'US/Pacific Daylight Time'},"
             f"{self.lat},"
             f"{self.lon}"),
            self.geo_db)
        self.jr_loc = geocoder.lookup('Jasper Ridge Biological Preserve', self.geo_db)

    #------------------------------------
    # daytime_recordings
    #-------------------
    
    def daytime_recordings(self, in_file, out_file, file_name_col):
        
        if not os.path.exists(in_file):
            print(f"Input file {in_file} does not exists; aborting.")
            sys.exit(1)
        # Outer try to ensure closure of files:
        try:
            try:
                if out_file is None:
                    out_fd = sys.stdout
                else:
                    if os.path.exists(out_file):
                        decision = input(f"Outfile {out_file} exists; overwrite? (y/n)")
                        if decision.lower() in ('', 'n', 'no'):
                            print("Aborting, file untouched")
                            sys.exit(1)
                    try:
                        out_fd = UniversalFd(out_file)
                    except Exception as e:
                        print(f"Cannot create out file {out_file} ({e}); aborting")
                        sys.exit(1)
            except Exception as out_exc:
                raise ValueError(f"Could not create outfile {out_file}: {out_exc}")
                
            reader, self.fd_to_close = self.open_file(in_file)
            try:
                cols = next(reader)
            except StopIteration:
                raise ValueError(f"File {in_file} was empty")
            
            # Find the column that holds the filenames:
            try:
                file_col_idx = cols.index(file_name_col)
            except ValueError:
                raise ValueError(f"File {in_file} has no column {file_name_col}")
    
            out_fd.write(cols)
            for row in reader:
                try:
                    fname = row[file_col_idx]
                    recording_time = self.time_from_fname(fname)
                    sunset = self.sunset_time(recording_time, round_to_minute=True)
                    if recording_time <= sunset:
                        out_fd.write(row)
                except IndexError:
                    raise ValueError(f"Row number {file_col_idx} not found in row {row}")
        finally:
            try:
                self.fd_to_close.close()
            except:
                pass
            try:
                out_fd.close()
            except:
                pass
                
    #------------------------------------
    # open_file
    #-------------------
    
    def open_file(self, in_file):
        '''
        Given a file path, open it and return
        a file-like iterable that returns one
        row of data at a time.
        
        Supporte are:
            .csv.
            .csv.gz
            .feather
            
        Note that for .feather files, the entire file
        is loaded into memory. One could re-write to load
        pyarrow chunks instead.
        
        Returns the reader, and an fd for whose closure the caller
        is responsible.
         
        :param in_file: file for which iterator is to be created
        :type in_file: str
        '''

        if in_file.endswith('.csv'):
            csv_fd = open(in_file, 'r')
            reader = csv.reader(csv_fd)
            return(reader, csv_fd)
        
        if in_file.endswith('.gz'):
            gz_fd = gzip.open(in_file, 'rt')
            reader = csv.reader(gz_fd)
            return(reader, gz_fd)

        if in_file.endswith('.feather'):
            df = pyarrow.feather.read_feather(in_file)
            return(DataFrameRows(df), None)

    #------------------------------------
    # sunset_time
    #-------------------
    
    def sunset_time(self, date, round_to_minute=True):
        '''
        Given a date return the time of
        sunset at the Jasper Ridge Preserve.
        All computations are in Pacific Daylight
        Time (PDT).
         
        :param date: date whose sunset is to be computed
        :type date: datetime
        :param round_to_minute: round to the nearest minute
        :type round_to_minute: bool
        :return the (possibly rounded) sunset time at Jasper Ridge,
            Stanford University at the given data and time.
        '''
        sun_info = sun.sun(self.jr_loc.observer, date=date, tzinfo=self.timezone)
        sunset = sun_info["sunset"]
        
        if round_to_minute:
            # Round to nearest minute:
            final_sunset = Utils.round_time(sunset, roundTo=60)
        else:
            final_sunset = sunset
        return final_sunset
    
    #------------------------------------
    # time_from_fname
    #-------------------
    
    def time_from_fname(self, fname_win_or_posix):
        '''
        Extract date/time from SonoBat classifier
        outputs:
        
           barn1_D20220205T192049m784-HiF.wav
           
        :param fname_win_or_posix: name to parse, wither Windows, 
            or posix convention.
        :type fname_win_or_posix: str
        :return: extracted data and time
        :rtype: datetime
        '''
        
        # Path library does not work with Windows
        # paths. So replace backslashes with Posix forward
        # slashes:
        fname = fname_win_or_posix.replace('\\', '/')
        path = Path(fname)
        name_no_ext = path.stem
        pat = re.compile(r'^[^_]*_.([\d]{8})T([\d]{6})m[\d]{3}')
        if (match := pat.match(name_no_ext)) is None:
            raise ValueError(f"File name {name_no_ext} does not have date/time")
        the_date, the_time = match.groups()
        yr = int(the_date[:4])
        mo = int(the_date[4:6])
        dy = int(the_date[6:])
        
        hr = int(the_time[:2])
        mi = int(the_time[2:4])
        sc = int(the_time[4:])
        
        # SonoBat recordings use Pacific Daylight Time (PDT)
        # throughout the year:
        dt = datetime(yr, mo, dy, hour=hr, minute=mi, second=sc, tzinfo=self.timezone)
        return dt
        
# --------------------- Class UniversalFd -----------

class UniversalFd:
    '''
    Instances behave like file descriptors. But depending
    on the output filename provided to the constructor,
    a UniversalFd can write to:
    
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
    important when writing to .feather files.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, out_fname):
        
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
    # __repr__
    #-------------------
    
    def __repr__(self):
        '''
        Representation will be like:
        
              <UniversalFd type=csv out.csv at 0xab4233>
        '''
        try:
            filetype = self.filetype
        except AttributeError:
            # Instance is not fully instantiated yet:
            filetype = 'pending'
        
        try:
            outname = self.out_name.name
        except AttributeError:
            # Instance is not fully instantiated yet:
            outname = 'pending'
           
        summary = f"<UniversalFd type={filetype} {outname} at {hex(id(self))}>"
        return summary

        
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
        
        # Must be a feather:
        df = pd.DataFrame(self.table)
        df.to_feather(self.out_name)
        
        
# --------------------- Class DataFrameRows -----------

class DataFrameRows:
    '''
    Iterates over the rows of a given dataframe
    almost the same as iterating over a .csv reader.
    This means that the first returned is the column 
    names. Also, rows are returned as lists, not pd.Series
    objects, as for csv readers. 
    
    The difference between a csv reader and this
    DataFrameRows reader is that for csv all numbers
    are delivered by strings, and consumers must turn
    them into numbers via int(), or float(). 
    
    This class could do that conversion to str to have
    complete equivalence. But consumers doing the conversion,
    because they might expect strings is not harmful:
    
           int('10') == 10 == int(10)

    '''
    
    def __init__(self, df):
        self.df = df
        self.it = df.iterrows()
        # Have not yet served the column names:
        self.served_colnames = False
    
    def __size__(self):
        # Length of iteration is number of
        # rows plus the column names: 
        return len(self.df) + 1
    
    def __iter__(self):
        return self
        
    def __next__(self):
        if not self.served_colnames:
            # Serve col names:
            self.served_colnames = True
            return list(self.df.columns)
        
        _idx, data = next(self.it)
        # Turn the data pd.Series into a list,
        # as would be true for csv readers:
        return list(data)
    
    
    
            
    