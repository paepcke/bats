#!/usr/bin/env python
'''
Created on Apr 20, 2024

@author: paepcke
'''
from astral import geocoder, sun
from data_calcs.universal_fd import UniversalFd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import csv
import gzip
import os
import pyarrow
import re
import sys

class DaytimeFileSelector:
    '''
    Given a SonoBat classification output file, and 
    the column name where SonoBat names the input .wav
    file, copy to a destination only measures from recordings 
    that occurred during daytime at the Jasper Ridge preserve.
    
    The class computes the  sunset and sunrise for each 
    recording day. SonoBat's .wav file names are stored
    in its classification measures output table. Those names
    contain their recording dates. Those dates are extracted,
    and checked to be between sunrise and sunset of the
    recording day at the Preserve.
    
    Given an output file, copies the rows of those classification
    measurement rows to that destination, if the calls logged
    in the row occurred during daytime.
    
    Also provided is a method is_daytime_recording(fname) that
    extract time and day from a file name, and returns True/False
    depending on whether the recording was during daylight hours.
    
    Example one-shot daytime determination:
    
        selector = DaytimeFileSelector()
        if selector.is_daytime_recording('/foo/bar/barn1_D20220205T192049m784-HiF.wav'):
            print('Daytime')
        else:
            print('Nighttime')
            
    Example copy rows from daytime recordings to a destination file:
    
        selector = DaytimeFileSelector()
        selector.daytime_recordings('/tmp/myrecordings.csv',
                                    '/tmp/only_daytime_recordings.csv, 
                                    'Filename')
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self):
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
    # is_daytime_recording
    #-------------------
    
    def is_daytime_recording(self, rec_time_src):
        '''
        Return True if the recording time that is provided
        in rec_time_src was daytime at Stanford's Jasper Ridge
        Preserve when the recording occurred.
        
        If rec_time_src is the name of a .wav recording, the 
        datetime of the recording is extracted from the filename.
        Else it is expected to be a datetime.datetime object.
        Filenames are expected to be of the form
        
             barn1_D20220205T192049m784-HiF.wav
          
        :param rec_time_src: file name from which to obtain recording
            date and time.
        :type rec_time_src: union[str | datetime.datetime]
        :return True if recording time is during daylight hours, else False
        :rtype bool
        :raise ValueError if given filename does not encode date and time
        '''
        # Extract a datetime from the fname if necessary:
        if type(rec_time_src) == str:
            # Source is a filename:
            recording_time = self.time_from_fname(rec_time_src)
        else:
            recording_time = rec_time_src

        sunrise, sunset = self.sunrise_sunset_times(
            recording_time, round_to_minute=True)
        if sunrise <= recording_time <= sunset:
            return True 
        return False

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
                        out_fd = UniversalFd(out_file, mode='w')
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
                    if self.is_daytime_recording(fname):
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
    # sunrise_sunset_times
    #-------------------
    
    def sunrise_sunset_times(self, date, round_to_minute=True):
        '''
        Given a date return the times of sunrise and sunset 
        at the Jasper Ridge Preserve. All computations are 
        in Pacific Daylight Time (PDT).
         
        :param date: date whose sunset is to be computed
        :type date: datetime
        :param round_to_minute: round to the nearest minute
        :type round_to_minute: bool
        :return the (possibly rounded) sunrise and sunset times
            at Jasper Ridge, Stanford University at the given data 
            and time.
        :rtype (datetime, datetime)
        '''
        sun_info = sun.sun(self.jr_loc.observer, date=date, tzinfo=self.timezone)
        sunset = sun_info["sunset"]
        
        if round_to_minute:
            # Round to nearest minute:
            final_sunset = self.round_time(sunset, roundTo=60)
        else:
            final_sunset = sunset
            
        sunrise = sun_info["sunrise"]
        
        if round_to_minute:
            # Round to nearest minute:
            final_sunrise = self.round_time(sunrise, roundTo=60)
        else:
            final_sunrise = sunrise
            
        return (final_sunrise, final_sunset)
    
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
        :raise ValueError if given filename does not encode date and time
        '''
        
        # Path library does not work with Windows
        # paths. So replace backslashes with Posix forward
        # slashes:
        fname = fname_win_or_posix.replace('\\', '/')
        path = Path(fname)
        name_no_ext = path.stem
        pat = re.compile(r'^[^_]*_.([\d]{8})T([\d]{6})m[\d]{3}')
        if (match := pat.match(name_no_ext)) is None:
            raise ValueError(f"File name {name_no_ext} does not encode date/time")
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
        
    #------------------------------------
    # round_time
    #-------------------

    @staticmethod
    def round_time(dt=None, roundTo=60):
        '''
        Round a datetime object to any time lapse in seconds
        Examples:
        
            roundTime(datetime.datetime(2012,12,31,23,44,59,1234),roundTo=60*60)
               => 2013-01-01 00:00:00
            
            roundTime(datetime.datetime(2012,12,31,23,44,59,1234),roundTo=30*60)
               => 2012-12-31 23:30:00
            
            roundTime(datetime(2012,12,31,23,44,29,1234),roundTo=60)
               => 2012-12-31 23:44:00
            
            print(roundTime(datetime(2012,12,31,23,44,31,1234),roundTo=60)
               => 2012-12-31 23:45:00
            
            roundTime(datetime(2012,12,31,23,44,30,1234),roundTo=60)
               => 2012-12-31 23:45:00

        Based on: Author: Thierry Husson 2012        

        :param dt: datetime object to round, default: now.
        :type dt: datetime.datetime
        :param roundTo: Closest number of seconds to round to, default 1 minute.
        :type roundTo: int
        :return a rounded datetime
        :rtype: datetime.datetime
        '''

        if dt == None : dt = datetime.now()
        seconds = (dt.replace(tzinfo=None) - dt.min).seconds
        rounding = (seconds+roundTo/2) // roundTo * roundTo
        return dt + timedelta(0,rounding-seconds,-dt.microsecond)
        
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

    