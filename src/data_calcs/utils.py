'''
Created on Apr 22, 2024

@author: paepcke
'''

from data_calcs.daytime_file_selection import (
    DaytimeFileSelector)
from datetime import (
    timedelta,
    datetime)
from enum import (
    Enum,
    auto)
from pathlib import (
    Path)
import json
import numpy as np
import os
import pandas as pd
import re

# ----------------------------- Class TimeGranularity ---------    

# Granularity of reapeating time units:
# 60 seconds in each minute, 24 hours in each day,
# 12 months in a year, etc. The numbers are 
# the max values of the respective granularity.
class TimeGranularity(Enum):
    SECONDS = auto()
    MINUTES = auto()
    HOURS   = auto()
    DAYS    = auto()
    MONTHS  = auto()
    YEARS   = auto()  # Decade; max: 10

    @staticmethod    
    def max_value(member):
        if member in (TimeGranularity.SECONDS, TimeGranularity.MINUTES):
            return 60
        if member == TimeGranularity.HOURS:
            return 24
        if member == TimeGranularity.DAYS:
            return 30
        if member == TimeGranularity.MONTHS:
            return 12
        if member == TimeGranularity.YEARS:
            # Decades
            return 10

# ----------------------------- Class Utils ---------    

class Utils:
    '''
    classdocs
    '''

    rec_time_util = DaytimeFileSelector()

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
  
    #------------------------------------
    # max_df_coords
    #-------------------
    
    @staticmethod
    def max_df_coords(df):
        '''
        Returns the row name and column name of
        the given dataframe that locate the largest
        element of the df.
        
        All elements must be comparable via ">"
        
        WARNING: The approach is brute force. Use only for small
        dfs. df.idxmax(axis=n) is likely significantly faster,
        but more confusing than this nested loop.
        
        :param df: dataframe whose largest element is to be found
        :type df: pd.DataFrame
        :return the name of the row index element and the name of
            the column index element that address the largest
            element
        :rtype: tuple[any, any]
        
        '''

        max_el = 0
        best_row = None
        best_col = None
        for row in df.index:
            for col in df.columns:
                new_el = df.loc[row].loc[col]
                if new_el > max_el:
                    max_el = new_el
                    best_row = row
                    best_col = col
        return (best_row, best_col)
        
    #------------------------------------
    # is_file_like
    #-------------------
    
    @staticmethod
    def is_file_like(value):
        '''
        Return True if value is a file-like, such 
        as a file descriptor or io.StringIO. Works
        by checking whether value has an attribute
        'write', which is a callable.
        
        :param value: value to examine
        :type value: any
        :return True if value is file-like
        :rtype bool
        '''
        try:
            return callable(value.write)
        except AttributeError:
            return False

    #------------------------------------
    # file_timestamp
    #-------------------
    
    @staticmethod
    def file_timestamp(time=None):
        '''
        Returns date and time in a format safe for use 
        in filenames. See safe_time_from_fname() to get
        that timestamp from a file name whose timestamp
        was created by this method.
        
        :param time: optinally, a datetime to turn into a 
            filename safe string. If None, current date and time
            are used.
        :type time: union[None | datetime.datetime]
        :return timestamp for use in file names
        :rtype str
        '''

        if time is not None and not isinstance(time, datetime):
            raise TypeError(f"Time argument must of None or a datetime.datetime")
                            
        if time is None:
            # Get like '2024-05-19T10:16:42.785678'
            dt_str = datetime.now().isoformat()
        else:
            dt_str = time.isoformat()
            
        # Remove the msecs part:
        # Replace colons with underscores:
        timestamp = dt_str
        timestamp = re.sub(r'[.][0-9]{6}', '', timestamp)
        timestamp = timestamp.replace(':', '_')
        return timestamp

    #------------------------------------
    # extract_file_timestamp
    #-------------------
    
    @staticmethod
    def extract_file_timestamp(fname):
        '''
        Given a string---usually a filename---,
        try to find an embedded timestamp formatted
        the way the file_timestamp() method outputs.
        
        Return the timestamp string if found, else None.
        
        NOTE: this method works with timestamps created
        by file_timestamp(), not with the timestamps 
        used in SonoBat .wav files. For those, use time_from_fname()
        
        :param fname: name to search
        :type fname: str
        :return the timestamp if found, else None
        :rtype {None | str}
        '''
        if isinstance(fname, Path):
            fname = str(fname)
        # Regex to find substrings created by
        # the file_timestamp() method embedded
        # in a string:
        pat = re.compile(r'.*([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}_[0-9]{2}_[0-9]{2}).*')
        
        the_match = pat.match(fname)
        if the_match is None:
            return None
        
        # Return the extracted timestamp:
        return the_match[1]
                    
    #------------------------------------
    # time_from_fname
    #-------------------
    
    @classmethod
    def time_from_fname(cls, fname):
        '''
        Given a SonoBat recording .wav filename, 
        extract, and return the recording time.
        Note: this method is not for timestamps of
        analysis results. For those, use extract_file_timestamp()
        
        :param fname: .wav file name from which to extract recording time
        :type fname: str
        :return recording date and time
        :rtype: datetime.datetime
        :raise ValueError if given filename does not encode date and time
        '''
        dt = cls.rec_time_util.time_from_fname(fname)
        return dt

    #------------------------------------
    # mk_fpath_from_other
    #-------------------
    
    @staticmethod
    def mk_fpath_from_other(other_fname, 
                            prefix='',
                            suffix='',
                            **name_components):
        '''
        Given a timestamped file name, like 'myfile_2024-02-28T20_15_20_truly.csv',
        create a new file name with the same timestamp. This is useful when
        several related computations are performed, and the result files are to
        have the same timestamp as the first calculation.
        
        Detail:
        Given a (1) filename with or without a timestamp, (2) a desired
        prefix and (3) suffix (fname extension incl. leading period),
        create a new filename. 
        
        The name_components is an optional dict containing as keys 
        filename fragments, whose values are the number of those 
        items are to be named in the filename.
        
        Example:
            other_fname : 'myfile_2024-02-28T20_15_20_truly.csv'
            prefix      : 'newfile_'
            suffix      : '.txt'
            
        Returns 'newfile_2024-02-28T20_15_20.txt'
        
        Adding name_components to the above args:
        
            name_components : samples=12, runs=10000
            
        returns: 'newfile_2024-02-28T20_15_20_12samples_10000runs.txt'
        
        
        :param other_fname: file name with timestamp
        :type other_fname: union[str, Path]
        :param prefix: text to place at start of new file name
        :type prefix: union[None, str]
        :param suffix: the file extension, including the period
        :type suffix: union[None, str]
        :returns the newly constructed filename, which shares
            other_fname's timestamp.
        :rtype str
        '''
        
        if not isinstance(other_fname, Path):
            other_fpath = Path(other_fname)
        else:
            other_fpath = other_fname
            
        timestamp = Utils.extract_file_timestamp(other_fname)

        fname = f"{prefix}{timestamp}"
        for name, quantity in name_components.items():
            fname += f"_{quantity}{name}"
        full_fpath = other_fpath.parent.joinpath(f"{fname}{suffix}")
        return str(full_fpath)

    #------------------------------------
    # find_file_by_timestamp
    #-------------------
    
    @staticmethod
    def find_file_by_timestamp(search_dir, 
                               timestamp=None, 
                               prefix=None, 
                               suffix=None, 
                               latest=True):
        '''
        Given a directory, search for all files with the given
        timestamp in its name. From among those, find the ones that have 
        the given prefix, and from that result the ones with a given suffix.
        If any of the those three filters is None, all files from search_dir 
        with timestamps in their name will be considered for the final step.  
        
        If multiple files pass the above filters, then the 'latest' argument
        decides final action. If True, the file with the latest timestamp in 
        its name is returned. Else all remaining candidates are returned in 
        a list.
        
        :param search_dir: directory to search
        :type search_dir: union[str, Path]
        :param timestamp: timestamp in filename, with the format produced by
            Utils.file_timestamp().
        :type timestamp: optional(str)
        :param prefix: text before the timestamp in the filename
        :type prefix: optional[str]
        :param suffix: file extension, including the period
        :type suffix: optional[str]
        :return: the absolute file name in a singleton list, or a longer list of 
            of filenames, if multiple files qualify. Empty list
            if no files qualify.
        :rtype union[list[str]]
        '''
        fnames = os.listdir(search_dir)
        if len(fnames) == 0:
            return []
        
        matching_fnames = []
        for fname in fnames:
            if timestamp is None or (found_timestamp := Utils.extract_file_timestamp(fname)) == timestamp:
                if prefix is None or fname.startswith(prefix):
                    if suffix is None or fname.endswith(suffix):
                        matching_fnames.append(os.path.join(search_dir, fname))
                        
        if not latest or len(matching_fnames) == 1:
            # Whether or not multiple files qualify, return the list
            return matching_fnames
        
        # Among the matches, find the one with the latest timestamp
        # in its name:

        cur_latest_dt    = None
        cur_latest_fname = None
        for fname in matching_fnames:
            timestamp = Utils.extract_file_timestamp(fname)
            dt = Utils.datetime_from_timestamp(timestamp)
            if cur_latest_dt is None or dt > cur_latest_dt:
                cur_latest_dt = dt 
                cur_latest_fname = fname
        return [] if cur_latest_fname is None else [cur_latest_fname]
    
    #------------------------------------
    # datetime_from_timestamp
    #-------------------
    
    @staticmethod
    def datetime_from_timestamp(str_timestamp):
        '''
        Given a timestamp that was created by time_for_fname() for use
        in file names, return a datetime.datetime object.
        
        The main task is to turn underscores back into colons.
        
        See also timestamp_from_datetime for the inverse.
        
        :param str_timestamp: timestamp to convert
        :type str_timestamp: str
        :return an equivalent datetime object
        :rtype datetime.datetime
        '''
    
        dt = datetime.fromisoformat(str_timestamp.replace('_', ':'))
        return dt
                
    #------------------------------------
    # timestamp_from_datetime
    #-------------------
    
    @staticmethod
    def timestamp_from_datetime(dt_timestamp):
        '''
        Given a datetime object, convert it to a date and time
        string that works in file names. I.e. into the format
        produced by time_for_fname()
        
        The main task is to turn colons into colons.
        
        See also datetime_from_timestamp for inverse
        
        :param dt_timestamp: timestamp datetime obj to convert
        :type dt_timestamp: datetime.datetime
        :return an equivalent timestamp string
        :rtype str
        '''
    
        str_timestamp = dt_timestamp.strftime('%Y-%m-%dT%H_%M_%S')
        return str_timestamp

    #------------------------------------
    # is_daytime_recording
    #-------------------
    
    @classmethod
    def is_daytime_recording(cls, time_src):
        '''
        Returns True or False, depending on whether the
        time_src encodes a recording time that was daylight
        at Jasper Ridge on the Stanford campus. If time_src
        is a file name, it is assumed to be a .wav file name
        that includes the datetime info. That filename is then
        decoded from the fname string.
        
        :param time_src: .wav name from which to extract recording date and time,
            or a Python datetime or Pandas Timestamp
        :type time_src: union[str | datetime.datetime | pd.Timestamp
        :return whether or not the recording time was during the day, or not.
        :rtype bool
        '''
        res = cls.rec_time_util.is_daytime_recording(time_src)
        return res

    #------------------------------------
    # cycle_time
    #-------------------
    
    @staticmethod
    def cycle_time(date_time, time_granularities=None):
        '''
        Given a datetime instance, and a time granularity,
        such as month, hour, day, etc., return either the
        sin and cos for that datetime and granularity, or 
        a dict mapping each time granularity to its respective
        sin/cos pair. The dict version is returned if time_granularity
        is passed in as None.
        
            {TimeGranularity.MINUTES : (<sin-float>, cos-float>),
             TimeGranularity.DAYS    : (<sin-float>, cos-float>),
                           ...
             }
        
        For convenience, the date_time may be an integer, which 
        indicates the number of time ticks. For instance, if the 
        caller wants the sin/cos mappings of the 7th day of any year
        or month, they can pass the integer 7. The time_granularities
        then disambiguates, and must not be None.
        
        NOTE: this method may be called many times. So no error checking
              is performed. Caveat caller.
              
        NOTE: the difference between this method and method cyclical_time_encoding()
              is that this method only transforms a single value, and tries
              to do that quickly. The cyclical_time_encoding() method operates
              on an entire pd.Series, and adds zero registration. 
         
        :param date_time: a datetime object for which sin/cos
            are to be found, or an integer datetime component
        :type date_time: union[int | datetime.datetime]
        :param time_granularities: for which time granularity
            cycle the sin/cos should be computed.
        :type time_granularities: TimeGranularity
        :return either a tuple with sin and cos for the given
            time granularity, or a dict with sin and cos tuples
            for all granularities in the TimeGranularity enum
        :rtype union[tuple[float, float], dict[TimeGranularity, tuple[float, float]]
        '''
        
        if isinstance(time_granularities, TimeGranularity):
            time_granularities = [time_granularities]
        elif time_granularities is None:
            # Make a list of all TimeGranularity members:
            time_granularities = list(iter(TimeGranularity))

        res_dict = {}            
        for time_gran in time_granularities:
            
            max_val = TimeGranularity.max_value(time_gran)

            # If datetime is just one of the integer time 
            # components, like number of seconds, or day of
            # the month, we have what we need for the calcs:
            if type(date_time) == int:
                tick = date_time
            else:
                # Got a datetime: extract the appropriate component:
                if time_gran == TimeGranularity.SECONDS:
                    tick = date_time.second
                elif time_gran == TimeGranularity.MINUTES:
                    tick = date_time.minute
                elif time_gran == TimeGranularity.HOURS:
                    tick = date_time.hour
                elif time_gran == TimeGranularity.DAYS:
                    tick = date_time.day
                elif time_gran == TimeGranularity.MONTHS:
                    tick = date_time.month
                elif time_gran == TimeGranularity.YEARS:
                    # Fractional decades: just use the 
                    # last digit of the year 0-9:
                    tick = date_time.year % 10
                
            sin_val = np.sin(2 * np.pi * tick/max_val)
            cos_val = np.cos(2 * np.pi * tick/max_val)
            sin_cos_tuple = (sin_val, cos_val)
            res_dict[time_gran] = sin_cos_tuple

        if len(res_dict) == 1:
            return sin_cos_tuple
        else:
            return res_dict
            
        
    #------------------------------------
    # cyclical_time_encoding
    #-------------------
    
    @staticmethod
    def cyclical_time_encoding(datetime_series, time_granularity, zero_registration=0):
        '''
        Encodes time into points on a circle. Useful in machine
        learning for discovering cyclical phenomena. 
        
        Takes a Pandas Series of datetime values, and a TimeGranularity
        object. The latter specifies whether to encode seconds, minutes, 
        hours, or months. Returns a DataFrame of two columns: 
                
                 <time_gran>_sin     <time_gran>_cos
                 
        The zero_registration indicates which time is to be considered 
        the 0-radian position. For example, if time_granularity is 
        TimeGranularity.HOURS, and zero_registration is 17, then any
        datetime dt with dt.hour == 17 is considered to be angle 0,
        i.e. in the 3-o'clock position. This feature could be used to
        have each cycle start at sundown.
        
        NOTE: the difference between this method and method cycle_time()
              is that this method operates on an entire pd.Series, and 
              adds zero registration, while cycle_time only transforms a 
              single value, and tries to do that quickly.  
          
                 
        :param datetime_series: datetime values from which to extract
            time for encoding
        :type datetime_series: pd.Series[datetime.datetime]
        :param time_granularity: whether to encode seconds, minutes, etc.
        :type time_granularity: TimeGranularity
        :return a two-column dataframe: the sin and cos encoding of
            the times, respectively
        '''
        if type(datetime_series) == pd.DatetimeIndex:
            datetime_series = pd.Series(datetime_series)
            
        try:
            max_val = TimeGranularity.max_value(time_granularity)
        except AttributeError:
            raise TypeError(f"Time granularity is not of type TimeGranularity, but {time_granularity}")
        
        # Create one cycle of the given time granularity:
        sin_cycle = [np.sin(2 * np.pi * tick/max_val)
                     for tick
                     in range(max_val)]
        # Round last entry to 0.0:
        sin_cycle[-1] = 0.0
        
        cos_cycle = [np.cos(2 * np.pi * tick/max_val)
                     for tick
                     in range(max_val)]
        # Round last entry to 1.0:
        cos_cycle[-1] = 1.0

        # Registration: 
        try:
            if time_granularity == TimeGranularity.HOURS:
                zero_idx = datetime_series.loc[datetime_series.dt.hour == zero_registration].index[0]
            elif time_granularity == TimeGranularity.MINUTES:
                zero_idx = datetime_series.loc[datetime_series.dt.minute == zero_registration].index[0]
            elif time_granularity == TimeGranularity.SECONDS:
                zero_idx = datetime_series.loc[datetime_series.dt.second == zero_registration].index[0]
            elif time_granularity == TimeGranularity.DAYS:
                zero_idx = datetime_series.loc[datetime_series.dt.day == zero_registration].index[0]
            elif time_granularity == TimeGranularity.MONTHS:
                zero_idx = datetime_series.loc[datetime_series.dt.month == zero_registration].index[0]
            else:
                raise NotImplementedError(f"Time granularity {time_granularity} is not implemented")
        except AttributeError:
            raise TypeError(f"Given Pandas series does not hold datetime-like objects; example {datetime_series[0]}")
        except IndexError:
            raise ValueError(f"Data has no date when time (at granularity {time_granularity} is {zero_registration}")

        col_nm_suffix = time_granularity.name.lower()
        sin_col_nm  = f"{col_nm_suffix}_sin"
        cos_col_nm  = f"{col_nm_suffix}_cos"
        
        # How many cycles fit between zero registration
        # and end of data?
        data_len  = len(datetime_series)
        cycle_len = len(sin_cycle)
        
        # Create a two-column df for sin, and cos sequences,
        # respectively. They are filled with NaN initially. We
        # want as many columns as are in the given datetime sequence:
        df = pd.DataFrame(index=datetime_series.index, 
                          columns=[sin_col_nm, cos_col_nm])
        
        num_full_cycles, partial = divmod(zero_idx, cycle_len)
        # Guard against 

        # Fill df from top to registration point, such
        # that the cycle starts again at zero registration point:
        df.loc[partial:zero_idx-1, sin_col_nm] = sin_cycle[:partial] + num_full_cycles * sin_cycle
        df.loc[partial:zero_idx-1, cos_col_nm] = cos_cycle[:partial] + num_full_cycles * cos_cycle

        # Fill result df from registration point down:
        num_full_cycles, partial = divmod(data_len-zero_idx, cycle_len)
        df.loc[zero_idx:, sin_col_nm] = num_full_cycles * sin_cycle + sin_cycle[:partial]
        df.loc[zero_idx:, cos_col_nm] = num_full_cycles * cos_cycle + cos_cycle[:partial]
        
        # Turn the sin/cos values to floats:
        df[sin_col_nm] = df[sin_col_nm].astype(float)
        df[cos_col_nm] = df[cos_col_nm].astype(float)
        
        return df 

# ------------------------------------- Class PDJson ----------

#******* NOT WORKING YET
class PDJson(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Series):
            jdict = {'__pd.series__' : {'data' : obj.to_json(), 'name' : obj.name}}
            jstr  = json.dumps(jdict) 
            return jstr
        elif isinstance(obj, pd.DataFrame):
            jstr = f'{{"__pd.dataframe__" : {{"data" : {obj.to_json()}}}}}'
            return jstr
        
        return super().default(obj)

    @staticmethod
    def decode(obj):
        # Check for key at any level using recursion
        def decode_object(value):
            
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    value[key] = decode_object(nested_value)
                    
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    value[i] = decode_object(item)
    
            # Check for your type marker:
            if isinstance(value, str):
                if value.startswith('{"__pd.series__"'):
                    # Get {'__pd.series__' : {'data' : <series data>}, 'name' : <ser name>}}
                    ser_dict = json.loads(value)
                    ser_data = ser_dict['__pd.series__']['data']
                    ser_name = ser_dict['__pd.series__']['name']
                    ser = pd.Series(ser_data, name=ser_name)
                    return ser
                elif value.startswith('{"__pd.dataframe__"'):
                    df_dict = json.loads(value)
                    df_data = df_dict['__pd.dataframe__']['data']
                    df = pd.DataFrame(df_data)
                    return df
                    
            return value
        return decode_object(obj)
        
