'''
Created on Apr 22, 2024

@author: paepcke
'''

from _datetime import timedelta, datetime

class Utils:
    '''
    classdocs
    '''

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
