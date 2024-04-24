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
        