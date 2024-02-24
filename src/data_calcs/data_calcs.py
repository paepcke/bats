#!/usr/bin/env python
'''
Created on Feb 20, 2024

@author: paepcke
'''
import argparse
import os
import pandas as pd  # pip install pyarrow
import re
import sys
from _ast import Or

class DataCalcs:
    '''
    Imports Sonobat data, and provides several 
    services:
    
    1. Method clean_data() 
        o Removes columns that are not physical
          measures, file path, and version columns
        o When measures are strings, such as 
             MaxSegLnght: 0.5 sec
          the values are turned into floats, such as
          0.5 in this case.
        o The categorical value 'medium'...:
             Preemphasis: 'medium'
          ... is turned into an int in range [0..2],
          for 'low', 'medium', 'high'
     
    2. Method compute_stats() creates a new
       .csv file with the same column names as the
       input, but with a single row that contains
       the variance of the respective data column.
    '''
    FilterXlation = {
        '5 kHz'               : 0,
        '30 kHz anti-katydid' : 1,
        '10 kHz cutoff'       : 2,
        '20 kHz anti-katydid' : 3,
        '25 kHz ant-katydid'  : 4,
        '15 kHz cutoff'       : 5   
       }
    

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 infile, 
                 outfile=None,       # Use False for no output
                 compute_stats=False,# True, or a filepath for saving stats
                 cull_threshold=None,
                 remove_nans=True,
                 admin_cols=None,    # Give [] if no administrative cols
                 unittesting=False
                 ):
        '''
        Load the df from infile. Remove purely administrative
        columns. If admin_cols kwarg is None, the SonoBats 
        administrative columns are used. If admin_cols is
        an empty array, it is assumed that no admin cols are
        present.
        
        As many columns as possible are transformed into 
        numeric values. E.g.: '0.5 sec' is turned into float 0.5.
        
        Columns with only NaNs are removed.
        
        If:
            o an outfile path is provided, the modified
              dataset is written as raw data .csv to outfile
            o outfile is None, a summary of the df is written
              to stdout
            o outfile is False, no output
        
        For all values of output, the result df is available
        via <cleaner>.df 
            
        If:
            o compute_stats is True, a df of statistics over 
              the column value stats is created
            o compute_stats is a file path, the statistics df
              is written as a dataframe .csv file to
              
        For all values other than False, the stats are available
        via <cleaner>.stats_df 
              
        If unittesting is True, none of the above operations
        are performed in this constructor.   
        '''
        
        # No statistics available yet:
        self.stats_df = None
        
        # Standard SonoBat administrative columns:
        if admin_cols is None:
            self.admin_cols = ['Filename',
                               'ParentDir',
                               'NextDirUp',
                               'TimeInFile', 
                               'Path',
                               'Version',
                               ]
        else:
            # User-provided admin columns:
            self.admin_cols = admin_cols
            
        if unittesting:
            return
        
        # Load the dataframe, excluding the above
        # non-numeric, administrative colums, and
        # removing rows with all columns === NaNs:
        df = self.load_sonobat_data(infile, 
                                    exclude_cols=self.admin_cols,
                                    remove_nans=remove_nans)

        # Turn values like '0.5 sec', which are partly string
        # typed into numbers
        self.df = self.make_numeric(df)
        
        if compute_stats:
            # The resulting stats dataframe
            # won't include the variance of the 
            # TimeIndex column:
            self.stats_df = self.compute_stats(df)
            
        if cull_threshold is not None:
            if (type(cull_threshold) not in (float, int)) or \
                not (0.0 <= cull_threshold <= 1.0): 
                raise ValueError(f"Cull threshold must be a number between 0.0 and 1.0, not {cull_threshold}")
            
            self.df = self.cull_columns(self.df, cull_threshold)

        # Output results or not: outfile may be None, False, or a path:
        if outfile is None:
            # Print summary to console:
            print(self.df)
            
        elif os.path.exists(outfile) and type(outfile) == str:
            # Ask user whether OK to overwrite:
            decision = input(f"Output file {outfile} exists; overwrite? (y/n)")            
            if decision == 'y':
                try:
                    self.df.to_csv(outfile, index=False)
                except Exception as e:
                    print(f"Could not write df to {outfile}: {e}") 
            else:
                print('No df written, but available via <cleaner>.df')
                
        elif type(outfile) == str:
            # Assume that the outfile string is a path:
            try:
                self.df.to_csv(outfile, index=False)
            except Exception as e:
                print(f"Could not write df to {outfile}: {e}")
        # Else, no output for self.df at all
        
        # Client asked to output stats to a file.
        
        if os.path.exists(compute_stats) and type(compute_stats) == str:
            # # Make file name <outfile>_stats.csv>
            # stats_path = (f"{compute_stats.parent}/"
            #               f"{compute_stats.stem}_stats"
            #               f"{compute_stats.suffix}")
            
            # Assume that compute_stats is a path. Check
            # whether exists:
            decision = input(f"File {compute_stats} exists; overwrite? (y/n)")
            if decision != 'y':
                try:
                    self.stats_df.to_csv(compute_stats, index=True)
                except Exception as e:
                    print(f"Could not save stats to {compute_stats}: {e}")
            else:
                print('No stats written, but available via <cleaner>.stats_df')
       
        elif compute_stats == True:
            # Just output to console:
            print(self.stats_df)
                
        elif type(compute_stats) == str:
            # Compute_stats is a path, which does not yet exist:
            try:
                self.stats_df.to_csv(compute_stats, index=True)
            except Exception as e:
                print(f"Could not save stats to {compute_stats}: {e}")
        
    #------------------------------------
    # load_sonobat_data
    #-------------------
    
    def load_sonobat_data(self, fpath, exclude_cols=[], remove_nans=False):
        '''
        Reads .csv file into a dataframe,
        and removes specified columns, if
        any
        
        :param fpath: location of .csv
        :type fpath: str
        :param exclude_cols: optionally: columns to remove from the
            result dataframe
        :type exclude_cols: list
        :param remove_nans: whether or not to remove rows
            where all columns are NaN values
        :type remove_nans: bool
        :return Dataframe
        :rtype pd.Dataframe
        '''
        
        # Read the .csv file as if it doesn't
        # contain dataframe index information:
        df = pd.read_csv(fpath, index_col=False)
        # Make df look the same, whether or not
        # the .csv file was a raw SonoBat file, or
        # a SonoBat file that we previously processed,
        # and saved via df.to_csv():
        df = self._normalize_frame(df)
        
        # Be nice, and handle user supplied administrative
        # columns are not present:
        user_cols = set(exclude_cols)
        true_cols = set(df.columns)
        exclude_cols = list(true_cols.intersection(user_cols))
         
        if type(exclude_cols) == list and len(exclude_cols) > 0:
            # Remove column(s) as per caller's request:
            df.drop(exclude_cols, axis='columns', inplace=True)

        # Copy column TimeIndex to serve as the 
        # dataframe index:      
        df.set_index('TimeIndex', drop=False, inplace=True)      
    
        if remove_nans:        
            # Remove rows in which all columns, except
            # the TimeIndex column are NaNs:
            cols_to_incl = df.columns.to_list()
            try:
                cols_to_incl.remove('TimeIndex')
            except ValueError:
                # TimeIndex is not one of the columns:
                pass
            df.dropna(axis='index', 
                      how='all',
                      subset=cols_to_incl, 
                      inplace=True)
            
        return df

    #------------------------------------
    # make_numeric
    #-------------------
    
    def make_numeric(self, df):
        '''
        Ensure that cell values Preemphasis,
        and MaxSegLnght, and Filter are numeric.
        
        :param df: dataframe to change
        :type df: pd.DataFrame
        :return: same df with columns modified in place.
        :rtype: pd.DataFrame 
        '''
        df = self._fix_preemph(df)
        df = self._fix_filter(df)        
        df = self._fix_max_seg_length(df)
        return df
        
    #------------------------------------
    # compute_stats
    #-------------------
    
    def compute_stats(self, df):
        '''
        Given a dataframe, compute the variance
        and other statistics separately for each
        column. 
        Return a dataframe with columns sorted by
        decreasing variance, like 
        
                   col3  col1  col2  col5   col4
        variance
        var_norm
        stdev
        max
        min
        mean
        median
        
        :param df: dataframe to analyze
        :type df: pd.DataFrame
        :return: a dataframe of column stats.
        :rtype: pd.DataFrame
        '''
        
        variances   = df.var(axis='index')
        vars_normed = variances.rank(pct=True) # normalize 0-1
        stdevs      = df.std(axis='index')
        maxies      = df.max(axis='index')
        minies      = df.min(axis='index')
        means       = df.mean(axis='index')
        medians     = df.median(axis='index')
        
        stats_df = pd.concat(
            [variances,vars_normed,stdevs,maxies,minies,means,medians],
            axis='columns',
            
            )
        
        stats_df.drop('TimeIndex', inplace=True)
        
        # Get:
        #        PrecedingIntrvl  LnExpB_StartAmp  Amp2ndMean  LnExpA_EndAmp
        #     0      1025.345088         1.751163    0.003516       0.003518
        #     1        32.021010         1.323315    0.059298       0.059315
        #     2       151.056000        -0.876543    0.855096       0.111705
        #     3        85.536000        -3.604043    0.726407      -0.027919
        #     4       103.068000        -1.964261    0.813731       0.028525
        #     5        87.840000        -1.688229    0.836711       0.015158
        
        stats_df = stats_df.transpose()
        stats_df.index = pd.Index(
            ['variance', 'vars_normed', 'stdev', 'max', 'min', 'mean', 'median'],
            name='Stats'
            )

        # Get just variance of each column:
        variances = stats_df.loc['variance']

        # Now have:
        #     PrecedingIntrvl    1025.345088
        #     LnExpB_StartAmp       1.751163
        #     Amp2ndMean            0.003516   <--- note order
        #     LnExpA_EndAmp         0.003518   <--- this one is larger
        #     Name: variance, dtype: float64
        
        # Sort the df by descending variance, getting like:


        # Sort variances to get:
        #      PrecedingIntrvl    1025.345088
        #      LnExpB_StartAmp       1.751163
        #      LnExpA_EndAmp         0.003518   <--- now by descending order
        #      Amp2ndMean            0.003516
        #      Name: variance, dtype: float64
        variances.sort_values(ascending=False, inplace=True)
        
        # Go back and sort columns of result df
        # to follow the index shown above:
        
        stats_df_sorted = stats_df.reindex(columns=variances.index)

        return stats_df_sorted

    #------------------------------------
    # cull_columns
    #-------------------
    
    def cull_columns(self, df, threshold):
        
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold must be between 0 and 1, not {threshold}")
        
        # Already have stats over cols?
        if self.stats_df is None:
            # Stats over column values have not
            # yet been computed. Compute them
            # now:
            self.stats_df = self.compute_stats(df)
        
        # Get the descending normalized variances:
        normed_vars = self.stats_df.loc['vars_normed', :]
        # Find the threshold point in the series: Get
        # series [<highes_var>, <second-highes-var> ... <last_acceptable_var>]
        variences_to_keep = normed_vars[normed_vars>=threshold]
        # Now chop off all columns below the threshold
        # in the given df:
        df_culled = df.iloc[:,0:len(variences_to_keep)]
        return(df_culled)

    #------------------------------------
    # _fix_preemph
    #-------------------
    
    def _fix_preemph(self, df):
        
        if 'Preemphasis' not in df.columns:
            return df
        
        # Change Preemphasis from 'low', 'medium', etc. to 0,1,2:
        def from_preemph(col_val):
            if col_val == 'low':
                return 0
            elif col_val == 'medium':
                return 1
            elif col_val == 'high':
                return 2
            else:
                raise ValueError(f"Preemphasis value {col_val} is not in ['low', 'medium', 'high']")
            
        new_preemph = df.loc[:,'Preemphasis'].apply(from_preemph)
        df.loc[:,'Preemphasis'] = new_preemph
        return df

    #------------------------------------
    # _fix_filter
    #-------------------
    
    def _fix_filter(self, df):

        if 'Filter' not in df.columns:
            return df
        
        # Convert Filter column into floats:
        def from_filter(col_val):
            try:
                numeric = self.FilterXlation[col_val]
            except KeyError:
                raise ValueError(f"Expected filter spec, got '{col_val}'")
            return numeric
        
        new_filter = df.loc[:,'Filter'].apply(from_filter)
        df.loc[:,'Filter'] = new_filter
        return df

    #------------------------------------
    # _fix_max_seg_length
    #-------------------

    def _fix_max_seg_length(self, df):

        if 'MaxSegLnght' not in df.columns:
            return df

        # Pattern for like '0.5 sec', but not
        # '0.5 secs' or 'sec', or '5sec':
        pat = re.compile(r'^[.\d]* sec$')
        def from_str(col_val):
            
            if pat.match(col_val) is None:
                raise ValueError(f"Expected '<float> sec', got '{col_val}' in col MaxSegLnght")
            just_num = col_val[:len(col_val)-len(' sec')]
            return float(just_num)
        
        new_seg_ln = df.loc[:,'MaxSegLnght'].apply(from_str)
        df.loc[:,'MaxSegLnght'] = new_seg_ln
        return df

    #------------------------------------
    # normalize_frame
    #-------------------

    @staticmethod
    def _normalize_frame(df):
        '''
        Used to modify df as needed, so that 
        a .csv file imported from raw, SonoBat-generated
        data, looks just like a df that we
        saved via df.to_csv() earlier.
        
        We detect the difference like this: If
        df was read (with pd.read_csv(<fname>, index_col=False))
        from a .csv file that is a previously saved df, it
        looks like:

          Was the df's index
               |
               V        
           TimeIndex  numbers    mixed strings                 enum  TimeIndex.1
        0          19        5  0.5 sec     car  30 kHz anti-katydid      19
        1          20        3  4.0 sec    bike        10 kHz cutoff      20
        2          21        1  4.0 sec    bike        10 kHz cutoff      21
        
        That is the csv_read created a range index,
        and since the TimeIndex column, which used to be the df's
        index is a regular column, the SonoBat TimeIndex
        column was renamed to TimeIndex.1 
		            
        If instead the df was read from a SonoBat
        raw export, the df will look like this:
        
               numbers    mixed strings                 enum  TimeIndex
            0        5  0.5 sec     car  30 kHz anti-katydid          19
            1        3  4.0 sec    bike        10 kHz cutoff          20
            2        1  4.0 sec    bike        10 kHz cutoff          21
        
        Thus, if loaded from raw, the TimeIndex column will be
        the original. In this case we copy the TimeIndex column
        as the new df's index.
        
        :param df: dataframe to normalize
        :type df: pd.DataFrame
        :return: dataframe with index mirroring the TimeIndex column
        :rtype: pd.DataFrame
        '''
        if 'TimeIndex.1' in df.columns:
            # Assume the .csv was from a df.to_csv().
            # We turn the TimeIndex column into the index,
            # removing the TimeIndex column itself:
            df.set_index('TimeIndex', drop=True, inplace=True)

            # Next, we rename the TimeIndex1 column
            # to TimeIndex:
            df.rename({'TimeIndex.1' : 'TimeIndex'}, 
                      inplace=True, 
                      axis='columns')
            return df
        
        # Df comes from raw SonoFile csv:
        # Replicate the TimeIndex column to the df's
        # index, leaving the TimeIndex column intact:
        df.set_index('TimeIndex', drop=False, inplace=True)
        return df

# ------------------------ Main ------------
if __name__ == '__main__':

    # *****************
    # fpath = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/data.csv'
    # df = DataCalcs(fpath)
    # variances = DataCalcs(fpath, compute_variance=True)
    # sys.exit(0)
    # *****************

    descr = ('Inputs a SonoBat data file. Accomplishes one or more tasks:\n'
             '\n'
             '    o Removes administrative columns\n'
             '    o Ensures that remaining columns have numeric values\n'
             '    o Computes stats over values in each col.\n'
             '    o Culls columns whose variance is below a given percentile\n'
             '\n'
             'Examples:\n'
             '\n'
             '  Print an excerpt of a cleaned input:\n'
             '    data_calcs my_input.csv\n'
             '\n'
             '  Save cleaned input to a .csv file:\n'
             '    data_calcs -o my_cleaned_file.csv my_input.csv\n'
             '\n'
             '  Print statistics for all columns to console:\n'
             '    data_calcs --statistics True my_input.csv\n'
             '\n'
             '  Save statistics to a file:\n'
             '    data_calcs --statistics my_stats.csv my_input.csv\n'
        )

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=descr
                                     )
    parser.add_argument('-o', '--outfile',
                        action='store',
                        default=None,
                        help='file for the finished df; default is to print the cleaned result',)

    parser.add_argument('-s', '--statistics',
                        help=('with this option, infile is assumed to be\n'
                              'a cleaned SonoBat file. If value True, print\n'
                              'stats to console. If a path, it will hold\n'
                              'the .csv with with stats for each column.'),
                        default=False
                        )
    parser.add_argument('-c', '--cull',
                        help=('with this switch, infile is assumed to be\n'
                              'a cleaned SonoBat file. The value must be a\n'
                              'number between 0.0 and 1.0, which indicates\n'
                              'variance threshold below which a column is removed'),
                        default=None
                        )
    
    # parser.add_argument('-d', '--dirty',
    #                     action='store_false',
    #                     help='with this switch, infile is NOT cleaned ahead of other operations.')

    # Required
    parser.add_argument('infile',
                        help='fully qualified Sonobat outfile file')


    args = parser.parse_args()

    #**********
    # print(f"Stats: {args.statistics}")
    # sys.exit()
    #**********


    if args.cull is not None:
        try:
            cull_thres = float(args.cull)
        except Exception:
            print(f"Cull threshold must be a number between 0.0 and 1.0, not {args.cull}")
            
        if not (0.0 <= cull_thres <= 1.0): 
            print(f"Cull threshold must be a number between 0.0 and 1.0, not {cull_thres}")
            sys.exit(1)

    if args.statistics:
        if args.statistics == 'True':
            do_stats = True
        else:
            do_stats = args.statistics
    else:
        do_stats = False

    if not os.path.exists(args.infile):
        print(f"SonoBat csv input file does not exist")
        sys.exit(1)
        
    if args.outfile and os.path.exists(args.outfile):
        decision = input(f"Output file {args.outfile} exists; overwrite (y/n)")
        if decision != 'y':
            print('Aborting')

    DataCalcs(args.infile, 
              args.outfile,
              compute_stats=do_stats
              )

    # '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/data.csv'        
