'''
Created on Apr 27, 2024

@author: paepcke
'''

from pathlib import Path
from pyarrow import feather
import os
import pandas as pd
import re

class DataPrep:
    '''
    Convert or extract data from transformer chirp 
    prediction outputs, and SonoBat classification 
    output data. Uses of the resulting outputs are
    clustering in various dimensions, and transformer
    output variance analysis
    '''

    sorted_mnames = [
        'LdgToFcAmp','HiFtoUpprKnAmp','HiFtoKnAmp','HiFtoFcAmp','UpprKnToKnAmp','KnToFcAmp',
        'Amp4thQrtl','Amp2ndQrtl','Amp3rdQrtl','PrecedingIntrvl','PrcntKneeDur','Amp1stQrtl',
        'PrcntMaxAmpDur','AmpK@start','FFwd32dB','Bndw32dB','StartF','HiFreq','UpprKnFreq',
        'FFwd20dB','FreqKnee','FFwd15dB','Bndwdth','FFwd5dB','FreqMaxPwr','FreqCtr','FBak5dB',
        'FreqLedge','AmpK@end','Fc','FBak15dB','FBak32dB','EndF','FBak20dB','LowFreq',
        'Bndw20dB','CallsPerSec','EndSlope','SteepestSlope','StartSlope','Bndw15dB',
        'HiFtoUpprKnSlp','HiFtoKnSlope','DominantSlope','Bndw5dB','PreFc500','PreFc1000',
        'PreFc3000','KneeToFcSlope','TotalSlope','PreFc250','CallDuration','CummNmlzdSlp',
        'DurOf32dB','SlopeAtFc','LdgToFcSlp','DurOf20dB','DurOf15dB','TimeFromMaxToFc','KnToFcDur',
        'HiFtoFcExpAmp','AmpKurtosis','LowestSlope','KnToFcDmp','HiFtoKnExpAmp','DurOf5dB','KnToFcExpAmp',
        'RelPwr3rdTo1st','LnExpBStartAmp','Filter','HiFtoKnDmp','LnExpBEndAmp','HiFtoFcDmp','AmpSkew',
        'LedgeDuration','KneeToFcResidue','PreFc3000Residue','AmpGausR2','PreFc1000Residue','Amp1stMean',
        'LdgToFcExp','FcMinusEndF','Amp4thMean','HiFtoUpprKnExp','HiFtoKnExp','KnToFcExp','UpprKnToKnExp',
        'Kn-FcCurviness','Quality','Amp2ndMean','HiFtoFcExp','LnExpAEndAmp','RelPwr2ndTo1st','LnExpAStartAmp',
        'HiFminusStartF','Amp3rdMean','PreFc500Residue','Kn-FcCurvinessTrndSlp','PreFc250Residue',
        'AmpVariance','AmpMoment','meanKn-FcCurviness','Preemphasis','MinAccpQuality','Max#CallsConsidered',
        'MaxSegLnght','AmpStartLn60ExpC','AmpEndLn60ExpC'        
        ]

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, measures_root, inference_root, fid_col='file_id'):
        '''
        Constructor
        '''
        self.measures_root = measures_root
        self.inference_root = inference_root
        self.fid_col = fid_col
        
        # Prepare a list of paths to all split files:
        split_fname_pat = re.compile(r'^split[\d]*$')

        # Filter split files from other files in the measures_root:        
        def split_name_detector(fname):
            fname_stem = Path(fname).stem
            return split_fname_pat.match(fname_stem) is not None
        
        # Get just the split file names from the measures dir
        split_fnames = filter(split_name_detector, os.listdir(measures_root))
        
        # Get dict mapping a split file ID (a running int) to 
        # the full pathname of a split file:
        self.split_fpaths = {i : os.path.join(self.measures_root, split_fname)
                             for i, split_fname
                             in enumerate(split_fnames)}
        
        # Create self.fid2split_dict for mapping a file id
        # to the split file that contains the data for that
        # file id:
        
        self.make_fid2split_dict()
        
        # Cache of dfs of measures split files we had 
        # to open so far: maps split id to df:
        self.split_file_dfs_cache = {}
        
    #------------------------------------
    # make_fid2split_dict
    #-------------------
    
    def make_fid2split_dict(self):
        '''
        Create a dict mapping measures file identifier ints
        to the measures split file that contains the measure
        created from the file identified by the file id. Like
        
            10 : '/foo/bar/split40.feather',
            43 : '/foo/bar/split4.feather',
                      ...
                      
        This dict is used, for example, to retrieve the chirp measures 
        that correspond to a given T-sne point.
        
        The result will be in self.fid2split_dict.
        '''
        
        self.fid2split_dict = {}
        for fpath in self.split_fpaths.values():
            fids_df = pd.read_feather(fpath, columns=[self.fid_col])
            # Get list of file ids in this split file as a list:
            fids = fids_df.file_id.values
            
            # Add all this split file's file ids to the
            # fid2split_dict:
            self.fid2split_dict.update({fid : fpath for fid in fids})

    #------------------------------------
    # measures_from_fid
    #-------------------
    
    def measures_from_fid(self, fid):
        '''
        Given a measures file id, return a pandas
        Series with the measures.
        
        :param fid: file id that identifies the row
            in a dataset where the related chirp measures
            are stored.
        :type fid: int
        :return the chirp measures created by SonoBat
        :rtype pd.Series
        '''
        
        try:
            df = self.split_file_dfs_cache[fid]
            measures = df.loc[fid]
            return measures
        except KeyError:
            # Df not available yet:
            pass
        
        split_path = self.fid2split_dict[fid]
        df = pd.read_feather(split_path)
        
        # Copy the file id column to the index
        df.index = df[self.fid_col]
        
        measures = df.loc[fid]
        return measures   
    
    #------------------------------------
    # sort_by_variance
    #-------------------
    
    @classmethod
    def sort_by_variance(cls, measures):
        '''
        Given a list of measure names, return a new list
        of the same names, sorted by decreasing variance.
        
        :param measures: list of measure names to sort
        :type measures: (str)
        :return names sorted by decreasing variance
        :rtype (str)
        '''
        # For informative error msg if name
        # is passed in that is not a SonoBat measure:
        global curr_el
        try:
            def key_func(el):
                # Initialize curr_el to be the 
                # element currently being sorted:
                global curr_el
                curr_el = el
                return cls.sorted_mnames.index(el)
            
            new_list = sorted(measures, key=key_func)
        except ValueError:
            raise ValueError(f"Measure named '{curr_el}' is not a known SonoBat measure")
        return new_list
    
    #------------------------------------
    # measures_by_var_rank
    #-------------------
    
    @classmethod
    def measures_by_var_rank(cls, src_info, min_var_rank):
        '''
        Given either a dataframe, or the path to a .csv/.feather file,
        return a new dataframe that is a subset of the given df, and
        contains data only for the measure names of sufficient variance.
        The min_var_rank specifies the variance threshold. Only measures
        with a variance rank greater than min_var_rank are included in the
        returned df.
        
        Example:
        
        src_info:

               'HiFtoKnAmp'  'LdgToFcAmp'   'HiFtoUpprKnAmp'
            0     10              20            30 
            1     100            200           300
            
        min_var_rank = 2
        
        Returns:
               'LdgToFcAmp','HiFtoUpprKnAmp'
            0      20            30
            1     200           300
            
        Because the three measures have variance rank order
        
           'LdgToFcAmp','HiFtoUpprKnAmp','HiFtoKnAmp'
           
        so the first column is not retained.  
          
        :param src_info: either a dataframe, or the path to a .csv or .feather file
        :type src_info: union[pd.DataFrame | str]
        :return extracted dataframe
        :rtype pd.DataFrame
        '''
        
        if type(src_info) == str:
            df_path = Path(src_info)
            if df_path.suffix == '.feather':
                df = pd.read_feather(df_path)
            elif df_path.suffix == '.csv':
                df = pd.read_csv(df_path)
            else:
                raise TypeError(f"Only .csv and .feather files are supported, not '{df_path.suffix}'")
        elif type(src_info) == pd.DataFrame:
            df = src_info
        else:
            raise TypeError(f"Dataframe source must be a path or a df, not {src_info}")
            
        given_cols = df.columns 
        cols_by_variance = cls.sort_by_variance(given_cols)
        try:
            cols_wanted = cols_by_variance[:min_var_rank]
        except IndexError:
            # Desired minimum rank is larger than the 
            # number of SonoBat measures, so use the whole
            # source df:
            cols_wanted = cols_by_variance
            
        new_df = df[cols_wanted]
        return new_df
        
        
    