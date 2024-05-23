'''
Created on Apr 27, 2024

@author: paepcke
'''

from collections import namedtuple
from data_calcs.data_viz import DataViz
from data_calcs.universal_fd import UniversalFd
from data_calcs.utils import Utils, TimeGranularity
from datetime import datetime
from enum import Enum
from io import StringIO
from logging_service.logging_service import LoggingService
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tempfile import NamedTemporaryFile
from itertools import chain
import csv
import json
import numpy as np
import os
import pandas as pd
import random
import re
import shutil
import time


# ----------------------------- Enum Action ---------------

# Members of the Action enum are passed to main
# to specify which task the program is to perform: 
class Action(Enum):
    HYPER_SEARCH  = 0
    PLOT          = 1
    ORGANIZE      = 2
    CLEAR_RESULTS = 3
    SAMPLE_CHIRPS = 4

# ---------------------------------- namedtuple ChirpID ---------------

ChirpIdSrc = namedtuple('ChirpIDSrc', 
                        ['wav_file_nm_col', 'more_key_cols'],
                        defaults=['file_id', ['chirp_idx']]
                        )
'''
Holds which columns in a SonoBat measures (split) file 
together uniquely identify a single chirp.

At least the name of the column that holds the .wav file's name
must be provided in the wav_file_nm field, since the file name 
contains the recording date and time.

If other columns are needed to uniquely identify a chirp, they are
to be in a list in the more_key_cols field. 
'''

# ---------------------------------- Class PerplexitySearchResult ---------------

class PerplexitySearchResult:
    '''
    Holds results of computing Tsne with different perplexities, and 
    clustering each with multiple n_clusters.

    Attributes:
		           data_df            : <df being analyzed>
		           tsne_df           : {perplexity : <df produced by tsne>}
		           cluster_results    : {perplexity : <ClusterResult objs>
		           optimal_perplexity : <perplexity that yielded the best clustering result>
		           optimal_n_clusters : <best n_cluster for the best optimal_complexity>
		
	Noteworthy methods:
		           silhouettes_df()
		           tsne_df(perplexity)
		           iter_tsne_dfs()
		           iter_cluster_results()
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, data_df):
        self.data_df  = data_df
        # Mapping of perplexity to tsne dfs:
        self._tsne_dfs = {}
        
        # Mapping of perplexity to ClusteringResult objs
        # that hold results of multiple KMeans computations
        # with different n_clusters: 
        self._clustering_results = {}
        
        self.optimal_perplexity = None
        self.optimal_n_clusters = None
        
        # Nobody asked for the silhouettes_df yet,
        # so its cache is empty:
        self._silhouettes_df_cache = None
        

    #------------------------------------
    # silhouettes_df
    #-------------------
    
    @property
    def silhouettes_df(self):
        '''
        Returns a dataframe with index being perplexities,
        and columns being n_clusters that were tried. The latter
        are in the _clustering_results. Fields are add_silhouette
        for the respective perplexity/n_clusters. Example: for
        two perplexities tried, each with n_clusters 2,4, and 10,
        you get:
        
                                N_CLUSTERS
                         2            4               10
            PERPLEXITY
                 5.0  silhouette  silhouette      silhouette
                10.0  silhouette  silhouette      silhouette
                
        The df is constructed only once, then cached
        '''
        
        if self._silhouettes_df_cache is not None:
            return self._silhouettes_df_cache
        
        # A dict mapping perplexities to pd.Series that will be rows in the result df:
        res = {}
        
        # Get perplexity multi-n_clusters result pairs:
        for perplexity, cl_res in self._clustering_results.items():

            # Get: {
            #        n_cluster1 : silhouettes1,
            #        n_cluster2 : silhouettes2,
            #            ... }
            n_clusters_silhouettes_dict = cl_res.silhouettes_dict()
            # Get a pandas Series: index is n_clusters, values are
            # perplexities. That's one row of the result df: 
            perplexity_cl_series = pd.Series(n_clusters_silhouettes_dict)
            res[perplexity] = perplexity_cl_series
        
        # Result for just one perplexity (value 5.0), and 
        # n_clusters 2 through 8 is like:
        # 
        #     {5.0: 2    0.517002
        #       	3    0.391266
        #       	4    0.248808
        #       	5    0.083461
        #       	6    0.060025
        #       	7    0.083971
        #       	8    0.030506
        #       	dtype: float32
        #           }
        #
        # where each value is a pd.Series.
        
        # Make a dataframe with perplexities being rows,
        # and the respective n_clusters the column (fields
        # are silhouette coefficients):
        
        if len(res) == 0:
            return pd.DataFrame()

        # The concat along cols builds a df
        # with perplexities being cols, and n_clusters being
        # index; therefore the transpose():
        sil_df = pd.concat(res, axis='columns').transpose()
        sil_df.index.name = 'Perplexities'
        
        self._silhouttes_df_cache = sil_df
        
        return sil_df
        
    #------------------------------------
    # clustering_result: special getter
    #-------------------
    
    def clustering_result(self, perplexity):
        '''
        Given a perplexity, return a ClusteringResult object
        
        :param perplexity:
        :type perplexity:
        '''
        if type(perplexity) == int:
            perplexity = float(perplexity)
        if type(perplexity) != float:
            raise TypeError(f"Must pass perplexity as a float, not {perplexity}")
        return self._clustering_results[perplexity]

    #------------------------------------
    # iter_cluster_results
    #-------------------
    
    def iter_cluster_results(self):
        '''
        Returns an iterator of 2-tuples: (perplexity, <ClusterResult obj>)
        '''
        return self._clustering_results.items()

    #------------------------------------
    # add_clustering_result
    #-------------------
    
    def add_clustering_result(self, perplexity, cluster_result):
        if type(perplexity) == int:
            perplexity = float(perplexity)
        if type(perplexity) != float:
            raise TypeError(f"Perplexity must be a float, not {perplexity}")
        if not isinstance(cluster_result, ClusteringResult):
            raise TypeError(f"Must pass ClusterResult as second part of tuple, not {cluster_result}")
        self._clustering_results[perplexity] = cluster_result

    #------------------------------------
    # tsne_df: special getter
    #-------------------
    
    def tsne_df(self, perplexity):
        if type(perplexity) == int:
            perplexity = float(perplexity)
        if type(perplexity) != float:
            raise TypeError(f"Must pass perplexity as a float, not {perplexity}")
        return self._tsne_dfs[perplexity]

    #------------------------------------
    # iter_tsne_dfs
    #-------------------
    
    def iter_tsne_dfs(self):
        '''
        Return an interator of all of this search result's tsne_dfs
        '''
        return self._tsne_dfs.values()
    

    #------------------------------------
    # tsne_df: setter
    #-------------------
    
    def add_tsne_dfs(self, tsne_dfs):
        '''
        Given a dict of {perplexity : tsne_df}, append those
        to the _tsne_dfs dict that is already in this result
        instance.  
        
        :param tsne_df: dict of perplexity to TSNE computation result
        :type tsne_df: dict[str : pd.DataFrame]
        '''
        self._tsne_dfs.update(tsne_dfs)
    
    #------------------------------------
    # to_json
    #-------------------

    def to_json(self, outfile=None):
        '''
        Exports the most important state of this instance
        of PerplexitySearchResult to JSON. Not everything is
        included, just:
        
             'optimal_perplexity',
             'optimal_n_clusters',
             'cluster_populations',
             'cluster_centers',
             'cluster_labels',
             'best_silhouette',
             'tsne_df'
        
        The inverse method: read_json() can produce a dict from this
        method's JSON output. But it cannot recover all the state.
        
        :param outfile: if provided, output file where resulting JSON is stored.
        :type outfile: union[None | str | file-like]
        :return JSON string if outfile is None, else returns None
        :rtype union[str | None]
        '''
        
        try:
            if type(outfile) == str:
                outfd = open(outfile, 'w')
            elif outfile is None:
                outfd = None
            elif Utils.is_file_like(outfile):
                # Outfile is file-like, i.e. has a write() method:
                outfd = outfile
            else:
                raise TypeError(f"Arg outfile must be None, filepath, or file-like, not {outfile}")
            
            optimal_cluster_res = self.clustering_result(self.optimal_perplexity)
            optimal_tsne_df     = self.tsne_df(self.optimal_perplexity)
            
            # If the df has datetime column(s), convert them to ISO strings: 
            # The filter expression below returns names of columns whose first-row
            # data's type is datetime. The body of the loop pulls each of
            # these columns from the df, and converts each of the resulting Series'
            # values to an ISO date string: 
            for colname in filter(lambda colname: isinstance(optimal_tsne_df[colname][0], datetime), optimal_tsne_df.columns):
                optimal_tsne_df[colname] = optimal_tsne_df[colname].apply(lambda date: date.isoformat())
            
            optimal_kmeans_obj  = optimal_cluster_res.get_kmeans_obj(self.optimal_n_clusters)
            res_dict = {}
            res_dict['optimal_perplexity']  = self.optimal_perplexity
            res_dict['optimal_n_clusters']  = self.optimal_n_clusters
            res_dict['cluster_populations'] = optimal_cluster_res.cluster_pops.tolist()
            res_dict['cluster_centers']     = optimal_kmeans_obj.cluster_centers_.tolist()
            res_dict['cluster_labels']      = optimal_kmeans_obj.labels_.tolist()
            res_dict['best_silhouette']     = float(optimal_cluster_res.best_silhouette)
            res_dict['tsne_df']             = optimal_tsne_df.to_json()
            
            if outfd is not None:
                json.dump(res_dict, outfd)
                return None
            else:
                jstr = json.dumps(res_dict)
                return jstr
        finally:
            if outfd is not None:
                outfd.close()
        
    #------------------------------------
    # read_json
    #-------------------
    
    @staticmethod
    def read_json(in_source):
        '''
        Given access to a JSON string that was produced by 
        the to_json() method, return a dict that contains the
        following keys:
        
             'optimal_perplexity',
             'optimal_n_clusters',
             'cluster_populations',
             'cluster_centers',
             'cluster_labels',
             'best_silhouette',
             'tsne_df'
        
        The string source may be a JSON string itself, or the path
        to a file that contains the JSON string, or an open file-like
        from which to read the JSON string.
        
        :param in_source: source of the JSON string to read
        :type in_source: union[str | path | file-like]
        :return a dict with the important state of the original PerplexitySearchResult
        :rtype dict[str : any]
        '''
        
        if Utils.is_file_like(in_source):
            jstr = in_source.read()
        elif os.path.exists(in_source):
            with open(in_source, 'r') as fd:
                jstr = fd.read()
        elif type(in_source) == str:
            jstr = in_source 
        else:
            raise TypeError(f"Input source must be a file-like, a path to a file, or a JSON string; not {in_source}")
            
            
        res_dict = json.loads(jstr)
        # Build the tsne_df from:
        #    '{"tsne_x":{"0":72.0819854736,"1":17.3076343536, ..., "9":-123.7122039795},
        #      "tsne_y":{"0":152.3043212891,"1":156.5667877197, ... ,"9":-123.7122039795}
        #      }'
        tsne_df = pd.read_json(StringIO(res_dict['tsne_df']))
        res_dict['tsne_df'] = tsne_df
        
        return res_dict
    
# ---------------------------------- Class ClusteringResult ---------------
class ClusteringResult:
    '''
    Holds final, and intermediate results of searches
    for the best Tsne perplexity and best KMeans n_clusters.
    Instances are just a convenient place to have everything
    organized during a search, and to easily access results.
    
    The results encapsulated are:
    
            kmeans_objs      : {n_clusters1 : <kmeans-obj-from-n_clusters1>,
                                n_clusters2 : <kmeans-obj-from-n_clusters2>,
                                      ...
                                }
            add_silhouette      : {n_clusters1 : <silhouette-coefficient1>,
                                n_clusters2 : <silhouette-coefficient2>,
                                      ...
                                }   
            tsne_df          : the passed-in Tsne dataframe
            best_n_clusters  : the list of cluster labels from the kmeans
                               that yielded the best result, i.e. the 
                               highest silhouette coefficient.
                               None if caller provided an n_cluster
            best_kmeans      : the KMeans object that yielded the highest
                               silhouette coefficient. 
            best_silhouette  : maximum silhouette achieved: the one
                               for best_n_cluster
            cluster_pops     : point populations for each cluster when
                               using best_nclusters for the KMeans population

    Get and set methods are provided for the kmeans_objs attribute.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self):
        
        # Map of n_cluster to kmeans objects:
        self._kmeans_objs    = {}
        # Map of n_cluster to resulting add_silhouette coefficient
        self._silhouettes    = {}
        
        self.tsne_df         = None 
        self.best_n_clusters = None 
        self.best_silhouette = None
        self.cluster_pops    = None

    #------------------------------------
    # get_silhouette
    #-------------------
    
    def get_silhouette(self, n_clusters):
        if type(n_clusters) != int:
            raise TypeError(f"Must pass number of clusters as an int, not {n_clusters}")
        return self._silhouettes[n_clusters]

    #------------------------------------
    # add_silhouette
    #-------------------
    
    def add_silhouette(self, n_clusters, silhouette):
        if type(n_clusters) != int:
            raise TypeError(f"Must integer n_clusters not {n_clusters}")
        self._silhouettes[n_clusters] = silhouette

    #------------------------------------
    # silhouettes_dict
    #-------------------
    
    def silhouettes_dict(self):
        '''
        Returns the dict {n_clusters : silhouette-coefficient}
        '''
        return self._silhouettes

    #------------------------------------
    # max_silhouette_n_cluster
    #-------------------
    
    def max_silhouette_n_cluster(self):
        '''
        Returns the n_clusters value that yielded the 
        maximum silhouette.
        '''
        
        silhouettes = list(self._silhouettes.values())
        max_sil_pos = silhouettes.index(max(silhouettes))
        best_n_clusters = list(self._silhouettes.keys())[max_sil_pos]
        return best_n_clusters
    
    #------------------------------------
    # get_kmeans_obj
    #-------------------
    
    def get_kmeans_obj(self, n_clusters):
        if type(n_clusters) != int:
            raise TypeError(f"Must pass number of clusters as an int, not {n_clusters}")
        return self._kmeans_objs[n_clusters]

    #------------------------------------
    # add_kmeans_obj
    #-------------------
    
    def add_kmeans_obj(self, n_clusters, kmeans_obj):
        if type(n_clusters) != int:
            raise TypeError(f"Must pass int for n_clusters, not {n_clusters}")
        
        self._kmeans_objs[n_clusters] = kmeans_obj

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        repr_str = f"ClusteringResult (kmeans run: {len(self._kmeans_objs)}) at {hex(id(self))}"
        return repr_str

    #------------------------------------
    # __str__
    #-------------------
    
    def __str__(self):
        return self.__repr__()

    
# ---------------------------------- Class DataCalcs ---------------

class DataCalcs:
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

    def __init__(self,
                 measures_root, 
                 inference_root, 
                 chirp_id_src=ChirpIdSrc('file_id', ['chirp_idx']),
                 cols_to_retain=None
                 ):
        '''
        Calculations for analyzing processed SonoBat data.
        These data are normalized results from SonoBat classification
        runs. Those contain measurements that characterize each chirp.
        
        The chirp_id_src is a named tuple with two fields: 'wav_file_nm_col',
        and 'more_key_cols'. The first is the col name that hold the .wav
        file name from which the chirp is taken. The second field contains
        a list of additional column names in the measurement files that,
        together with the .wav file name, uniquely identifies the chirp.
        
        By default, the 'file_id' column contains the .wav file name, and
        the additional column needed is the 'chirp_idx.
        
        If cols_to_retain is not none, it needs to be a list of additional
        column names in the measurements that should be transferred to the
        TSNE result embedding. For instance, if cols_to_retain is 
        ['more_col1', 'more_col2'], the TSNE result df will have columns:
        
           tsne_x    tsne_y    more_col1           more_col2 
            xxx1       yyy1  from-measures1_1   from-measures1_2
            xxx2       yyy2  from-measures2_1   from-measures2_2
                               ...
        
        Other data are next-chirp predictions.
        
        NOTE: we assume that measures_root contains a file called
              split_filename_to_id.csv. It must map .wav file names
              to the file_id values used in measurement files:
              
				               Filename,                file_id
				     barn1_D20220205T192049m784-HiF.wav,11
				     barn1_D20220205T200541m567-Myca-Myca.wav,12
                                     ...
        
        :param measures_root: root directory of normalized 
            SonoBat measurements in 'split' files (split1.feather, 
            split2.feather, etc) 
        :type measures_root: str
        :param inference_root: root directory of transformer next-chirp
            predictions
        :type inference_root: src
        :param chirp_id_src: name of column, or list of columns that uniquely
            identify a chirp, including the .wav file of which the chirp
            is a part. That file name includes the recording date and time
        :type chirp_id_src: union[str | list[str]]
        :param cols_to_retain: columns to carry over from measurements df to
            tsne df
        :type cols_to_retain: union[None | list[str]]
        '''

        self.measures_root  = measures_root
        self.inference_root = inference_root
        self.chirp_id_src   = chirp_id_src
        self.cols_to_retain = cols_to_retain
        
        self.log = LoggingService()
        
        # Read the mapping of .wav file to file_id values in 
        # the measurements file_id column. The header
        # is ['Filename', 'file_id']
        map_file = os.path.join(measures_root, 'split_filename_to_id.csv')
        with open(map_file, 'r') as fd:
            fname_id_pairs = list(csv.reader(fd))
        # We now have:
        #   [[fname1, id1],
        #    [fname2, id2],
        #        ...
        #    ]
        # Build id -> fname dict:
        # We need the reverse dict: ID-->wav_fname
        # (the fname_id_pairs[1:] skips over the header:
        self.fid_to_fname_dict = {int(fid) : fname
                                  for fname, fid
                                  in fname_id_pairs[1:]
                                  } 
        
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
        
        # Create self.fid2split_dict for mapping a split file id
        # to the split file that contains the data for that
        # file id:
        
        #self._make_fid2split_dict()
        
        # Cache of dfs of measures split files we had 
        # to open so far: maps split id to df:
        self.split_file_dfs_cache = {}
        
        # Make lookup dicts to find all possible (sin, cos) pairs for
        # any time granularity. Example:
        self.trig_secs_lookup    = {sec : Utils.cycle_time(sec, TimeGranularity.SECONDS)
                                    for sec in range(1, 60)}
        self.trig_mins_lookup    = {min_ : Utils.cycle_time(min_, TimeGranularity.MINUTES)
                                    for min_ in range(1, 60)}
        self.trig_hrs_lookup     = {hr : Utils.cycle_time(hr, TimeGranularity.HOURS)
                                    for hr in range(0, 24)}
        self.trig_days_lookup    = {day : Utils.cycle_time(day, TimeGranularity.DAYS)
                                    for day in range(1, 32)}
        self.trig_months_lookup  = {month : Utils.cycle_time(month, TimeGranularity.MONTHS)
                                    for month in range(1, 13)}
        self.trig_years_lookup   = {year : Utils.cycle_time(year, TimeGranularity.YEARS)
                                    for year in range(0, 10)}

        self.hr_dy_mn_yr_dicts = [self.trig_hrs_lookup,
                                  self.trig_days_lookup,
                                  self.trig_months_lookup,
                                  self.trig_years_lookup
                                  ]                                
        

    #------------------------------------
    # _make_fid2split_dict
    #-------------------
    
    # def _make_fid2split_dict(self):
    #     '''
    #     Create a dict mapping measures file identifier ints
    #     to the measures split file that contains the measure
    #     created from the file identified by the file id. Like
    #
    #         10 : '/foo/bar/split40.feather',
    #         43 : '/foo/bar/split4.feather',
    #                   ...
    #
    #     This dict is used, for example, to retrieve the chirp measures 
    #     that correspond to a given T-sne point.
    #
    #     The result will be in self.fid2split_dict.
    #     '''
    #
    #     self.fid2split_dict = {}
    #     for fpath in self.split_fpaths.values():
    #         fids_df = pd.read_feather(fpath, columns=[self.chirp_id_src.wav_file_nm_col])
    #         # Get list of file ids in this split file as a list:
    #         fids = fids_df.file_id.values
    #
    #         # Add all this split file's file ids to the
    #         # fid2split_dict:
    #         self.fid2split_dict.update({fid : fpath for fid in fids})

    #------------------------------------
    # measures_from_fid
    #-------------------
    
    # def measures_from_fid(self, fid):
    #     '''
    #     Given a measures file id, return a pandas
    #     Series with the measures.
    #
    #     :param fid: file id that identifies the row
    #         in a dataset where the related chirp measures
    #         are stored.
    #     :type fid: int
    #     :return the chirp measures created by SonoBat
    #     :rtype pd.Series
    #     '''
    #
    #     try:
    #         df = self.split_file_dfs_cache[fid]
    #         measures = df.loc[fid]
    #         return measures
    #     except KeyError:
    #         # Df not available yet:
    #         pass
    #
    #     split_path = self.fid2split_dict[fid]
    #     df = pd.read_feather(split_path)
    #
    #     # Take the .wav file names, extract 
    #     # the recording datetime, and replace
    #     # the values in the file_id column with
    #     # date and time of the recording:
    #     wav_file_col_nm = self.chirp_id_src.wav_file_nm_col 
    #     # We now have a column of .wav file integer IDs. 
    #     # Resolve those into their .wav file names:
    #     fnames = [self.fid_to_fname_dict[fid]
    #               for fid 
    #               in df[wav_file_col_nm]
    #               ]
    #
    #     rec_times = list(map(lambda fname: Utils.time_from_fname(fname), fnames))
    #
    #     # Adjust the column name:
    #     df.rename({wav_file_col_nm : 'rec_datetime'}, axis='columns', inplace=True)
    #     df['rec_datetime'] = rec_times
    #
    #     # Add a column 'daytime' with True or False, depending
    #     # on whether the recording was at daytime as seen at
    #     # Stanford's Jasper Ridge Preserve:
    #     was_day_rec = df['rec_datetime'].apply(lambda dt: Utils.is_daytime_recording(dt))
    #     df.loc[:, 'is_daytime'] = was_day_rec
    #
    #     # Use the default index of 0,1,2,...
    #     # df.reset_index(drop=True, inplace=True)
    #
    #     measures = df.loc[fid]
    #     return measures   
    
    #------------------------------------
    # sort_by_variance
    #-------------------
    
    @classmethod
    def sort_by_variance(cls, measures, delete_non_measures=True):
        '''
        Given a list of measure names (i.e. SonoBat classifier-produced
        column names), return a new list of the same names, 
        sorted by decreasing variance in the overall data. 
        
        Treatment of columns that are not SonoBat measures depends 
        on the value of delete_non_measures. If it is true, such
        columns won't be in the returned sorted list. If False, 
        the non_measure elements will be at the end of the list. 
        
        :param measures: list of measure names to sort
        :type measures: lisg[str]
        :param delete_non_measures: whether or not to delete columns
            that are not SonoBat measures
        :rtype bool
        :return names sorted by decreasing variance
        :rtype list[str]
        '''
        
        # List of columns that are not known SonoBat measurements
        non_measure_els = []
        # For informative error msg if name
        # is passed in that is not a SonoBat measure:
        global curr_el
        def key_func(el):
            # Initialize curr_el to be the 
            # element currently being sorted:
            global curr_el
            curr_el = el
            try:
                return cls.sorted_mnames.index(el)
            except ValueError:
                # Given name is not in the list 
                # of top SonoBat chirp measures.
                # Put it at the end of the result
                # list for now (we delete them later
                # if requested):
                non_measure_els.append(el)
                return float("inf") 
            
        sorted_list_unabridged = sorted(measures, key=key_func)
        if delete_non_measures:
            new_list = [el
                        for el in sorted_list_unabridged
                        if el not in non_measure_els
                        ]
        else:
            new_list = sorted_list_unabridged 
            
        return new_list
    
    #------------------------------------
    # measures_by_var_rank
    #-------------------
    
    @classmethod
    def measures_by_var_rank(cls, 
                             src_info, 
                             min_var_rank,
                             cols_to_keep=None
                             ):
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
        
        The cols_to_keep argument may list columns to retain. These
        might be non-measure columns, such as identifiers, datetimes, etc.
          
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
            
        # Now that we have the df, check whether 
        # caller wants columns to be kept in the returned
        # df, which are not in the given df:
        if cols_to_keep is not None:
            if not all([col_nm in df.columns for col_nm in cols_to_keep]):
                raise ValueError(f"The cols_to_keep list contains column(s) that are not in the dataframe")

        given_cols = df.columns

        # Get (only true measures) columns, sorted by decreasing
        # variance:        
        cols_by_variance = cls.sort_by_variance(given_cols)
        try:
            cols_wanted = cols_by_variance[:min_var_rank]
        except IndexError:
            # Desired minimum rank is larger than the 
            # number of SonoBat measures, so use the whole
            # source df:
            cols_wanted = cols_by_variance
            
        # Df of just the wanted measures:
        new_df = df[cols_wanted]
        
        if cols_to_keep is not None:
            # 'Attach' the to-kept columns to the right of the measures,
            # if they are not already in the df:
            cols_to_attach = list(set(cols_to_keep) - set(cls.sorted_mnames))
            # Pull the missing columns from the original df:
            sub_df = df[cols_to_attach]
            new_df = pd.concat([new_df, sub_df], axis='columns')
        return new_df

    #------------------------------------
    # add_recording_datetime
    #-------------------

    def add_recording_datetime(self, df):
        '''
        Modifies df in three ways:
        
            o Replaces the values of the .wav file id column
              with datetime objects that hold the recording time
              of the row's chirp
            o Renames the column name for the .wav fname source
              (by default 'file_id' with the name 'rec_datetime'
            o Adds a new boolean column 'is_daytime' that indicates
              whether the chirp was recorded during daytime.
              
        These changes occur in place.  
        
        :param df: dataframe to modify
        :type df: pd.DataFrame
        :return the modified df
        :rtype pd.DataFrame
        '''
        # Take the .wav file names, extract 
        # the recording datetime, and replace
        # the values in the file_id column with
        # date and time of the recording. The column
        # that names the .wav file of each row (i.e. of 
        # each chirp) contains integer file IDs for the
        # .wav files. Resolve that first:
        
        # Name of column with .wav file ID ints:
        wav_file_col_nm = self.chirp_id_src.wav_file_nm_col 
        # Resolve those into their .wav file names:
        fnames = [self.fid_to_fname_dict[fid]
                  for fid 
                  in df[wav_file_col_nm]
                  ]
    
        rec_times = list(map(lambda fname: Utils.time_from_fname(fname), fnames))
        rec_times_series = pd.Series(rec_times, name='rec_datetime')
        # Add recording times column:
        df['rec_datetime'] = rec_times_series
    
        # Add a column 'daytime' with True or False, depending
        # on whether the recording was at daytime as seen at
        # Stanford's Jasper Ridge Preserve:
        was_day_rec = df['rec_datetime'].apply(lambda dt: Utils.is_daytime_recording(dt))
        df.loc[:, 'is_daytime'] = was_day_rec

        return df

    #------------------------------------
    # run_tsne
    #-------------------
    
    def run_tsne(self,
                 df,
                 #******num_points=10000, 
                 num_points=100,
                 num_dims=50,
                 perplexity=None,
                 sort_by_bat_variance=True,
                 cols_to_keep=[]
                 ):
        '''
        Infile must be a .feather, .csv. or .csv.gz file
        of SonoBat measures. The file is loaded into a DataFrame. 
        
        This function pulls the first num_dims columns from that
        df, and limits the T-sne embedding input to those columns.
        If num_dims is None, all input columns are retained. 
        
        If sort_by_bat_variance is True, columns must be SonoBat
        program bat chirp measurement names. With that switch 
        being True, the columns of the input df are sorted by 
        decreasing variance over the Jasper Ridge bat recordings.
        The num_dims columns are then taken from that sorted list
        of measurement result columns. 
        
        after sorting the columns by decreasing variance.
        That is the num_dims'th ranked variance measures are used
        for each chirp. 
        
        Uses the min(num_points, dfsize) (i.e. num_points rows) from the df.
        
        After running the number of points will be in:
        
                self.effective_num_points
        
        Tsne is run over the resulting num_points x num_dims dataframe.  
        
        :param df: the dataframe to examine
        :type df: pd.DataFrame
        :param num_points: number of chirps to include from the 
            given self.infile.
        :type num_points: int
        :param num_dims: number of measures of each chirp to use.
            I.e. how many columns.
        :type num_dims:
        :param point_id_col: name of column to use for identifying each
            data point in the clustering. The values of that column will
            be placed into the df index. If None, the index will be the
            default (0,1,2,...)
        :type point_id_col: union[None | str]
        :param perplexity: the perplexity hyper parameter value.
            If None, the TSNE constructor's default is chosen: 30 or
            one less than the length of the df.
        :type perplexity: union[None | float]
        :param sort_by_bat_variance: if True, all column names must
            be SonoBat measure names. The num_dims number of columns
            will be selected from the input dataframe such that they
            have highest rank in variance over the bat recordings.
        :type sort_by_bat_variance: bool
        :param cols_to_keep: list of columns from the df that is being
            tsne-fied that should be carried over unchanged to the tsne df,
            which usually only has two cols: tsne_x, and tsne_y
        :param cols_to_keep: union[None | list[str]
        :result the T-sne embeddings
        :rtype pd.DataFrame
        '''
        
        if cols_to_keep is None:
            cols_to_keep = []
        else:
            # Check whether caller wants columns to be kept in the returned
            # df, which are not in the given df:
            if not all([col_nm in df.columns for col_nm in cols_to_keep]):
                raise ValueError(f"The cols_to_keep list contains column(s) that are not in the dataframe")
        
        # Sort by variance, and cut off below-threshold
        # columns:
        if sort_by_bat_variance:
            
            self.log.info(f"Tsne: sorting cols; selecting {num_dims}")
            df_all_rows = DataCalcs.measures_by_var_rank(df, 
                                                         min_var_rank=num_dims,
                                                         cols_to_keep=cols_to_keep
                                                         )
            
        else:
            df_all_rows = df
        
        # Keep only the wanted rows:
        if type(num_points) == int: 
            df = df_all_rows.loc[0:num_points-1, :]
        elif num_points is None:
            df = df_all_rows
        else:
            raise TypeError(f"The num_points arg must be None or an integer, not {num_points}")

        # Perplexity must be less than number of points:
        if perplexity is not None:
            if type(perplexity) != float:
                raise TypeError(f"Perplexity must be None, or float, not {perplexity}")
            if perplexity >= len(df):
                perplexity = float(len(df) - 1)
        else:
            # Mirror the TSNE constructor's default:
            perplexity = min(30.0, len(df)-1)

        effective_num_points = len(df)
        log_msg = (f"Running tsne; perplexity '{perplexity}';"
                   f"num_dims: {num_dims};"
                   f"num_points: {effective_num_points}")
        self.log.info(log_msg)

        tsne_obj = TSNE(n_components=2, 
                        init='pca', 
                        perplexity=perplexity,
                        metric='cosine',
                        n_jobs=8,
                        random_state=3
                        )
        # tsne_obj = openTSNE.TSNE(
        #         perplexity=perplexity,
        #         initialization="pca",
        #         metric="cosine",
        #         n_jobs=8,
        #         random_state=3,
        #         )
        
        # Only use columns for TSNE computation that 
        # are both, in the measures, and are wanted
        # for carry-over:

        cols_to_use = list(set(self.sorted_mnames).intersection(set(df.columns)))
        embedding_arr = tsne_obj.fit_transform(df.loc[:, cols_to_use])
        
        # For each embedding point, add the cols_to_keep, after
        # turning the np.array TSNE result into a df:
        tsne_df_abbridged = pd.DataFrame(embedding_arr, columns=['tsne_x', 'tsne_y'])
        # Attach the source df to the TSNE result:
        tsne_df = pd.concat([tsne_df_abbridged, df], axis='columns')
        return tsne_df
    
    #------------------------------------
    # cluster_tsne
    #-------------------
    
    def cluster_tsne(self, tsne_df, n_clusters=None, cluster_range=range(2,10)):
        '''
        Computes Kmeans on the tsne dataframe. The returned
        Kmeans object can be interrogated for the assignment
        of each Tsne-embedded data point to a cluster. Example
        for a Tsne df of 10 rows (i.e. 10 datapoints in both the
        orginal df, and the Tsne embedding). Something like this:
        
            kmeans-obj.labels_ => expected: [0, 0, 2, 0, 2, 1, 0, 2, 2, 1]
            
        The numbers are cluster labels.
        
        Cluster centers in Tsne space are available as:
        
            kmeans-obj.cluster_centers():
          returns like: array([[ 17.117634 ,  23.18263  ],
                               [  6.5119925, -38.010742 ],
                               [-52.590286 ,   3.915401 ]], dtype=float32)
      
        If n_clusters is None, this method finds the best n_clusters by
        the silhouette method, trying range(2,10), subject to 2 <= n_clusters <= numSamples-1
        
        In this case the silhouette coefficients for each tested n_clusters 
        will be available in the 
        
                self.silhouettes
                
        attribute. If an n_cluster is provided by the caller, this attribute
        will be undefined.
        
        Also in the case of the given n_clusters being None, the computed 
        silhouette coefficient for each tried n_clusters will be available
        in the list attribute:
        
                self.n_cluster_range
        
        The kMeans object will be available in:
        
                self.kmeans
                
        Sizes of clusters in number of contained points will be available in
        
                self.cluster_populations
                
        Returned is an instance of ClusteringResult, which contains
            o a mapping of n_clusters to their kmeans objects
            o a mapping of n_clusters to resulting silhouette coefficient
            o the given tsne_df, 
            o the best_kmeans object, the one for which n_clusters is best_n_clusters, 
            o the best_n_clusters found, 
            o the best_silhouette coefficient, the one corresponding to the best_n_clusters,
            o the populations of each cluster for the optimal kmeans.
                
        :param tsne_df: dataframe of points in Tsne space, as returned
            by the run_tsne() method.
        :type tsne_df: pd.DataFrame
        :param n_clusters: number of clusters to find. If None,
            a silhouette analysis is performed over n_cluster in [2...10].
            The n_cluster resulting in the largest average silhouette coefficient
            is chosen as n_clusters 
        :type n_clusters: UNION[None | int]
        :param cluster_range: Used when n_clusters is None. The range of
            n_cluster values to test for leading to optimal clustering.
        :type cluster_range: range
        :return: all results
        :rtype ClusteringResult
        '''

        # Start filling in the result
        result = ClusteringResult()
        result.tsne_df = tsne_df


        np_arr = tsne_df[['tsne_x', 'tsne_y']].to_numpy()
        # Case: caller specified the number of
        # clusters to find:
        if n_clusters is not None:
            
            self.log.info(f"Clustering Tsne result to {n_clusters} clusters")
            
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(np_arr)
            # Just one n_cluster/kmeans pair:
            result.add_kmeans_obj(n_clusters, kmeans)
            result.cluster_pops = np.bincount(kmeans.labels_)
            
            # Make the only kmeans object the 'best' one:
            result.best_kmeans = kmeans

            # Did not look for best n_cluster, since caller provided it:
            result.best_n_clusters  = None
            result.best_silhouette  = None

            return result

        # Need to find best number of clusters via
        # silhouette method. Try different n_clusters
        # as specified in the cluster_range arg:
        
        self.n_cluster_range = list(cluster_range)
        num_samples = len(np_arr)
                
        # List of silhouette coefficients resulting
        # from each n_cluster: 
        self.silhouettes = []
        
        self.log.info(f"Finding best n_clusters...")
        
        for n_clusters in cluster_range:
            
            # n_clusters must be 2 <= n_clusters <= n_samples-1
            # for silhouette coefficient to be define:
            if not (2 <= n_clusters <= len(np_arr)-1):
                warn = (f"Not trying n_clusters {n_clusters} to {self.n_cluster_range[-1]}, "
                        f"because number of samples is {num_samples}, and "
                        f"(2<=n_clusters<={num_samples}) must be true")
                self.log.warn(warn)
                break
            
            self.log.info(f"Computing KMeans for {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(np_arr)
            # Save the computed kmeans as the i'th iteration:
            result.add_kmeans_obj(n_clusters, kmeans)
            # Compute the silhouette coefficient for this kmeans,
            # and append it to the silhouttes list in the i'th position: 
            result.add_silhouette(n_clusters, silhouette_score(np_arr, cluster_labels))
            
        # Find the index of the largest (average) silhouette:
        max_silhouette_n_clusters = result.max_silhouette_n_cluster()
        # Now we know which of the kmeans results to use:
        kmeans = result.get_kmeans_obj(max_silhouette_n_clusters)
        # The best n_clusters:
        result.best_n_clusters = len(set(kmeans.labels_))
        result.best_kmeans     = kmeans
        result.best_silhouette = result.get_silhouette(result.best_n_clusters)
        
        self.log.info(f"Best silhouette (-1 to 1; 1 is best): {result.best_silhouette}; n_clusters: {result.best_n_clusters}")
        
        self.kmeans = kmeans
        self.cluster_populations = np.bincount(kmeans.labels_)
        result.cluster_pops = np.bincount(kmeans.labels_)
        
        return result

    #------------------------------------
    # find_optimal_tsne_clustering
    #-------------------
    
    def find_optimal_tsne_clustering(
            self,
            df, 
            perplexities=[5.0, 10.0, 20.0, 30.0, 50.0],
            n_clusters_to_try=list(range(2,10)),
            outfile=None,
            cols_to_keep=['Max#CallsConsidered', 'rec_datetime', 'is_daytime', 'chirp_idx']
            ):
        '''
        Determine the Tsne 'perplexity' hyper parameter, and
        the optimal number of clusters for KMeans of Tsne 
        embeddings.
        
        The cols_to_keep argument instructs the Tsne-returned to
        include the provided columns from the data df. For example,
        say, the original df includes column 'NodeID', and the caller
        wishes later to identify points in Tsne space by NodeID, the
        caller would make cols_to_keep be ['NodeID']. The returned
        Tsne df will then include the NodeID for wach tsne-x/tsne-y pair.
        If a column is requested for keeping that is not in the given
        dataframe, a ValueError results.
        
        Returned is a PerplexitySearchResult that contains:
        
        PerplexitySearchResult:
            o optimal_perplexity
            o tsne_df(perplexity)          : access to dict {perplexity : TSNE result df}
            o clustering_result(perplexity): access to dict {perplexity : ClusteringResult}
            o silhouettes_df               : df of cluster_n x perplexity; cells are perplexity coefficients.
        
        where Clustering Result:    
            o best_n_clusters
            o kmeans_objs
            o tsne_df
            o best_kmeans
            o best_silhouette
            o cluster_pops
        
        :param df: the dataframe of normalized SonoBat measures
            over which to run the analyses
        :type df: pd.DataFrame
        :param perplexities: the perplexity values for which to 
            run Tsne analysis
        :type perplexities: list[float]
        :param n_clusters_to_try: the number of clusters to try
            passing into KMeans.
        :type n_clusters_to_try: list[int]
        :param outfile: file to write the PerplexitySearchResult 
            instance. If None: result is not written out, and is just returned.
        :rtype outfile: union[None | str | file-like]
        :param cols_to_keep:
        :type cols_to_keep: union[None | list[str]] 
        :return a PerplexitySearchResult containing
        :rtype PerplexitySearchResult
        '''

        if cols_to_keep not in (None, []):
            # Check whether caller wants columns to be kept in the returned
            # df, which are not in the given df:
            if not all([col_nm in df.columns for col_nm in cols_to_keep]):
                wanted_set = set(cols_to_keep)
                avail_set  = set(df.columns)
                missing    = wanted_set - avail_set
                self.log.warn(f"Wanted cols unavailable in split file: {missing}")
                cols_to_keep = list(wanted_set - missing)
        num_samples = len(df)
        # Perplexity must be less than number of samples:
        good_perplexities = [perp for perp in perplexities if perp < num_samples]
        # Give warning about skipping perplexities, if needed:
        if good_perplexities != perplexities:
            dropped_perps = set(perplexities).difference(set(good_perplexities))
            self.log.warn(f"Not testing perplexities {sorted(dropped_perps)}, because they are larger than number of samples ({num_samples}).")

        # Run multiple Tsne embedding calculations, each
        # with a different perplexity. Result will be
        # a dict mapping perplexities to TSNE embedding coordinate dataframes:
        
        result = PerplexitySearchResult(df)
        
        self.tsne_df   = {perp : self.run_tsne(df, perplexity=perp, cols_to_keep=cols_to_keep)
                           for perp in good_perplexities}
        
        result.add_tsne_dfs(self.tsne_df)

        # Run KMeans on each of the Tsne embeddings, noting
        # the silhouette coefficient for each clustering.
        # A coefficient of 1 indicates great clustering.
                
        # Save the (by default eight) silhouettes that each
        # KMeans computation places into self.add_silhouette,
        # creating a nested array 'silhouettes'
        #       [[sil2Clusters, sil3Clusters,...], # for Tsne perplexity 5.0
        #        [sil2Clusters, sil3Clusters,...], # for Tsen perplexity 20.0
        #                  ...
        #        ]
        self.kmeans_objs = []
        for perplexity, tsne_df in self.tsne_df.items():
            # Run multiple kmeans, specifying different number of clusters:
            clustering_result = self.cluster_tsne(tsne_df, cluster_range=n_clusters_to_try)
        
            result.add_clustering_result(perplexity, clustering_result)    

        # Find the best tsne perplexity and n_clusters_to_try by
        # building the following df:
        #
        #                     N_CLUSTERS
        #                  2      3  ...  9
        #    PERPLEXITY
        #        5.0      0.3    0.8 ... 0.4
        #       10.0           ...
        #       20.0           ...
        #       30.0           ...
        #       50.0           ...
        #        
        # Then find the max cell, which is the highest 
        # silhouette coefficient:

        # Find the perlexity and n_clusters that correspond
        # to the highest silhouette coefficient in the dataframe:
        (result.optimal_perplexity, result.optimal_n_clusters) = \
            Utils.max_df_coords(result.silhouettes_df)
            
        if outfile is not None:
            result.to_json(outfile)
    
        return result

    #------------------------------------
    # make_chirp_sample_file
    #-------------------
    
    def make_chirp_sample_file(self, num_chirps, save_dir=None, unittests=None):
        '''
        Opens each of the n measurements files in self.measures_root.
        From each of the resulting df's randomly selects num_chirp/n
        chirps (i.e. rows). Builds one df from the results.
        
        If unittests is a list, then the list must contain dataframes
        such as the ones that would normally be pulled from split files.
        Those dfs will then be used for testing the implementation. If
        unittests is None, then the paths to split files is used to 
        obtain dataframes from which to sample. Those are in self.split_fpaths.
        
        Result df will have:
             o All the columns in the split files
             o Columns
                	'sin_hr', 'cos_hr', 
                	'sin_day', 'cos_day', 
                	'sin_month', 'cos_month', 
                	'sin_year', 'cos_year'               
               
               which are each rec_datetime's hour, day, month, etc.
               mapped onto a circle to detect periodicity.
               
        If 'save_dir' contains a directory, saves the df in .csv 
        format in that directory. The file name will be:
        
                  <self.res_file_prefix>_samples_<timestamp>.csv
         
                
        :param num_chirps: total number of chirps to sample across 
            all split files.
        :type num_chirps: int
        :param save_dir: if provided, save resulting df to
            <save_dir>/<self.res_file_prefix>_samples_<timestamp>.csv
        :type save_dir: union[None | src]
        :param unittests: whether or not to save the resulting df
        :type unittests: union[None | list[pd.DataFrame]]
        :return the constructed df
        :rtype pd.DataFrame
        '''

        # Number of rows in each df:
        fsizes = {}

        # Make copy of the split file path list,
        # because we may need to go through the
        # loop more than once, and want to randomize
        # the list on rounds 2-plus:
        if type(unittests) == list:
            sample_paths = unittests
        else:
            sample_paths = self.split_fpaths.copy()
        # Need num_chirps rows total:
        rows_needed = num_chirps
        
        # Numpy array for final result rows:
        res_np_arr = None
        
        res_cols = None
        
        # Place to remember the true number of
        # rows in each file:
        fsizes_true = {}
        
        # Must sample without replacement from each df:
        # dict mapping nth df to set of row indices already
        # pulled:
        row_idxs_taken = {nth : set() for nth in range(len(sample_paths))}
        
        # Go through each split file, possibly multiple times.
        # During unittests, sample_paths are actually dataframes
        # pre-loaded from actual split files: 
        while True:
            for ith_src, measure_src in enumerate(sample_paths):
                if type(measure_src) == str:
                    # Load the df:
                    df_one_split = UniversalFd(measure_src, 'r').asdf()
                else:
                    df_one_split = measure_src
                
                # First time through loop:
                if res_cols is None:
                    # All cols are the same across splits:
                    res_cols = df_one_split.columns
                    # Make an empty np array with proper dimensions
                    # into which we will collect rows:
                    res_np_arr = np.empty((0, len(res_cols)), float)
                
                    # Estimate how may rows to take from each file,
                    # assuming for now that all files have as many
                    # rows as the first df that we just read:
                    fsizes = {nth : len(df_one_split)
                              for nth 
                              in range(len(sample_paths))
                              }
                    to_take = self._allocate_sample_takes(fsizes, rows_needed, row_idxs_taken)


                rows_in_this_df = len(df_one_split)
                # Remember the true number of rows in this df:
                fsizes_true[ith_src] = rows_in_this_df
                
                # Get list of random row numbers to pull from this df,
                # taking care not to pull rows we pulled in an earlier
                # run through the loop:
                taken = row_idxs_taken[ith_src]
                elligible    = list(set(range(rows_in_this_df)) - taken)
                num_to_take  = min(len(elligible), to_take[ith_src])
                row_nums = random.sample(elligible, num_to_take)
                # Update what we've already taken from this file:
                row_idxs_taken[ith_src].update(row_nums)
    
                # Pull out the lines:
                res_np_arr = np.concatenate([res_np_arr, df_one_split.iloc[row_nums].to_numpy()])
    
            num_samples = len(res_np_arr)
            
            # Do we have enough samples? Or maybe too many?
            if num_samples == rows_needed:
                break
            elif num_samples > rows_needed:
                # Randomly remove rows:
                num_to_cull = num_samples - num_chirps
                # Select
                idxs = random.sample(range(num_samples), num_to_cull)
                res_np_arr = np.delete(res_np_arr, idxs, axis=0)
                break
            else:
                # Must add more samples:
                rows_needed = num_chirps - num_samples
                # We now know the number of rows in all the files.
                # So we can take the remainder of the samples from
                # all files in equal proportion:
                to_take = self._allocate_sample_takes(fsizes_true, rows_needed, row_idxs_taken)
                
                continue

        # Make a df from the extraced samples:            
        df = pd.DataFrame(res_np_arr, columns=res_cols)
        df.reset_index(drop=True, inplace=True)
        return df

    #------------------------------------
    # _add_trig_cols
    #-------------------
    
    def _add_trig_cols(self, df, dt_col_nm):
        '''
        Given a df with a column named dt_col, add columns
        
                	'sin_hr', 'cos_hr', 
                	'sin_day', 'cos_day', 
                	'sin_month', 'cos_month', 
                	'sin_year', 'cos_year'               
               
        which are each rec_datetime's hour, day, month, etc.
        mapped onto a circle to detect periodicity.
        
        :param df: dataframe holding a datetime column for which
            to add the trig function columns
        :type df: pd.DataFrame
        :param dt_col_nm: name of column that contains the 
            datetime.datetime instances
        :type dt_col_nm: str
        :return dataframe with columns added on the right
        :rtype pd.DataFrame
        '''

        # For each recording time, add sin and cos mappings for
        # the times' hour, day, month, and year. I.e. add 
        # eight additional columns:
        cols = ['sin_hr', 'cos_hr', 
                'sin_day', 'cos_day', 
                'sin_month', 'cos_month', 
                'sin_year', 'cos_year']  
        # Build an array of pd.Series, each with the
        # needed six trigonometric columns: 
        trig_rows = list(map(self._sin_cos_cache, df[dt_col_nm]))
        
        new_df = pd.concat([df, pd.DataFrame(trig_rows, columns=cols)], axis='columns')
        
        return new_df


    #------------------------------------
    # _sin_cos_cache
    #-------------------
    
    def _sin_cos_cache(self, dt):
        
        sin_cos_pairs = [trig_dict[time_val]
                         for trig_dict, time_val 
                         in zip(self.hr_dy_mn_yr_dicts, 
                                (dt.hour, dt.day, dt.month, dt.year % 10))]
        return tuple(chain(*sin_cos_pairs))
            

    #------------------------------------
    # _allocate_sample_takes
    #-------------------
    
    def _allocate_sample_takes(self, fsizes, rows_needed, rows_taken):
        '''
        Given a dict (fsizes), which maps {nth : num_of_rows},
        where nth is the index into a list of files from which
        to sample, and num_of_rows is the number of rows available
        in that file.
        
        Given also a number of needed samples of rows. Return a
        new dict {nth : num_to_take}, that contains the number
        of rows to select from the nth file.
        
        :param fsizes: dict mapping df-source index to number of
            rows in the df_source
        :type fsizes: dict[int : int]
        :param rows_needed: total number of rows to sample
        :type rows_needed: int
        :param rows_taken: dict mapping df-source index to set
            of row indexes already taken
        :type rows_taken: dict[str : set[int]]
        :return dict mapping df_source to number rows to take from
            that source
        :rtype {int : int}
        '''
        
        total_in_dfs  = sum(fsizes.values())
        already_taken = sum([len(taken_set) for taken_set in rows_taken.values()])
        total_avail   = total_in_dfs - already_taken
        
        if total_avail <= 0:
            raise ValueError(f"Asking for more samples than are available across all files.")
        # Percentage of samples needed relative to the total
        # number of available rows:
        percentage_needed = rows_needed / total_avail
        to_take = {i : int(np.ceil(percentage_needed * rows_in_file))
                   for i, rows_in_file
                   in fsizes.items()
                   }
        return to_take

    
# -------------------------- run_experiments --- the mains options    



# ---------------------------- Class MainActions ------------

class Activities:

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self,
                 dst_dir,
                 data_dir='/tmp',
                 res_file_prefix='perplexity_n_clusters_optimum'):
        
        self.log = LoggingService()
        self.dst_dir = dst_dir
        self.data_dir = data_dir
        self.res_file_prefix = res_file_prefix
        
    #------------------------------------
    # organize_results
    #-------------------
    
    def organize_results(self):
        '''
        Finds temporary files that hold PerplexitySearchResult
        exports, and those that contain plots made for those
        results. Moves all of them to self.dst_dir, under a 
        name that reflects their content. 
        
        For example:
        
                       perplexity_n_clusters_optimum_3gth9tp.json in /tmp
            may become 
                       perp_p100.0_n2_20240518T155827.json
            in self.dst__dir
            
        Plot figure png files, like perplexity_n_clusters_optimum_plot_20251104T204254.png
        will transfer unchanged.
        
        For each search result, the tsne_df will be replicated into a
        .csv file in self.dst_dir.
            
        '''
        
        for fname in self._find_srch_results():
            # Guard against 0-length files from aborted runs:
            if os.path.getsize(fname) == 0:
                self.log.warn(f"Empty hyperparm search result: {fname}")
                continue
            
            # Saved figures are just transfered:
            if fname.endswith('.png'):
                shutil.move(fname, self.dst_dir)
                continue
            
            srch_res = PerplexitySearchResult.read_json(fname)
            mod_time = self._modtimestamp(fname)
            perp = srch_res['optimal_perplexity']
            n_clusters = srch_res['optimal_n_clusters']
            
            dst_json_nm   = f"perp_p{perp}_n{n_clusters}_{mod_time}.json"
            dst_json_path = os.path.join(self.dst_dir, dst_json_nm)
            
            dst_csv_nm = f"{Path(dst_json_path).stem}.csv"
            dst_csv_path = Path(dst_json_path).parent.joinpath(dst_csv_nm)          
            
            src_path = os.path.join(self.data_dir, fname)
            shutil.move(src_path, dst_json_path)
            
            # Write the TSNE df to csv:
            tsne_df = srch_res['tsne_df']
            
            tsne_df.to_csv(dst_csv_path, index=False)
            
            #print(srch_res)
            print(dst_json_nm)

    #------------------------------------
    # hyper_parm_search
    #-------------------
    
    def hyper_parm_search(self, repeats=1, overwrite_previous=True):
        '''
        Runs through as many split files as specified in the repeats
        argument. For each split file, computes TSNE with each 
        perplexity (see hardcoded values in _run_hypersearch()). Then,
        with each TSNE embedding, runs multiple clusterings with varying
        n_clusters. Notes the silhouette coefficients. 
        
        Returns a PerplexitySearchResult with all the results.
        
        All search result summaries are saved in /tmp/perplexity_n_clusters_optimum*.json.
        The Action.ORGANIZE moves those files to a more permanent
        destination, under meaningful names. If this method is run
        multiple times without intermediate Action.ORGANIZE, then the 
        overwrite_previous arg controls whether users are warned about
        the intermediate files in /tmp being overwritten.
        
        :param repeats: number different split files on which to
            repeat the search
        :type repeats: int
        :param overwrite_previous: if True, silently overwrites previously
            saved hyper search results in /tmp. Else, asks permisson
        :type overwrite_previous: bool
        :return an object that contains all the TSNE results, and all
            the corresponding clusterings with different c_clusters.
        :rtype PerplexitySearchResult
        '''

        # Offer to remove old search results to avoid confusion.
        # Asks for OK. 
        # If return of False, user aborted.
        if not overwrite_previous and not self.remove_search_res_files():
            return
        
        src_results = []
        
        measurement_split_num = random.sample(range(0,10), repeats)
         
        for split_num in measurement_split_num:
            split_file = f"split{split_num}.feather"
            start_time = time.monotonic()
            
            with NamedTemporaryFile(dir=self.data_dir, 
                                    prefix=self.res_file_prefix,
                                    suffix='.json',
                                    delete=False
                                    ) as fd:
            
                srch_res = self._run_hypersearch(search_res_outfile=fd.name, split_file=split_file)
                
                stop_time = time.monotonic()
                duration = stop_time - start_time
                src_results.append(srch_res)
                print(f"Runtime measurement file split{split_num}.feather: {int(duration)} seconds")
                print(f"... {int(duration / 60)} minutes")
                print(f"...{duration / 3600} hours")
        return src_results
    
    #------------------------------------
    # remove_search_res_files
    #-------------------
    
    def remove_search_res_files(self):

        # Check whether any hyper search results even exist:
        fnames = self._find_srch_results()
        if len(fnames) == 0:
            # Nothing to do
            return True

        # There are search result files to remove; double check with user:        
        resp = input("Remove previous search results? (Yes/no): ")
        if resp != 'Yes':
            print('Aborting, nothing done.')
            return False

        for srch_res in fnames:
            self.log.info(f"Removing {srch_res}")
            os.remove(srch_res)
            
        return True

    #------------------------------------
    # _run_hypersearch
    #-------------------
    
    def _run_hypersearch(self,
                         chirp_id_src=ChirpIdSrc('file_id', ['chirp_idx']),
                        search_res_outfile=None,
                        split_file='split5.feather'
                        ):
        '''
        Run different data analyses. Comment out what you
        don't want. Maybe eventually make into a command
        line interface utility with CLI args.
        
        The search_res_outfile may be provided if the hyperparameter
        search result should be saved as JSON. It can be recovered
        via PerplexitySearchResult.read_json(), though in abbreviated
        form. The value of this arg may be a file path string, a 
        file-like object, like an open file descriptor, or None. If None, no output.
        
        A PerplexitySearchResult object with all results is returned.
        
        :param chirp_id_src: name of columns in SonoBat measures
            split files that contain the file id and any other
            columns in the measurements df that together uniquely identify
            each chirp
        :type key_col: ChirpIdSrc
        :param search_res_outfile: if provided, save the hyperparameter search
            as JSON in the specified file
        :type search_res_outfile: union[str | file-like | None]
        :parm split_file: name of split file to use as measurements source.
            File is just the file name, relative to the measures root.
        :type split_file:
        :returned all results packaged in a PerplexitySearchResult
        :rtype PerplexitySearchResult
        '''
    
        outfile        = '/tmp/cluster_perplexity_matrix.csv'
        #measures_root  = '/Users/paepcke/quintus/home/vdesai/bats_data/new_dataset/splits'
        measures_root = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/Clustering'
        inference_root = '/Users/paepcke/quintus/home/vdesai/bats_data/inference_files/model_outputs'
    
        if Utils.is_file_like(search_res_outfile):
            res_outfile = search_res_outfile.name 
        elif type(search_res_outfile) == str:
            # Make sure we can write there:
            try:
                with open(search_res_outfile, 'w') as fd:
                    fd.write('foo')
                os.remove(search_res_outfile)
            except Exception:
                raise ValueError(f"Cannot write to {search_res_outfile}")
            # We'll be able to write the result:
            res_outfile = search_res_outfile
        else:
            res_outfile = None
    
        path = os.path.join(measures_root, split_file)
    
        calc = DataCalcs(measures_root, inference_root, chirp_id_src=chirp_id_src)
    
        with UniversalFd(path, 'r') as fd:
            calc.df = fd.asdf()
            
        # Use the .wav file information in file_id column to  
        # obtain each chirp's recording date and time. The new
        # column will be called 'rec_datetime', and an additional
        # column: 'is_daytime' will be added. This is done inplace,
        # so no back-assignment is needed:
        calc.add_recording_datetime(calc.df) 
        
        # Find best self.optimal_perplexity, self.optimal_n_clusters:
        # Perplexity for small datasets should be small:
        if len(calc.df) < 100:
            perplexities = [5.0, 10.0, 20.0, 30.0]
        elif len(calc.df) < 2000:
            perplexities = [30.0, 40.0, 50.0]
        else:
            perplexities = [40.0, 50.0, 60.0, 70.0, 100.0]
    
        print(f"Will try perplexities {perplexities}...")
        
        # The columns of the measurements file to retain in
        # the final search result object's TSNE df: the measurements
        # with high variance (DataCalcs.sorted_mnames), plus the
        # composite chirp key, file_id, chirp_idx:
        (important_cols := DataCalcs.sorted_mnames.copy()).extend(['file_id', 'chirp_idx', 'rec_datetime', 'is_daytime'])
        hyperparms_search_res = calc.find_optimal_tsne_clustering(
            calc.df, 
            perplexities=perplexities,
            n_clusters_to_try=list(range(2,10)),
            cols_to_keep=important_cols,
            outfile=outfile
            )
        print(f"Optimal perplexity: {hyperparms_search_res.optimal_perplexity}; Optimal n_clusters: {hyperparms_search_res.optimal_n_clusters}")
        
        if res_outfile is not None:
            print(f"Saving hyperparameter search to {res_outfile}...")
            hyperparms_search_res.to_json(res_outfile)
            
        return hyperparms_search_res
        #viz.plot_perplexities_grid([5.0,10.0,20.0,30.0,50.0], show_plot=True)

    #------------------------------------
    # _find_srch_results
    #-------------------
    
    def _find_srch_results(self):
        '''
        Find the full paths of search results that have 
        been saved in the data_dir directory. Done by
        finding file names that start with self.res_file_prefix

        :return all search result file paths
        :rtype list[str]
        '''
        fnames = filter(lambda fname: fname.startswith(self.res_file_prefix),
                        os.listdir(self.data_dir))
        full_paths = [os.path.join(self.data_dir, fname)
                      for fname 
                      in fnames]
        return full_paths

    #------------------------------------
    # _modtimestamp
    #-------------------
    
    def _modtimestamp(self, fname):
        
        # Get float formatted file modification time,
        # and turn into int:
        mod_timestamp = int(os.path.getmtime(fname))
        # Make into a datetime object:
        moddt = datetime.fromtimestamp(mod_timestamp)
        # Get a datetime ISO formatted str, and remove
        # the dash and colon chars to get like
        #   '20240416T152351' 
        stamp = Utils.timestamp_fname_safe(time=moddt)
        return stamp

    #------------------------------------
    # _plot_search_results
    #-------------------
    
    def _plot_search_results(self, search_results):
    
        # The following will be a list: 
        # [perplexity1, ClusteringResult (kmeans run: 8) at 0x13f197b90), 
        #  perplexity2, ClusteringResult (kmeans run: 8) at 0x13fc6c2c0),
        #            ...
        #  ]
        cluster_results = []
        
        # As filepath for saving the figure at the end,
        # use the file prefix self.res_file_prefix, and
        # the current date and time:
        filename_safe_dt = Utils.timestamp_fname_safe()
        fig_save_fname   = f"{self.res_file_prefix}_plots_{filename_safe_dt}.png"
        fig_save_path    = os.path.join(self.data_dir, fig_save_fname) 
        
        # Collect all the cluster results from all search result objs
        # into one list:
        for srch_res in search_results:
            cluster_results.extend(list(srch_res.iter_cluster_results()))        

        plot_contents = []
        for perplexity, cluster_res in cluster_results:
            n_clusters = cluster_res.best_n_clusters
            silhouette = round(cluster_res.best_silhouette, 2)
            plot_contents.append({
                'tsne_df'         : cluster_res.tsne_df,
                'cluster_labels'  : cluster_res.best_kmeans.labels_,
                'title'           : f"Perplexity: {perplexity}; n_clusters: {n_clusters}; silhouette: {silhouette:{4}.{2}}"
                })
        fig = DataViz.plot_perplexities_grid(plot_contents)
        self.log.info(f"Saving plots to {fig_save_path}")
        fig.savefig(fig_save_path)
        fig.show()
        input("Any key to erase figs and continue: ")

    #------------------------------------
    # main
    #-------------------
    
    def main(self, actions, **kwargs):
        '''
        Perform one task using class DataCalcs methods.
        Keyword arguments are passed to the executing 
        methods, if appropriate.

        :param actions: one or more Actions to perform. If
            multiple actions are specified, they are executed
            in order.
        :type actions: union[Action | list[Action]]
        '''

        if type(actions) != list:
            actions = [actions]
            
        for action in actions:
            if action == Action.ORGANIZE:
                self.organize_results()
                
            elif action == Action.HYPER_SEARCH:
                srch_results= self.hyper_parm_search(repeats=1)
                
            elif action == Action.CLEAR_RESULTS:
                self.remove_search_res_files()
                
            elif action == Action.PLOT:
                self._plot_search_results(srch_results)
                
            elif action == Action.SAMPLE_CHIRPS:
                # Possible kwarg: num_chirps, which is
                # the number of chirp
                self._sample_chirps(kwargs)
                
            else:
                raise ValueError(f"Unknown action: {action}")
            
            
# ------------------------ Main ------------
if __name__ == '__main__':

    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    dst_dir = os.path.join(proj_dir, 'results/hyperparm_searches')

    actions = [Action.HYPER_SEARCH, Action.PLOT]
    #*******actions  = Action.HYPER_SEARCH
    #*******actions  = Action.ORGANIZE
        
    activities = Activities(dst_dir)
    activities.main(actions)
