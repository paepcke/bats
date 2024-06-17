'''
Created on Apr 27, 2024

@author: paepcke
'''

from collections import (
    namedtuple)
from data_calcs.universal_fd import (
    UniversalFd)
from data_calcs.utils import (
    Utils,
    TimeGranularity)
from datetime import (
    datetime)
from enum import (
    Enum)
from io import (
    StringIO)
from itertools import (
    chain,
    accumulate)
from logging_service.logging_service import (
    LoggingService)
from pathlib import (
    Path)
from scipy import (
    stats)
from sklearn.cluster import (
    KMeans)
from sklearn.decomposition import (
    PCA)
from sklearn.manifold import (
    TSNE)
from sklearn.metrics import (
    silhouette_score)
from sklearn.metrics.pairwise import (
    pairwise_distances,
    paired_distances)
import csv
import joblib
import json
import numpy as np
import os
import pandas as pd
import random
import re
from torch.jit import isinstance
from tifffile.tifffile import FILETYPE

class Localization:
    # There is also a .csv version of the following .feather file:
    #all_measures   = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/AnalysisReady/concat_10__chirps_20240527T100032.314015.feather'
    proj_dir       = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    # Results and visualizations:
    proj_records   = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/' 
    #analysis_dst   = os.path.join(proj_dir, 'results/chirp_analysis/PCA_AllData')
    analysis_dst   = os.path.join(proj_dir, 'results/chirp_analysis/Classifications')
    sampling_dst   = os.path.join(proj_dir, 'results/chirp_samples')
    srch_res_dst   = os.path.join(proj_dir, 'results/hyperparm_searches')

    # All measures, but transformed via PCA:    
    pca_xformed    = os.path.join(
        proj_dir,
        'results/chirp_analysis/Classifications/PCA23Components_all_but_is_daytime/xformed2024-06-10T16_44_54_23components_297476samples.feather' 
        )
    
    all_measures   = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/AnalysisReady/scaled_chirps_2024-06-17T08_43_17.feather'
    all_measures_descaled = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/AnalysisReady/concat_10__chirps_orig_2024-05-27T10_00_32.feather'
    scaler         = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/Descaling/split_scaler_1.5.0.joblib'
    #measures_root  = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/Clustering'
    measures_root  = '/Users/paepcke/quatro/home/vdesai/data/training_data/all/splits'
    inference_root = '/Users/paepcke/quintus/home/vdesai/bats_data/inference_files/model_outputs'


# ----------------------------- Enum Action ---------------


class FileType(Enum):
    FEATHER       = '.feather'
    CSV           = '.csv'
    CSV_GZ        = '.csv.gz'
    
# ---------------------------------- Enum FileType ---------------

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
                 measures_root=None, 
                 inference_root=None, 
                 chirp_id_src=ChirpIdSrc('file_id', ['chirp_idx']),
                 cols_to_retain=None,
                 fid_map_file=None
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
        :param fid_map_file: path to file that maps bat file IDs to .wav file names.
            if None, the default: measures_root/split_filename_to_id.csv
        :type fid_map_file: union[None | str]
        '''

        if measures_root is None:
            measures_root = Localization.measures_root
        if inference_root is None:
            inference_root = Localization.inference_root  # The Global one
            
        if fid_map_file is None:
            self.fid_map_file = 'split_filename_to_id.csv'
        else:
            self.fid_map_file = fid_map_file

        self.measures_root  = measures_root  # Either the arg, or the global one
        self.inference_root = inference_root # Either the arg, or the global one
        self.chirp_id_src   = chirp_id_src
        self.cols_to_retain = cols_to_retain
        
        self.log = LoggingService()
        
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

        # If measure_root is None, we cannot look for
        # mappings of split file IDs to file names. So
        # some functionality won't work until caller calls
        # the initialization themselves:
        
        if measures_root is not None:
            self.init_split_file_list(measures_root, inference_root)
        
        # Create self.fid2split_dict for mapping a split file id
        # to the split file that contains the data for that
        # file id:
        
        #self._make_fid2split_dict()
        
    #------------------------------------
    # init_split_file_list
    #-------------------
    
    def init_split_file_list(self, measures_root, inference_root):
        '''
        Finds files of the form split<n>.feather in measures_root,
        and initializes self. split_fpaths, which is a list of all
        those split measures files.
        
        Must be called before hyper parameter search related functionality.
        
        :param measures_root: root of measures split files
        :type measures_root: list[str]
        :param inference_root: root of transformer inference results
        :type inference_root: union[None, list[str]]
        '''
        
        # Read the mapping of .wav file to file_id values in 
        # the measurements file_id column. The header
        # is ['Filename', 'file_id']
        map_file = os.path.join(measures_root, self.fid_map_file)
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
        Modifies df:
        
            o Adds column 'rec_datetime', which is the recording
              date and time of the chirp recording. That time is
              extracted from the recording's .wav file name. 

            o Adds a new boolean column 'is_daytime' that indicates
              whether the chirp was recorded during daytime.
              
        These changes occur in place.  
        
        :param df: dataframe to modify
        :type df: pd.DataFrame
        :return the modified df
        :rtype pd.DataFrame
        '''
        # Take the .wav file names, extract 
        # the recording datetime. The column
        # that names the .wav file of each row (i.e. of 
        # each chirp) contains integer file IDs for the
        # .wav files. First, resolve file_id to a .wav
        # filename:
        
        # Name of column with .wav file ID ints:
        wav_file_col_nm = self.chirp_id_src.wav_file_nm_col 
        # Resolve .wav file names:
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
        
        # Using the .wav file names again, extract the 
        # bat species series from them, if SonoBat made them
        # available. The cases are:
        #     barn1_D20220207T215546m654-Laci-Tabr.wav
        #     barn1_D20220207T214358m129-Coto.wav
        #     barn1_D20220720T020517m043.wav
        species_lists = list(map(lambda fname: Utils.extract_species_from_wav_filename(fname),
                                 fnames))
        species_lists_series = pd.Series(species_lists, name='species')
        # Add recording times column:
        df['species'] = species_lists_series

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
        of measurement result columns. That is the num_dims'th ranked 
        variance measures are used for each chirp. 
        
        Uses the min(num_points, dfsize) (i.e. num_points rows) from the df.
        
        After running, the number of points will be in:
        
                self.effective_num_points
        
        Tsne is run over the resulting num_points x num_dims dataframe.
        
        Returned is a dataframe with num_points rows, and the following
        columns:
        
              'tsne_x'    'tsne_y'   [original values of the cols_to_keep columns]
        
        The index of the returned dataframe will be the same as the 
        index passed in, except for the redacted rows.
        
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
            df = df_all_rows.iloc[0:num_points, :]
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

        self.effective_num_points = len(df)
        log_msg = (f"Running tsne; perplexity '{perplexity}';"
                   f"num_dims: {num_dims};"
                   f"num_points: {self.effective_num_points}")
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
        
        # If we are asked to sort by variance of SonoBat measures, then
        # Only use columns for TSNE computation that 
        # are both, in the measures, and are wanted
        # for carry-over:
        
        if sort_by_bat_variance:
            cols_to_use = list(set(self.sorted_mnames).intersection(set(df.columns)))
            if len(cols_to_use) == 0:
                raise ValueError(f"None of the columns in given df are measure names ({df.columns})")
        else:
            cols_to_use = df.columns
        
        embedding_arr = tsne_obj.fit_transform(df.loc[:, cols_to_use])
        
        # For each embedding point, add the cols_to_keep, after
        # turning the np.array TSNE result into a df:
        tsne_df_abbridged = pd.DataFrame(embedding_arr, columns=['tsne_x', 'tsne_y'], index=df.index)
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
    # find_optimal_k
    #-------------------
    
    @staticmethod
    def find_optimal_k(X_df, k_range):
        sil_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=1066)
            sil_scores.append([k, 
                               silhouette_score(X_df, kmeans.fit_predict(X_df)),
                               kmeans
                               ])
        return pd.DataFrame(sil_scores, columns=['k', 'silhouette_score', 'kmeans'])

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
    # correlate_all_against_one
    #-------------------

    @staticmethod
    def correlate_all_against_one(df, target_col):
        '''
        Calculates the correlation of each variable in a DataFrame 
        with one other. The df is of dimension n_samples x n_features
    
        :param df: The DataFrame containing the data.
        :type df: pd.DataFrame
        :param target_col: The column name to which correlations are computed
        :type target_col: str
        :return Series with Pearson correlation of each 
            original variable against the target. The Series
            index will be the measure names.
        :rtype pd.Series
        '''

        target_ser = df[target_col].copy()       
        # Exclude the target column
        filtered_df = df.drop(target_col, axis=1)
          

        # Is the target column dichotomous?
        uniq_target_vals = df[target_col].unique() 
        if len(uniq_target_vals) == 2:
            # Values must be 1 and 0. Make a mapping, just
            # in case. It's a pain to ensure that the values
            # aren't True or False, which are == to 1 and 0,
            # respectively, 
            val0, val1 = uniq_target_vals
            if type(val0) != int or val0 not in [0,1] or \
               type(val1) != int or val1 not in [0,1]:
                val_map = {val0 : 0, val1 : 1}
                target_ser = target_ser.replace(val_map) 
            
            # Result from one pointbiserial corr computation:
            # A correlation coefficient, and an associated p_value:
            # Dichotomous, use pointserial correlation variant:
            def point_serial_one_var(continuous_vals):
                point_biserial_r, p_value = stats.pointbiserialr(continuous_vals, target_ser)
                #****return pt_ser_res(point_biserial_r, p_value)
                return pd.Series([point_biserial_r, p_value], index=[f"Corr_all_against_{target_col}", 
                                                                     'p_value'], 
                                 name=continuous_vals.name)
            
            # Get like the following:
            #                        TimeInFile  PrecedingIntrvl  CallsPerSec  chirp_idx
            #    point_biserial_r         0.0              0.0          0.0   0.944911
            #    p_value                  1.0              1.0          1.0   0.212296
            #
            # and transpose it to get like:
            #                     point_biserial_r   p_value
            #    TimeInFile               0.000000  1.000000
            #    PrecedingIntrvl          0.000000  1.000000
            #    CallsPerSec              0.000000  1.000000
            #    chirp_idx                0.944911  0.212296

            res_df = filtered_df.apply(point_serial_one_var, axis='rows').T

        else:
            # Target is not dichotomous:
            # Calculate correlation matrix (excluding target column itself)
            corr_ser = filtered_df.corrwith(target_ser, axis='rows')
            corr_ser.name = f"Corr_all_against_{target_col}"
            nans = pd.Series([np.nan] * len(corr_ser), index=corr_ser.index, name='p_value')
            res_df = pd.concat([corr_ser, nans], axis='columns')
        return res_df

    #------------------------------------
    # distances
    #-------------------
    
    @classmethod
    def distances(cls, obj1, obj2, metric='euclidean'):
        '''
        Returns the distance(s) between obj1 and obj2.
        These args may be DataFrames of dimension 
        (n_samples, n_features), or Series of length 
        n_samples. When obj1 and obj2 have different
        types, the dimensions must match. Cases:

              obj1     obj2       res 
             Series   Series     Series of one element
             Series   Df         Series of as many elements as df has rows
                                   i.e distances are measured between the
                                   Series and each of the df's rows.
                                   
            Df        Df         Dataframe with distances of corresponding 
                                   rows
            same with reversed  
        
        
        
        :param obj1: origin
        :type obj1: union[pd.Series, pd.DataFrame]
        :param obj2: destination
        :type obj2: union[pd.Series, pd.DataFrame]
        :returns pairwise distances
        :rtype pd.Series
        '''
        if isinstance(obj1, pd.Series):
            if isinstance(obj2, pd.Series):
                # Series ==== Series
                # distances between (X[0], Y[0]), (X[1], Y[1]), etc…:
                # Need numpy and shapes corresponding to 
                # 1 sample, multiple features:
                obj1np = obj1.to_numpy().reshape(1,-1)
                obj2np = obj2.to_numpy().reshape(1,-1)
                res = paired_distances(obj1np, obj2np, metric=metric)
                res_ser = pd.Series(res)
                return res_ser
            elif isinstance(obj2, pd.DataFrame):
                # Series ==== DataFrame
                # distances between obj1 and each row of obj2:
                obj1np = obj1.to_numpy().reshape(1,-1)
                res = pairwise_distances(obj1np, obj2, metric=metric)
                res_ser = pd.Series(res[0])
                return res_ser
            else:
                raise TypeError(f"Given obj1 of type pd.Series, obj2 must be pd.Series, or pd.DataFrame, not {obj2}")         
        elif isinstance(obj1, pd.DataFrame):
            if isinstance(obj2, pd.Series):
                # Obj1 ==== DataFrame
                # obj2 == Series
                obj2np = obj2.to_numpy().reshape(1,-1)
                res = pairwise_distances(obj1, obj2np, metric=metric)
                # Got like:
                # array([[ 0.        ],
                #        [17.32050808]])
                # Flatten the array into a Series
                res_ser = pd.Series(res.ravel())
                return res_ser
            elif isinstance(obj2, pd.DataFrame):
                # Obj1 ==== DataFrame
                # Obj2 ==== DataFrame
                res = pairwise_distances(obj1, obj2, metric=metric)
                res_flat = np.apply_along_axis(lambda res_pair: res_pair[0], 1, res)
                res_ser = pd.Series(res_flat)
                return res_ser
        return res
    
    #------------------------------------
    # make_chirp_sample_file
    #-------------------
    
    def make_chirp_sample_file(self, num_chirps, save_dir=None, prefix=None, unittests=None):
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
        
                  <save_dir>/<prefix>_<num-chirps>_chirps_<timestamp>.csv
         
                
        :param num_chirps: total number of chirps to sample across 
            all split files.
        :type num_chirps: int
        :param save_dir: if provided, save resulting df to
            <save_dir>/<self.res_file_prefix>_chirps_<timestamp>.csv
        :type save_dir: union[None | src]
        :param prefix: prefix to use on the result file name
        :type prefix: union[None | str]
        :param unittests: whether or not to save the resulting df
        :type unittests: union[None | list[pd.DataFrame]]
        :return dict: 'df' : the constructed df. 'out_file' : file where df was saved
        :rtype dict[str : union[pd.DataFrame | str | None]
        '''

        if prefix is None:
            prefix = ''
        # Before investing time, check whether caller wants
        # to save the resulting df. If so, see whether we
        # succeed in finding, or creating the target directory:
        if save_dir is not None:
            if os.path.exists(save_dir):
                # Ensure it's a dir:
                if not os.path.isdir(save_dir):
                    raise ValueError(f"A file with name {save_dir} exists; so cannot create a directory of that name")
            else:
                # Try to create the dir:
                os.makedirs(save_dir, exist_ok=True)
            save_fname = os.path.join(save_dir, f"{prefix}_{num_chirps}_chirps_{Utils.file_timestamp()}.csv")
        else:
            save_fname = None

        # Number of rows in each df:
        fsizes = {}

        # Make copy of the split file path list,
        # because we may need to go through the
        # loop more than once, and want to randomize
        # the list on rounds 2-plus:
        if type(unittests) == dict:
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
            for ith_src, measure_src in sample_paths.items():
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
        df_raw = pd.DataFrame(res_np_arr, columns=res_cols)
        # Replace file_id with recording datetime object, and
        # add whether recording was daylight or not, as well
        # as species:
        df_with_rectime = self.add_recording_datetime(df_raw)
        # Add sin and cos columns for each granularity of rec_datetime:
        df = self._add_trig_cols(df_with_rectime, 'rec_datetime')  
        df.reset_index(drop=True, inplace=True)
        
        if save_fname is not None:
            df.to_csv(save_fname, index=None)
        
        return {'df' : df,
                'out_fname' : save_fname}

    #------------------------------------
    # conditional_samples
    #-------------------
    
    @staticmethod
    def conditional_samples(df, 
                            num_samples, 
                            cond=None, 
                            save_dir=None, 
                            prefix=None, 
                            timestamp=None, 
                            save_format=FileType.CSV):
        '''
        Takes a dataframe and a number of samples. Returns a
        subset of the df with num_samples rows. 
        
        If cond is provided, it must be a function that, when 
        given a row, returns True if the row should be considered
        for sampling. Else it must return False. 
        
        If save_dir is provided, a filename is created of the form:
        
           {prefix}{num_samples}_of_{population_size}_samples_{timestamp}{extension}
           
        where population_size is the number of rows left after 
        rows are eliminated that do not satisfy cond. The file 
        extension is automatically derived from the save_format enum 
        member value.
        
        The timestamp is used if not None; else current datetime is used.
        
        It is a ValueError to ask for more samples than the available
        qualified rows.
        
        :param df: dataframe from which to sample
        :type df: pd.DataFrame
        :param num_samples: number of samples without replacement
        :type num_samples: int
        :param cond: function that decides whether or not to include a given row
        :type cond: optional[func]
        :param save_dir: optional directory where to save the resulting df
        :type save_dir: optional[str]
        :param prefix: optional string to place a start of filename if saving
        :type prefix: optional[str]
        :param timestamp: optional timestamp string to use in filename
        :type timestamp: optional[str]
        :param save_format: whether to save as .csv, .csv.gz, or .feather
        :type save_format: FileType
        :return: a dataframe with num_samples rows that all satisfy cond
        :rtype: pd.DataFrame
        :raise ValueError: if requested num_samples exceeds number of
            qualifying rows.
        '''
    
        if cond is not None:
            # Get a subset of the df that fulfills the condition:
            df_excerpt = df.loc[cond]
        else:
            df_excerpt = df
        
        population_size = len(df_excerpt)
        if num_samples > population_size:
            raise ValueError(f"Dataframe only has {population_size} samples, but {num_samples} were requested")
        if num_samples == population_size:
            return df_excerpt
        
        sample_row_nums = random.sample(range(population_size), num_samples)
        res = df_excerpt.iloc[sample_row_nums]
        
        if save_dir is not None:
            # Create a file name that includes num of samples, population
            # size (after applying condition), and timestamp:
            if timestamp is None:
                timestamp = Utils.file_timestamp()
            fname = f"{prefix}{num_samples}_of_{population_size}_samples_{timestamp}{save_format.value}"
            full_path = os.path.join(save_dir, fname)
            with UniversalFd(full_path, 'w') as fd:
                fd.write_df(res)
        
        return res

    #------------------------------------
    # pca_computation
    #-------------------
    
    def pca_computation(self, df, n_components=None, columns=None, dst_dir=None):
        '''
        Given a dataframe, return an sklearn.PCA object that is fitted
        to df, but df has not been transformed into the PCA component space.
        The call must issue pca.transform() on the returned df to map df
        into the PCA component space.
                    
        If columns is provided it must be a list of column names in the df. Only
        those columns are included as input to the PCA. If columns is
        None, then all of df's columns are included.
        
        n_components may be one of the following:
        
            o Int   : specifies the exact number of components desired
            o Float : number between 0 and 1 indicates the desired percentage
                      of variance explained by the PCA result. The number
                      of components is automatically chosen accordingly.
            o 'mle' : Minka's algorithm is use to select the best number
                      of components. It is often better than the scree plot
                      method (finding the elbow), or using a desired explained
                      variance ratio threshold and cumulatively summing elements
                      of the PCA's explained_variance_ratio_ list of variance
                      explained by each component.
                      
        Returns a dict with keys 'pcs', 'weight_matrix', 'xformed_data'. The 
        first is the sklearn.PCA object. The second is a dataframe whose columns a
        names, and the row index is 'component1', 'component2', ... The rows 
        re the measure are the weights assigned to each original feature.
        
        The PCA instance will have an additional attribute: create_date with
        the datetime object of the object's creation time.
        
        If dst_dir is not None, the PCA object will be saved in 
            
            pca_<cur-time>.joblib
        
        :param df: dataframe from which to construct principal components 
        :type df: pd.DataFrame
        :param n_components: target number of principal components. If None,
            constructs as many components as there are columns
        :type n_components: union[None | int)
        :param columns: columns to include from the df. If None: all columns
        :type columns: union[None, list[str]]
        :param dst_dir: if dst_dir is not None, it must be a full path
            to where the result PCA is to be stored. The format will be joblib,
            and the extension will be .joblib.
        :type dst_dir: optional[str]
        :return a PCA model, weight matrix, and transformed data
        :rtype dict[str : union[sklearn.PCA, pd.DataFrame]xs
        
        '''
        self.log.info(f"Running PCA with target of {n_components} components")
        pca = PCA(n_components=n_components)
        
        if columns is None:
            df_pca = df 
        else:
            df_pca = df[columns]
            
        # fit() returns the pca instance itself:
        pca = pca.fit(df_pca)

        self.log.info("Done.")
        
        # Add attribute create_date with the current
        # date and time as a datetime object:
        pca.create_date = datetime.now() 

        # Transform the data into the component space:
        self.log.info(f"Transforming original data into {n_components} components")
        xformed_np = pca.fit_transform(df_pca)
        self.log.info("Done.")

        # Columns of transformed data will be 'comp<i>': 
        col_names = [f"comp{i}" for i in range(pca.n_components_)]
        # Transform the original data into the component space:
        xformed_df = pd.DataFrame(xformed_np, columns=col_names)
        
        # Matrix components x features with each feature's weight in 
        # each of the components (rows0;
        weight_df = pd.DataFrame(pca.components_, columns=df_pca.columns)
        weight_df.index.name = 'component_num'
        
        if dst_dir is not None:
            dt_str = Utils.timestamp_from_datetime(pca.create_date)
            # The dump() method will append an appropriate file suffix
            dst_fname_root = os.path.join(dst_dir, f"pca_{dt_str}")
            # Dump returns the final save file name:
            dst_fname = self.dump_pca(pca, dst_fname_root)
        
        return {'pca' : pca, 
                'weight_matrix' : weight_df,
                'xformed_data'  : xformed_df,
                'pca_save_file' : dst_fname
                }
        
    #------------------------------------
    # pca_needed_dims
    #-------------------
    
    def pca_needed_dims(self, df, variance_threshold, columns=None):
        '''
        Given a fitted PCA object, return the number 
        components needed to explain variance_threshold
        percent of the total variance of the dataframe
        provided to the pca_computation().
        
        The variance_threshold may either be:
         
                    1.0 < variance_threshold <= 100
        
        or:
                    0 < variance_threshold <= 1
                    
        In either case, the number is used as a percentage, transforming
        to 0...1.0 as needed.
                
        The algorithm is to find 
        
        An alternative is to pass n_components='mle' to the pca_computation()
        method to have an optimal dimensionality chosen.
        
        :param df: data for which PCA is to be performed.
        :type df: pd.DataFrame
        :param variance_threshold: least amount of variance that
            the combined components need to explain in percent.
        :type variance_threshold: union[int, float]
        :param columns: columns to use from the df
        :type columns: union[None, list[str]
        :return the number of dimensions required to reach 
            variance_threshold percent explanation of variance
        :rtype int
        '''
        
        if variance_threshold > 1.0:
            variance_threshold = variance_threshold / 100.
            
        pca_all = self.pca_computation(df, n_components=None, columns=columns)
        for component_idx, var_explained in enumerate(accumulate(pca_all.explained_variance_ratio_)):
            if var_explained >= variance_threshold:
                return component_idx
        return component_idx

    #------------------------------------
    # dump_pca
    #-------------------
    
    @staticmethod
    def dump_pca(pca, dst_fname, force=False):
        '''
        Save the given sklearn.PCA object in dst_fname.
        If force is True, then dst_fname is used even if
        it would overwrite an existing file. Else user is
        asked for confirmation on the console.
        
        If the file suffix of dst_fname is not '.joblib', that
        suffix is added.
        
        Storage format is joblib.dump().  
        
        :param pca: the PCA to save
        :type pca: sklearn.PCA
        :param dst_fname: full path to where PCA is to be saved.
        :type dst_fname: src
        :param force: whether or not to overwrite dst_fname if a
            file of that path already exists.
        :type force: bool
        :return the full path to the stored object
        :rtype str
        '''
        fpath = Path(dst_fname)
        if fpath.is_dir():
            raise ValueError(f"PCA save destination {dst_fname} is a directory; must be a file")
        if force and fpath.exists():
            if input(f"File {dst_fname} exists; overwrite? (Yes/no") != 'Yes':
                return
        if fpath.suffix != '.joblib':
            fpath = Path(f"{fpath}.joblib")
        path_str = str(fpath)
        joblib.dump(pca, path_str)
        return path_str

    #------------------------------------
    # load_pca
    #-------------------
    
    @staticmethod
    def load_pca(src_fname):
        '''
        Returns an sklearn.PCA object that was previously 
        saved in src_fname by self.dump_pca()
        
        To use the retrieved PCA to transform new data:
           
           transformed_data = loaded_pca.transform(new_data)
        
        :param src_fname: location of the saved PCA object
        :type src_fname: str
        :return the retrieved PCA instance. 
        :rtype sklearn.PCA
        '''
        fpath = Path(src_fname)
        if not fpath.exists():
            raise FileNotFoundError(f"PCA file {src_fname} not found")
        pca = joblib.load(str(fpath))
        return pca

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
        # needed six trigonometric columns. We use the cache
        # for that:
        trig_rows = list(map(self._sin_cos_cache, df[dt_col_nm]))
        
        new_df = pd.concat([df, pd.DataFrame(trig_rows, columns=cols)], axis='columns')
        
        return new_df


    #------------------------------------
    # _sin_cos_cache
    #-------------------
    
    def _sin_cos_cache(self, dt):
        '''
        Given a datetime, return a tuple of
        eight elements: the sin and cos for each
        granularity of HOURS, DAYS, MONTHS, and YEARS.
         
        :param dt: a datetime instance
        :type dt: datetime.datetime
        :return sin and cos mappings for each granularity 
            of HOURS, DAYS, MONTHS, and YEARS.
        :rtype tuple[float]
        '''
        
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


