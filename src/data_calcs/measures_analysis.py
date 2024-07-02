'''
Created on May 28, 2024

@author: paepcke
'''
from data_calcs.classification_result import (
    ClassifictionResult)
from data_calcs.data_calculations import (
    DataCalcs,
    ChirpIdSrc,
    FileType,
    Localization,
    PerplexitySearchResult)
from data_calcs.data_cleaning import (
    DataCleaner)
from data_calcs.data_viz import (
    DataViz)
from data_calcs.universal_fd import (
    UniversalFd)
from data_calcs.utils import (
    Utils)
from datetime import (
    datetime)
from enum import (
    Enum)
from imblearn.over_sampling import (
    SMOTE)
from logging_service.logging_service import (
    LoggingService)
from matplotlib import (
    pyplot as plt)
from matplotlib.pyplot import (
    clf)
from pathlib import (
    Path)
from sklearn.decomposition import (
    PCA)
from sklearn.ensemble import (
    RandomForestClassifier)
from sklearn.linear_model import (
    LogisticRegression)
from sklearn.metrics._plot.confusion_matrix import (
    ConfusionMatrixDisplay)
from sklearn.metrics._plot.precision_recall_curve import (
    PrecisionRecallDisplay)
from sklearn.model_selection import (
    StratifiedKFold)
from sklearn.preprocessing import (
    LabelEncoder)
from tempfile import (
    NamedTemporaryFile)
import itertools
import numpy as np
import os
import pandas as pd
import random
import shutil
import sys
import time

# Members of the Action enum are passed to run
# to specify which task the program is to perform: 
class Action(Enum):
    HYPER_SEARCH        = 0
    PLOT_SEARCH_RES     = 1
    ORGANIZE            = 2
    CLEAR_RESULTS       = 3
    SAMPLE_CHIRPS       = 4
    EXTRACT_COL         = 5
    CONCAT              = 6
    PCA                 = 7
    PCA_ANALYSIS        = 8
    PRINT_CLF_RESULTS   = 9
    CHIRP_SIMILARITY    = 10

# Options for how the SMOTE algorithm fixes
# class imbalance during classification:
class SMOTEOption(Enum):
    MINORITY     = 'minority'
    NOT_MINORITY = 'not minority'
    NOT_MAJORITY = 'not majority'
    ALL          = 'all'
    AUTO         = 'auto'

        
# ----------------------------- Class MeasuresAnalysis ---------

class MeasuresAnalysis:
            
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, action, **kwargs):
        '''
        Initializes various constants. Then, if action is 
        one of the Action enum members, executes the requested
        experiment.
        
        I action is None, just initializes instance vars, 
        and returns.
        
        Each experiment generates a dictionary with relevant
        results. These results are stored in the MeasuresAnalysis
        experiment_result attribute. The action that identifies
        the experiment done is in the action attribute. 
        
        For each type of experiment there may be required and/or
        optional keyword arguments. Provide them all as kwargs,
        though the args without a default below are mandatory:
        
            PCA:              [n_components]
                returns       {'pca' : pca, 
                               'weight_matrix'      : weight_matrix, 
                               'xformed_data'       : xformed_data,
                               'pca_file'           : pca_dst_fname,
                               'weights_file'       : weight_fname,
                                  'xformed_data_fname' : xformed_data_fname                
                               }
                               
            PCA_ANALYSIS       [pca_info]
            
            HYPER_SEARCH      repeats=1, 
                              overwrite_previous=True
                returns       PerplexitySearchResult instance
                              
            PLOT_SEARCH_RES   search_results  # result from prior HYPER_SEARCH
                              data_dir
                returns       <nothing>
            
            SAMPLE_CHIRPS     num_samples=None, 
                              save_dir=None
                returns       {'df' : the constructed df,
                               'out_file' : to where the df was written}
                              
            CONCAT            df_sources,
                              idx_columns=None, 
                              dst_dir=None, 
                              out_file_type=FileType.CSV, 
                              prefix=None,
                              augment=True,
                              include_index=False
                returns       {'df' : the constructed df,
                               'out_file' : to where the df was written}
            
            EXTRACT_COL       df_src, 
                              col_name, 
                              dst_dir=None, 
                              prefix=None
                returns       {'col' : the requested column,
                               'out_file' : path where column was saved}
                              
            CLEAR_RESULTS     <none>
                returns       True
            
            ORGANIZE          <none>
                returns       None
                
            CHIRP_SIMILARITY  <none>
        
        :param action: action to perform
        :type action: Action
        '''
        
        self.log = LoggingService()

        self.proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # Default destination directory of analysis files:
        self.analysis_dst = Localization.analysis_dst
        self.chirps_dst   = Localization.sampling_dst
        self.srch_res_dst = Localization.srch_res_dst
        
        self.action = action
        
        self.res_file_prefix = 'perplexity_n_clusters_optimum'
        self.fid_map_file = os.path.join(Localization.measures_root, 'split_filename_to_id.csv')
        self.data_calcs = DataCalcs()
        self.data_calcs = DataCalcs(measures_root=Localization.measures_root,
                                    inference_root=Localization.inference_root,
                                    fid_map_file=self.fid_map_file
                                    )
        
        # All measures from many split files combined
        # into one df, and augmented with rec_datetime,
        # the recording time's sin/cos transformations for
        # hour, day, month, and year, and the is_daytime
        # column. This df includes the split number, file_id,
        # And a few other administrative data that should not
        # be used for training. This file may not yet exist,
        # for instance when the action is CONCAT, which creates
        # the file in the first place:
        
        all_measures_with_admins_fname = Localization.all_measures
        try:
            self.all_measures_with_admins = pd.read_feather(all_measures_with_admins_fname)
            
            # Take out the administrative data:
            self.all_measures = self.all_measures_with_admins.drop(columns=['file_id', 'split','rec_datetime'])
        except FileNotFoundError:
            self.log.info(f"The all_measures dataframe not loaded, b/c all_measures '{all_measures_with_admins_fname}' not found")

        if action is None:
            # Caller just wanted data init:
            return
                
        if action == Action.PCA:
            self.log.info("Computing PCA")
            kwargs['to_exclude'] = ['species', 'rec_datetime', 'split', 'chirp_idx', 'file_id']
            res = self.pca_action(**kwargs)
            
        elif action == Action.PCA_ANALYSIS:
            self.log.info("Analyzing PCA")
            res = self.pca_analysis_action(**kwargs)
            
        elif action == Action.ORGANIZE:
            self.log.info("Organizing hyper-search results")
            res = self.organize_results()            

        elif action == Action.HYPER_SEARCH:
            self.log.info("Starting TSNE hyperparameter search")
            # Returns PerplexitySearchResult instance:
            res = self.hyper_parm_search(**kwargs)

        elif action == Action.PLOT_SEARCH_RES:
            self.log.info("Plotting TSNE hyperparameter search results")
            required_args = ['search_results']
            if not all([kwarg_nm in kwargs.keys() for kwarg_nm in required_args]):
                raise ValueError(f"Plotting hyper parameter search results requires args {required_args}")
            
            res = self._plot_search_results(**kwargs)

        elif action == Action.SAMPLE_CHIRPS:
            self.log.info("Sampling chirps from entire chirp collection")
            required_args = ['num_samples']
            if not all([kwarg_nm in kwargs.keys() for kwarg_nm in required_args]):
                raise ValueError(f"Sampling chirps requires args {required_args}")
            
            if 'save_dir' not in kwargs.keys():
                chirp_samples_dir = os.path.join(self.proj_dir, 'results/chirp_samples/')
            else:
                chirp_samples_dir = kwargs['save_dir']
            kwargs['save_dir'] = chirp_samples_dir
            
            res = self._sample_chirps(**kwargs)
            
        elif action == Action.EXTRACT_COL:
            required_args = ['df_src', 'col_name']
            if not all([kwarg_nm in kwargs.keys() for kwarg_nm in required_args]):
                raise ValueError(f"Column extraction requires args {required_args}")
            self.log.info(f"Extracting column {kwargs['col_name']} from data")
            
            try:
                dst_dir = kwargs['dst_dir']
            except ValueError:
                dst_dir = self.chirps_dst
            kwargs['dst_dir'] = dst_dir
            
            try:    
                prefix = kwargs['prefix']
                if prefix is None:
                    prefix = f"col_{kwargs['col_name']}_"
            except KeyError:
                prefix = f"col_{kwargs['col_name']}_"
                
            kwargs['prefix'] = prefix
            res = self._extract_column(**kwargs)

        elif action == Action.CONCAT:
            self.log.info("Concatenating chirp files")
            required_args = ['df_sources']
            if not all([kwarg_nm in kwargs.keys() for kwarg_nm in required_args]):
                raise ValueError(f"CONCAT requires args {required_args}")

            try:
                dst_dir = kwargs['dst_dir']
            except KeyError:
                dst_dir = self.chirps_dst
                kwargs['dst_dir'] = dst_dir
            
            # Was a timestamp provided that is to be used in
            # generated files? (If not, now-time is used):
            try:
                timestamp = kwargs['timestamp']
            except KeyError:
                timestamp = None 
                kwargs[timestamp] = None
            
            res = self._concat_files(**kwargs)
            
        elif action == Action.CLEAR_RESULTS:
            self.log.info("Cleanung up TSNE hyper parameter search debris")
            res = self.remove_search_res_files()
            
        elif action == Action.PRINT_CLF_RESULTS:
            self.log.info("Printing classification results:")
            self.print_clf_results_action()
            res = None
            
        elif action == Action.CHIRP_SIMILARITY:
            self.log_info("Examining chirp similarity within 10sec recordings")
            res = self._chirp_similarities()
        
        # Save the results, usually packaged as a dict:
        if res is not None:
            self.experiment_result = res
        
    #------------------------------------
    # pca_action
    #-------------------
    
    def pca_action(self, 
                   n_components=None,
                   to_exclude=None,
                   to_include=None,
                   comment=None,
                   data_src=None
                   ):
        '''
        Runs PCA on all chirp data. Saves PCA object, weight matrix,
        and the transformed original data, as well as several
        figures. All these will be in Localization.analysis_dst.

        The to_include and to_exclude are optional lists of measures
        (columns in the measurements data) to exclude from the PCA.
        At most one of these may be used. 
         
        Returns PCAResult instance.
                
        The PCA object will have an attribute create_date. It
        can be used to timestamp both the PCA object itself, and
        related analysis files.
        
        :param n_components: number of components to which the
            data is to be reduced
        :type n_components: union[None, int]
        :param to_exclude: variables to exclude from computations
        :type to_exclude: optional[list[str]]
        :param to_include: which variables to include in computations
            either to_exclude, or to_include must be None
        :type to_include: optional[list[str]]
        :param comment: textural description to include in 
            computed, and returned records.
        :type optional[str]
        :param data_src: if provided: the path from where to 
            load the to main data. Default is None, which 
            means the already loaded self.all_measures is used.
        :type data_src: optional[str]
        :return: a PCAResult instance with all sorts of nice results
        :rtype PCAResult
        '''

        if to_include is not None and to_exclude is not None:
            raise ValueError('Only one of to_include and to_exclude may be non_None.')

        # Load source data.
        # If data source has a timestamp embedded in 
        # its filename, then use that timestamp in files
        # created further on. If data_src has no timestamp,
        # it ends up as None, and a current timestamp is
        # created further on:
        if data_src is None:        
            with UniversalFd(Localization.all_measures, 'r') as fd:
                all_data = fd.asdf()
                timestamp = Utils.extract_file_timestamp(Localization.all_measures)
        else:
            with UniversalFd(data_src, 'r') as fd:
                all_data = fd.asdf()
                timestamp = Utils.extract_file_timestamp(data_src)
        
        # Drop the measures to exclude (other than the
        # measure to be predicted):
        if to_exclude is not None:
            # Drop the cols in place     
            all_data.drop(axis='columns', columns=to_exclude, inplace=True)
            
        elif to_include is not None:
            cols_to_drop = set(all_data.columns) - set(to_include)
            all_data.drop(axis='columns', columns=cols_to_drop, inplace=True)
        
        if n_components is None:
            n_components = len(all_data.columns)
            
        # PCA returns a dict with keys 'pca', 'weight_matrix',
        # and 'xformed_data'. Also saves the PCA, returning the
        # file path to the saved pca in a dict:
        pca_result = self.data_calcs.pca_computation(
            df=all_data,
            n_components=n_components,
            timestamp=timestamp
            )

        return pca_result
    
    #------------------------------------
    # pca_analysis_action
    #-------------------
    
    def pca_analysis_action(self):
        
        interpretations = ResultInterpretations()
        # Run a new PCA only if necessary:
        analysis_results = interpretations.pca_run_and_report(run_new_pca=None)
        return analysis_results
    
    
    #------------------------------------
    # organize_results
    #-------------------
    
    def organize_results(self):
        '''
        Finds temporary files that hold PerplexitySearchResult
        exports, and those that contain plots made for those
        results. Moves all of them to self.srch_res_dst, under a 
        name that reflects their content. 
        
        For example:
        
                       perplexity_n_clusters_optimum_3gth9tp.json in /tmp
            may become 
                       perp_p100.0_n2_20240518T155827.json
            in self.dst__dir
            
        Plot figure png files, like perplexity_n_clusters_optimum_plot_20251104T204254.png
        will transfer unchanged.
        
        For each search result, the tsne_df will be replicated into a
        .csv file in self.srch_res_dst
            
        '''
        
        for fname in self._find_srch_results():
            # Guard against 0-length files from aborted runs:
            if os.path.getsize(fname) == 0:
                self.log.warn(f"Empty hyperparm search result: {fname}")
                continue
            
            # Saved figures are just transfered:
            if fname.endswith('.png'):
                shutil.move(fname, self.srch_dst_dir)
                continue
            
            srch_res = PerplexitySearchResult.read_json(fname)
            mod_time = self._modtimestamp(fname)
            perp = srch_res['optimal_perplexity']
            n_clusters = srch_res['optimal_n_clusters']
            
            dst_json_nm   = f"perp_p{perp}_n{n_clusters}_{mod_time}.json"
            dst_json_path = os.path.join(self.srch_dst_dir, dst_json_nm)
            
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
    # print_clf_results_action
    #-------------------
    
    def print_clf_results_action(self, json_path=None):
        
        # File path given?
        if json_path is not None \
            and os.path.exists(json_path) \
            and json_path.endswith('.json'):
            # Try to find a timestamp in the file:
            try:
                timestamp = Utils.time_from_fname(json_path)
            except ValueError:
                timestamp = ''
            # Where to write output:
            fname = f"classification_pretty_.txt"
            ClassifictionResult.printf(json_path,
                                       os.path.join(self.analysis_dst, fname) 
                                       )
            return
        
        elif json_path is None:
            # Find latest classification eval json
            # in Localization.analysis_dst:
            json_paths = Utils.find_file_by_timestamp(
                self.analysis_dst,
                prefix='clf_classification_eval_',
                suffix='.json',
                latest=True)
            if len(json_paths) == 0:
                FileNotFoundError(f"No classification file found in {self.analysis_dst}")
            json_path = json_paths[0]
            timestamp = Utils.extract_file_timestamp(json_path)
            # Where to write output:
            fname = f"classification_pretty_{timestamp}.txt"
            # Print to the file:
            self.log.info(f"Printing results to {fname}")
            ClassifictionResult.printf(json_path,
                                       os.path.join(self.analysis_dst, fname) 
                                       )
            # Also print a csv file that is only for spreadsheets,
            # not for reading back:
            fname = f"classification_pretty_{timestamp}.csv"
            self.log.info(f"Writing csv-style results to {fname}")
            ClassifictionResult.to_csv(json_path,
                                       os.path.join(self.analysis_dst, fname))

        else:
            raise FileNotFoundError(f"Bad classification result path: {json_path}")

    #------------------------------------
    # _chirp_similarities
    #-------------------
    
    def _chirp_similarities(self):
        '''
        From the current self.all_measures, extract some 
        chirp sequences, and examine how similar the
        chirps within a sequence tend to be to each other.
        '''
        
        
        

    #------------------------------------
    # _sample_chirps
    #-------------------
    
    def _sample_chirps(self, num_samples=None, save_dir=None):
        '''
        Sample chirps from many split files. For sampling
        chirps that also satisfy a condition from a single
        df, see DataCalcs.conditional_samples()
        
        :param num_samples:
        :type num_samples:
        :param save_dir:
        :type save_dir:
        '''
        data_calculator = DataCalcs(measures_root=self.data_calcs.measures_root,
                                    inference_root=self.data_calcs.inference_root,
                                    fid_map_file=self.fid_map_file
                                    )
        res_dict = data_calculator.make_chirp_sample_file(num_samples, save_dir=save_dir)
        _df, save_file = res_dict.values()
        self.log.info(f"Saved {num_samples} chirps in {save_file}")
        return res_dict

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
    
        path = os.path.join(self.data_calcs.measures_root, split_file)
    
        calc = DataCalcs(self.data_calcs.measures_root, 
                         self.data_calcs.inference_root, 
                         chirp_id_src=chirp_id_src,
                         fid_map_file=self.fid_map_file)
    
        with UniversalFd(path, 'r') as fd:
            calc.df = fd.asdf()
            
        # Use the .wav file information in file_id column to  
        # obtain each chirp's recording date and time. The new
        # column will be called 'rec_datetime', and an additional
        # column: 'is_daytime' will be added. This is done inplace,
        # so no back-assignment is needed:
        calc.add_recording_datetime_and_more(calc.df) 
        
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
        stamp = Utils.file_timestamp(time=moddt)
        return stamp

    #------------------------------------
    # _plot_search_results
    #-------------------
    
    def _plot_search_results(self, search_results, dst_dir=None):
    
        # The following will be a list: 
        # [perplexity1, ClusteringResult (kmeans run: 8) at 0x13f197b90), 
        #  perplexity2, ClusteringResult (kmeans run: 8) at 0x13fc6c2c0),
        #            ...
        #  ]
        cluster_results = []
        
        # As filepath for saving the figure at the end,
        # use the file prefix self.res_file_prefix, and
        # the current date and time:
        if dst_dir is not None:
            filename_safe_dt = Utils.file_timestamp()
            fig_save_fname   = f"{self.res_file_prefix}_plots_{filename_safe_dt}.png"
            fig_save_path    = os.path.join(dst_dir, fig_save_fname) 
        
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
        if dst_dir is not None:
            self.log.info(f"Saving plots to {fig_save_path}")
            fig.savefig(fig_save_path)
            fig.show()
        input("Any key to erase figs and continue: ")


    #------------------------------------
    # _extract_column
    #-------------------
    
    def _extract_column(self, df_src, col_name, dst_dir=None, prefix=None):
        '''
        Given either a dataframe, or a file the contains a dataframe
        extract the column col_name, and return it as a pd.Series.
        If dest_dir is non-None, it must be a destination dir. The
        file name will be:
        
                <dst_dir>/<prefix>_<col_name>_<now-time>.csv
                
        Source files may be csv, csv.gz, or .feather
        
        :param df_src: either a df, or source file that holds a df
        :type df_src: union[pd.DataFrame | src]
        :param col_name: name of column to extract
        :type col_name: src
        :param dst_dir: directory to store the resulting pd.Series.
            If None, Series is just returned, but not saved.
        :type dst_dir: union[None | src]
        :param prefix: prefix to place in destination file name
        :type prefix: union[None | str]
        :return: dict {'col' : the requested column,
                       'out_file' : path where column was saved}
        :rtype dic[str : union[str : union[None | pd.Series]
        '''
        
        if type(df_src) == str:
            if not os.path.exists(df_src):
                raise FileNotFoundError(f"Did not find dataframe source file {df_src}")
            else:
                with UniversalFd(df_src, 'r') as fd:
                    df = fd.asdf()
        else:
            if type(df_src) != pd.DataFrame:
                raise TypeError(f"Dataframe source {df_src} is not a dataframe.")
            df = df_src

        if prefix is None:
            prefix = ''
            
        try:
            col = df[col_name]
        except KeyError:
            raise ValueError(f"Column {col_name} not found in dataframe")
        
        if dst_dir is not None:
            # If caller passed a file, bad:
            if os.path.isfile(dst_dir):
                raise ValueError(f"Destination {dst_dir} is a file; should be a directory")
            
            os.makedirs(dst_dir, exist_ok=True)
            
            # Save the result:
            filename_safe_dt = Utils.file_timestamp()
            fname = os.path.join(dst_dir, f"{prefix}_column_{col_name}_{filename_safe_dt}.csv")
            col.to_csv(fname)
            self.log.info(f"Saved column {col_name} in {fname}")
        else:
            fname = None
        return {'col' : col, 'out_file' : fname}


    #------------------------------------
    # _concat_files
    #-------------------
    
    def _concat_files(self, 
                      df_sources,
                      idx_columns=None, 
                      dst_dir=None, 
                      out_file_type=FileType.CSV, 
                      prefix=None,
                      augment=True,
                      timestamp=None
                      ):
        '''
        Given a list of sourcefiles that contain dataframes, create one df,
        and save it to a file if requested. Control the output file type between
        .feather, .csv, and .csv.gz via the out_file_type arg.
        
        The outfile will be of the form:
        
                <dst_dir>/<prefix>_chirps_<now-time>.csv
        
        If idx_columns is provided, it must be a list of columns
        names with the same length as df_sources. If any of the
        files are .feather files, place a None at that list slot.
        This information is needed for a following .csv file:
           
              Idx  Col1   Col2
               0   'foo'  'bar'
               1   'blue' 'green'
               
        where the intention for the resulting dataframe is:
        
                   Col1    Col2
            Idx    
             0     'foo'   'bar'
             1     'blue'  'green'
        
        :param df_sources: list of dataframe file sources
        :type df_sources: union[str | list[str]]
        :param idx_columns: if provided, a list of column names that are
            to be used as index in the respective input file, rather than
            as a column name. Only used for .csv and .csv.gz. If the arg
            is a single string, or a single-value list, the string is taken 
            as a column name that is common to all input files. 
        :type idx_columns: union[None | list[str] | str]
        :param dst_dir: directory where combined file is placed. No saving, if None
        :type dst_dir: union[None | str]
        :param out_file_type: if saving to disk, which file type to use 
        :type out_file_type: FileType
        :param prefix: prefix to place in destination file name
        :type prefix: union[None | str]
        :param augment: if True, add columns for recording time, and
            sin/cos of recording time for granularities HOURS, DAYS, MONTHS, and YEARS 
        :type augment: bool
        :param timestamp: timestamp to integrate into the names of the generated files
        :type timestamp: union[datetime, str]
        :return: a dict with fields 'df', and 'out_file'
        :rtype dict[str : union[None | str]
        '''

        if type(df_sources) == str:
            df_sources = [df_sources]
            
        if prefix is None:
            prefix = ''
            
        # If timestamp is given, convert it to a string
        # of appropriate format:
        if isinstance(timestamp, datetime):
            timestamp = Utils.timestamp_from_datetime(timestamp)
        
        supported_file_extensions = [file_type.value for file_type  in FileType]
        
        # Ensure that idx_columns are the same length as df_sources:
        if idx_columns is not None:
            idx_col = None
            if type(idx_columns) == list and len(idx_columns) == 1:
                idx_col = idx_columns[0]
            elif type(idx_columns) == str:
                idx_col = idx_columns 
            if idx_col is not None:
                idx_columns = [idx_col]*len(df_sources)
            
            elif len(idx_columns) != len(df_sources):
                # One last chance: if 
                raise ValueError(f"A non-None idx_columns arg must be same length as df_sources (length {len(df_sources)}, not {len(idx_columns)}")
        else:
            # For convenience, generate a list of None idx_column names:
            idx_columns = [None]*len(df_sources)
        
        # Check that all files are of supported type, and exist:
        for src_file in df_sources:
            path = Path(src_file)
            # Existence:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Cannot find dataframe file {path}")
            # File type:
            if path.suffix not in supported_file_extensions: 
                raise TypeError(f"Dataframe source files must be one of {supported_file_extensions}")
            
        # Check destination:
        if dst_dir is not None:
            # If caller passed a file, bad:
            if os.path.isfile(dst_dir):
                raise ValueError(f"Destination {dst_dir} is a file; should be a directory")
            
            os.makedirs(dst_dir, exist_ok=True)
            
            # Compose file to which to save the result:
            if timestamp is None:
                filename_safe_dt = Utils.file_timestamp()
            else:
                filename_safe_dt = timestamp
            fname = os.path.join(dst_dir, f"{prefix}_chirps_{filename_safe_dt}{out_file_type.value}")
        else:
            fname = None
        
        dfs  = []
        cols = None
        for path, idx_col_nm in zip(df_sources, idx_columns):
            self.log.info(f"Concatenating {path}...")
            with UniversalFd(path, 'r') as fd:
                dfs.append(fd.asdf(index_col=idx_col_nm))
            if cols is None:
                cols = dfs[-1].columns 
            else:
                left_cols = dfs[-1].columns
                if len(left_cols) != len(cols) or any(left_cols != cols):
                    raise TypeError(f"Dataframe in file {path} does not have same columns as previous dfs; should be [{cols}]")

        df_raw = pd.concat(dfs, axis='rows', ignore_index=True)
        if augment:
            # Get a DataCalc instance, not needing 
            data_calc = DataCalcs(fid_map_file=self.fid_map_file)
            # Use the .wav file information in file_id column to  
            # obtain each chirp's recording date and time. The new
            # column will be called 'rec_datetime', and an additional
            # column: 'is_daytime' will be added. This is done inplace,
            # so no back-assignment is needed:
            df_with_rectime = data_calc.add_recording_datetime_and_more(df_raw)
            self.log.info("Adding time encoding via trig functions...")
            df = data_calc._add_trig_cols(df_with_rectime, 'rec_datetime')
            df.reset_index(drop=True, inplace=True)
        else:
            df = df_raw
 
        df.index.name = 'row_num'
        if fname is not None:
            with UniversalFd(fname, 'w') as fd:
                fd.write_df(df)
            #df.to_csv(fname, index=include_index)
            self.log.info(f"Concatenated {len(df_sources)} bat measures files with total of {len(df)} rows (chirps) to {fname}")

        res_dict = {'df' : df, 'out_file' : fname}
        #*********
        #df_recovered = DataCleaner().recover_orig_from_scaled_data(
        #    '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/AnalysisReady/split_scaler.joblib',
        #    df)
        #*********
        return res_dict

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        rep_str = f"<MeasuresAnalysis chirps:{self.action.name} at {hex(id(self))})"
        return rep_str

# ------------------------------- Class ResultInterpretations -------------


class ResultInterpretations:
    '''
    Methods in this class poke around in results of measurement
    runs in the MeasurementsAnalysis class. For each measurement
    type, a method pulls up the results, and extracts what would
    be reported in a paper, or would be needed for next steps. 
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, measure_analysis=None, dst_dir=None):
    
        self.log = LoggingService()    
        self.proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.measure_analysis = measure_analysis
        if dst_dir is None:
            self.dst_dir = Localization.analysis_dst
        else:
            self.dst_dir = dst_dir
    
    #------------------------------------
    # pca_run_and_report
    #-------------------
    
    def pca_run_and_report(self, run_new_pca=None):
        '''
        Entry point to compute a PCA, and generate data
        and chart files with the results.
        
        The run_new_pca may be None, True, or False. If
        the argument is:
        
            None:  checks the Localization.analysis_dst directory
                   for the existence of a PCA result. Uses the
                   latest of those. Else, runs a new PCA
            True:  run a new PCA whether or not any PCA results
                   are already available
            False: Use existing PCA, but raise FileNotFoundError
                   if no PCA is found in the Localization.analysis_dst
                   directory. 
        
        :param run_new_pca: whether or not to run a new
            PCA first
        :type run_new_pca: union[None, bool]
        :return dict with analysis results
        :rtype dict[str : any]
        '''

        # Find any existing PCA results in Localization.analysis_dst:
        # Get list of saved PCA object file names
        # in the destination dir:
        pca_files = Utils.find_file_by_timestamp(Localization.analysis_dst,
                                                 prefix='pca_', 
                                                 suffix='.joblib',
                                                 latest=True)
        
        if run_new_pca:
            # PCA all data no matter whether PCA results
            # are already available
            action = Action.PCA
            # Create PCA object and weight matrix files in Localization.analysis_dst: 
            analysis = MeasuresAnalysis(action)
            pca_info = analysis.experiment_result['pca']
        
        elif run_new_pca is False and len(pca_files) == 0:
            raise FileNotFoundError(f"No pca files in {Localization.analysis_dst}. Aborting")
        else:
            # If available, use existing PCA:
            if len(pca_files) == 0:
                action = Action.PCA
                # Create PCA files:
                analysis = MeasuresAnalysis(action)
                # Analysis now has in experiment_result:
                #      dict_keys(['pca', 
                #                 'weight_matrix', 
                #                 'xformed_data', 
                #                 'pca_file', 
                #                 'weights_file', 
                #                 'xformed_data_fname'])
                # Grab the PCA object:
                pca_info = analysis.experiment_result['pca']                
        
            else:
                # Grab the filename where the PCA was saved:
                pca_info = pca_files[0]
                
        # Create analysis files: charts and data:
        if type(pca_info) == str:
            self.log.info(f"About to report on PCA in file {pca_info}")
        else:
            pca_timestamp = pca_info.create_date
            self.log.info(f"About to report on PCA from {pca_timestamp}")
        res = self.pca_report(pca_info)
        return res
    
    #------------------------------------
    # pca_report
    #-------------------
    
    def pca_report(self, pca_info):
        '''
        Look at result from pca that retains all features. 
        
           1. For each component, find the most important feature. Create a df:
           
                                   mostImportant       weight       loading     direction
                    ComponentNum
                          0         'feature3'         -0.436        0.190096        -1
                          1         'feature50'         0.987        0.974169        +1
                              ... 

               where mostImportant is the feature name of the SonoBat measure
               that is most important to the component represented by its row.
               
               Weight is the PCA weight for the most important feature.
               
               Loading is the feature's loading, which is the square of the weight.
               It indicates how strongly the feature impacts the component, and varies
               between 0 and 1.
               
               The direction is whether the loading impact is positive or negative.
               
               Write the df to file as:
                
                      component_importance_analysis_{pca_timestamp}.csv   

           2. Write a dataframe to file that shows the accumulating explained
              variance as increasingly more components would be used:
              
                                      percExplained    cumPercExplained
                    componentNum    
                         0                0.33             0.33
                         1                0.22             0.55
                               ...
                               
              The frame is written to:
              
                      variance_explained_{pca_timestamp}.csv
              
           3. Create figure of variance explained vs. number of components.
           
              Store it in:
              
                      variance_explained_{pca_timestamp}.png
           
           4. Analyze which features have the most impact on components.
           
           
        For input, must provide either the pca_result object, or the pca_fname 
        where the pca is stored.
        
        Returns a dict that combines results from this method, and
        from features_from_components():
        
           {
             'top_feature_per_component'   : df,
             'var_explained_per_component' : df,
             'features_impact'             : df with loadings for each feature in each
                                             component, and cumulative impact on all components 
                                             together, scaled by the importance of each
                                             component.
             'features_topn_90Perc'        : list of original features that contribute
                                             90% of the impact on components.
                        
        :param pca_result: PCA result object
        :type pca_result: union[None, sklearn.PCA]
        :param pca_fname: file where PCA object is stored
        :type pca_fname: union[None, str]
        :param pca_weights_matrix: the weight matrix df computed by the PCA
        :type pca_weights_matrix: optional[str]
        :param pca_weights_matrix_fname: file where pca's weights matrix df is stored 
        :type pca_weights_matrix_fname: union[None, str]
        :return dict with all results
        :rtype dict[str : any]
        '''
        
        if type(pca_info) == str:
            # Load file at path pca_info:
            pca_result = DataCalcs.load_pca(pca_info)
        elif isinstance(pca_info, PCA):
            pca_result = pca_info
        else:
            raise TypeError(f"Argument pca_info must be a file path or a PCA object, not {pca_info}")
        
        # Get the weight matrix
        weights = pd.DataFrame(pca_result.components_, columns=pca_result.feature_names_in_)

        # Build a dataframe
        #
        #                  mostImportant       weight       loading     direction
        #   ComponentNum
        #         0         'feature3'         -0.436        0.190096        -1
        #         1         'feature50'         0.987        0.974169        +1
        #                                 ... 
        
        # Get a Series of feature names. Each element at index idx of the
        # Series holds that name of the feature that is most 
        # imoprtant for the idxth component:
        #
        #     0 Feature3
        #     1 Feature50
        #        ...
        #
        # Get the featureNm
        max_importance_feature_per_component = weights.idxmax(axis=1)
        max_importance_feature_per_component.name = 'mostImportant'
        max_importance_feature_per_component.index.name = 'componentNum'

        # Get the most important features' weights themselves:
        #    Feature3   -0.436
        #    Feature50   0.987
        #        ...
        # A Series:
        max_weights = weights.max(axis=1)
        # Row names are the per component maximally important features
        # we got earlier via idxmax():
        max_weights.name  = 'weight'
        
        # The loadings are the squares of the weights:
        loadings = max_weights.pow(2)
        loadings.name  = 'loading'
        
        # And the direction of each component's max-impact feature
        # on its component: 1.0 and -1.0
        direction = max_weights / max_weights.abs()
        direction.name = 'direction'
        
        component_importance_analysis = pd.concat([max_importance_feature_per_component,
                                                   max_weights,
                                                   loadings,
                                                   direction
                                                   ],axis=1)

        # For the file name, use the same timestamp as was used for the PCA file:
        pca_timestamp = Utils.timestamp_from_datetime(pca_result.create_date)
        important_features_fname = f"component_importance_{pca_timestamp}.csv"
        
        self.log.info(f"Saving summary of how features impact components to {important_features_fname}")
        component_importance_analysis.to_csv(os.path.join(self.dst_dir, important_features_fname))
        
        # Next: explained variance. Create a df:
        #                    percExplained    cumPercExplained
        #  componentNum    
        #       0                0.33             0.33
        #       1                0.22             0.55
        #             ...
        
        explained_var  = pd.Series(pca_result.explained_variance_ratio_, name='percExplained')
        cum_explained  = pd.Series(itertools.accumulate(explained_var), name='cumPercExplained')
        explain_amount = pd.concat([explained_var, cum_explained], axis=1)
        explain_amount.index.name = 'numComponents'
        
        # For the file name, use the same timestamp as was used for the PCA file:
        # The datetime 
        cum_expl_fname = f"variance_explained_{pca_timestamp}.csv"
        
        self.log.info(f"Saving summary for explanatory power of each PCA component to {cum_expl_fname}")
        
        explain_amount.to_csv(os.path.join(self.dst_dir, cum_expl_fname))
        
        # Create a chart to show the increase in explained variance
        # as number of components is increased:
        
        fig = DataViz.simple_chart(explain_amount.cumPercExplained, 
                                   xlabel='Components Deployed', 
                                   ylabel='Total % Variance Explained' 
                                   )
        # Add dashed horizontal and vertical lines to 
        # mark the 90% point:
        ax = fig.gca()
        x_coord = 22
        y_coord = 0.9
        DataViz.draw_xy_lines(ax, 
                              x_coord, y_coord, 
                              color='gray',
                              linestyle='dashed', 
                              alpha=0.5)
        
        
        fig_fname = f"variance_explained_{pca_timestamp}.png"
        
        self.log.info(f"Saving figure with explained variance as components are added to {fig_fname}")
        
        fig.savefig(os.path.join(self.dst_dir, fig_fname))
        
        # Next, analyze how the original *features* impact
        # the components:
        
        features_res_dict = self.features_from_components(pca_result)
        # Add our own results:
        features_res_dict['feature_impact']
        features_res_dict['features_topn90_perc']
        
        features_res_dict['var_explained_per_component'] = explain_amount
        features_res_dict['top_feature_per_component']   = component_importance_analysis 
        
        self.log.info("Done PCA")
        
        return features_res_dict
        
    #------------------------------------
    # features_from_components
    #-------------------
    
    def features_from_components(self, pca_info):
        '''
        Given a PCA object and a number of components that was
        deemed to sufficiently explain variance.
        
        Return a dict:
        
            {'feature_impact'      : df with loadings for each feature in each
                                     component, and cumulative impact on all components 
                                     together, scaled by the importance of each
                                     component.
             'features_topn90_perc' : list of original features that contribute
                                      90% of the impact on components.
            }
        
        Saves a figure to Localization.analysis_dst that shows the
        chart of features vs. impact: feature_importance_{pca_timestamp}.csv
        
        Approach: for each component, identify the features whose
        loadings are high within that component. Those are features to keep
        for this component. The loadings of each feature on a given component
        are scaled by the component's amount of explained variance.
        
        :param pca_info: from which file load the pca, or the pca object itself 
        :type pca: union[str, sklearn.decomposition.PCA
        :return dict with all results
        :rtype dict{str : union[pd.DataFrame, list[str]]
        '''
    
        if type(pca_info) == str:
            # Load file at path pca_info:
            pca = DataCalcs.load_pca(pca_info)
        elif isinstance(pca_info, PCA):
            pca = pca_info
        else:
            raise TypeError(f"Argument pca_info must be a file path or a PCA object")
            
        weights       = pd.DataFrame(pca.components_, columns=pca.feature_names_in_)        
        loadings      = weights.pow(2)
        
        # For each feature F, compute its importance: multiply
        # the loading of F in each principal component by that 
        # component's contribution to explaining variance. I.e.
        # multiple F's column in the loadings matrix by the column
        # vector that hold's the components' contribution.
        # Then sum those scaled loadings to get Series:
        #     TimeInFile         3.734815e-01
        #     PrecedingIntrvl    1.457879e-01
        #     CallsPerSec        6.105740e-02
        #     CallDuration       4.194274e-02
        #     Fc                 3.319674e-02
        #                            ...     
        #     cos_day            1.950079e-33
        #     sin_month          1.950079e-33
        #     cos_month          1.950079e-33
        #     sin_year           1.950079e-33
        #     cos_year           1.950079e-33
        #     Length: 116, dtype: float64
                 
        rel_feature_importance  = loadings.mul(pca.explained_variance_ratio_)\
                                        .sum(axis=0)\
                                        .sort_values(ascending=False)
        rel_feature_importance.name = 'impact'                                
                                        
        # The rel_feature_importance Series adds to 1, which 
        # represents to total impact of all features on all
        # components. Compute the cumulative percentage of 
        # impact contributed by the features:
        
        feature_impact_contribution = pd.Series(itertools.accumulate(rel_feature_importance),
                                                name='impact_contribution_perc',
                                                index=rel_feature_importance.index)
        # Df with the individual scaled loading for each
        # feature, and the cumulative impact of successive
        # features to the overall loading in its two columns:
        feature_impact_df = pd.concat([rel_feature_importance, feature_impact_contribution],
                                      axis=1
                                      ) 

        pca_timestamp = Utils.timestamp_from_datetime(pca.create_date)
        feature_impact_fname = f"feature_importance_{pca_timestamp}.csv"

        self.log.info(f"Saving feature impact on components in {feature_impact_fname}")
         
        feature_impact_df.to_csv(os.path.join(self.dst_dir, feature_impact_fname))
        
        # Find the number features after which 
        # 90% of impact is explained:
        
        # In our data:  ==> 'TotalSlope'
        feature_name = feature_impact_contribution.loc[feature_impact_contribution < 0.9].idxmax()
        # The how manieth feature is that? The 1+ is 
        # because feature_name is the last feature that
        # has not reached 90%
        top_n = 1 + feature_impact_contribution.index.get_loc(feature_name)
        
        # The +1 is to include the 90th percent feature:
        top_n_feature_names = feature_impact_df.index[:top_n + 1]
        
        # Plot the progressive feature impact: x axis
        # are the feature names in order of importance
        # Y axis are the cumulative percentage:
        
        # Make a copy of the df column, because we'll change
        # the index labels to fit the figure:
        data = feature_impact_df['impact_contribution_perc'].copy()
        # Shorten labels:
        new_labels = [label if len(label) <= 10 else f"{label[:7]}..."
                      for label 
                      in data.index
                      ]
        data.index = new_labels
        
        fig = DataViz.simple_chart(data,
                                   ylabel='Cumulative feature impact (%)',
                                   xlabel='',
                                   figsize=[6.79, 6.67]
                                   )
        ax = fig.gca()
        # Rotate the long measurement names on the
        # X axis:
        ax.tick_params(axis='x', labelrotation=45)
        
        # Highlight the 90% point with vertical and
        # horizontal line:

        # x and y values we want to highlight:
        x_coord = top_n
        y_coord = feature_impact_contribution.iloc[top_n]
        
        # Add dashed horizontal and vertical lines to 
        # mark the 90% point:
        DataViz.draw_xy_lines(ax, 
                              x_coord, y_coord, 
                              color='gray',
                              linestyle='dashed', 
                              alpha=0.5)
        
        # Place a text box with the top 22 highest impact
        # features in a text box:
        textstr = '\n'.join(list(feature_impact_df.index[:top_n + 1]))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.7, 0.9, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)

        fig_fname = f"feature_impact_on_components_{pca_timestamp}.png"
        
        self.log.info(f"Saving figure summarizing how features impact components in {fig_fname}")
        
        fig.savefig(os.path.join(self.dst_dir, fig_fname))
        
        
        res = {'feature_impact'      : feature_impact_df,
               'features_topn90_perc': top_n_feature_names
               }
        
        return res
            
# ------------------------- Classification ------------

class Classification:
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, dst_dir=None):
        self.log = LoggingService()
        if dst_dir is None:
            self.dst_dir = Localization.analysis_dst
        else:
            self.dst_dir = dst_dir
    
    #------------------------------------
    # binary_classification
    #-------------------
    
    def classify(self, 
                 to_predict, 
                 to_exclude=None,
                 to_include=None,
                 classifier=None,
                 n_fold=5, 
                 timestamp=None,
                 cm_title=None,
                 pr_title=None,
                 class_label_map=None,
                 class_imbalance_fix=None,
                 positive_label=None,
                 cm_normalization=None,
                 comment=None,
                 data_src=None # Default: use preloaded self.all_measures
                 ):
        '''
        Build a classifier to predict the variable(s) listed
        in to_predict. The dataset is assumed to be in 
        Localization.all_measures. 
        
        After that dataframe is loaded, all columns listed
        in to_exclude are removed from the loaded model.
        If to_include is a list of columns, then those are
        included in the computations. Only one of to_include
        and to_exclude must the non_None.
        
        If an sklearn classifier object is passed in via the
        classifier argument, it is used. Else, Logistic Regression
        is used for binary classification, and Multinomial Logistic
        Regression for multiple target classes.
        
        Stratified n-fold cross validation is deployed, meaning that
        classifications are run n-fold times, and results are averaged.
        The number of folds can be specified in the n_fold arg.
        
        If a timestamp is provided, it is embedded in the names of  
        all the files that are created in the method. If timestamp
        is None, the current datetime is used.
        
        The class_label_map argument helps replacing difficult-to-interpret
        data values with human readable labels. Those will be used in
        result dataframes and charts. Example:
        
            {True  : 'daytime',
             False : 'nighttime}
             
            {'sm'  : 'Small'
             'md'  : 'Medium'
             'lg'  : 'Large'}
        
        The class_imbalance_fix argument may be specified to cause 
        data augmentation via the SMOTE algorithm, which upsamples
        some or all of the target class training data
        (Chawla, Nitesh V., et al. "SMOTE: synthetic minority over-sampling technique." 
         Journal of artificial intelligence research 16 (2002): 321-357.)
         
        Other than None, the following values are allowed:
            SMOTEOption.
               MINORITY     : resample only the minority class 
               NOT_MINORITY : resample all classes but the minority class
               NOT_MAJORITY : resample all classes but the majority class
               ALL          : resample all classes
               AUTO         : equivalent to 'not majority'

        The positive_label is the name of the class of interest.
        This is vague, but useful when selecting which value of
        the prediction variable you want information on, the predicted 
        variable having the value True? Or maybe the value False?
        In multi-class specifying the value is even more useful.
      
        The classifiers use the set of unique values of the
        prediction variable as class names. Those, plus their
        ordering are in classifier_obj.classes_ 
         
        The conf_mat_normalization argument determines whether
        the confusion matrix of the mean results across all folds
        will be normalized:
        
             o over the true (rows)
             o the predicted (columns), or
             o all of the population (all)
             
        If None, no normalization. See signature of sklearn.confusion_matrix()
                   
        
        The following are returned, and also saved to self.dst_dir:

           - A correlation matrix in numeric form (f"clf_conf_matrix_{timestamp}.csv") 
           - Correlation matrix image (f"clf_conf_matrix_fig_{timestamp}.png")
           - A precision-recall chart, both numeric and as image
           
        The returned dict will contain:

           - A ClassificationResult instance holding all the mean of
             classification score results from all folds.

             The ClassificationResult will include:             
           
	             - Average (naive) accuracy score
	             - Balanced average accuracy score
	             - Confusion matrix
	             - Mean of F1 scores across folds
	             - Number of cases for each class
	             
	                                   : res_dict['mean_scores']
	       
	       ******* Need to include PR in result dict?
           - Results from precision/recall   res_dict['prec_rec_results']
             analysis:
               {
                'pr_df'                : pr_df,
                'pos_label'            : pr_curve.pos_label,
                'prevalence_pos_label' : pr_curve.prevalence_pos_label,
                'average_precision'    : pr_curve.average_precision,
                'fig'                  : <the precision-recall figure>
               }
               
        The comment argument will be added to the returned result ClassificationResult
        object.
        
        :param to_predict: variable(s) to predict
        :type to_predict: union[str, list[str]]
        :param to_exclude: variables to exclude from computations
        :type to_exclude: optional[list[str]]
        :param to_include: which variables to include in computations
            either to_exclude, or to_include must be None
        :type to_include: optional[list[str]]
        :param classifier: an already instantiated sklearn classifier to use
        :type classifier: optional[sklearn.*.Classifier]
        :param n_fold: how many folds to use for cross validation
        :type n_fold: optional[int]
        :param timestamp: datetime to use in the filenames of saved results
        :type timestamp: optional[str, datetime]
        :param cm_title: title to place at top of confusion matrix chart
        :type cm_title: optional[str]
        :param pr_title: title to place at top of precision/recall chart
        :type pr_title: optional[str]
        :param class_label_map: map of data values to human-readable labels
        :type class_label_map: optional[dict[str : str]
        :param class_imbalance_fix: the SMOTE sampling strategy to use.
            Default: no resampling
        :type class_imbalance_fix: SMOTEOption
        :param positive_label: class label of interest. Refers to the
            value of the prediction variable whose probability is to
            be determined in the classification.
        :type positive_label: optional[str]
        :param cm_normalization: whether to normalize the result
            confusion matrix: 'true': over the 'true' (rows) conditions,
                the predicted 'pred' (columns) condition, whole population
                'all'. Or no normalization None (default)
        :type cm_normalization: optional[list['true', 'pred', 'all']]
        :param data_src: if None, assumes data to use is self.all_measures,
            i.e. the measures dataframe. Else this must be the path
            to a file with the data to load.
        :type data_src: optional[src] 
        :returns a dict of results
        :rtype dict[str : any]
        '''
    
        if to_include is not None and to_exclude is not None:
            raise ValueError('Only one of to_include and to_exclude may be non_None.')
    
        if timestamp is None:
            # Get current datetime as a string:
            timestamp = Utils.file_timestamp()
        elif type(timestamp) == str:
            # All good:
            pass
        elif isinstance(timestamp, datetime):
            timestamp = Utils.timestamp_from_datetime(timestamp)
        else:
            raise TypeError(f"Timestamp must be None, a datetime instance, or timestamp string, not {timestamp}")

        if data_src is None:        
            with UniversalFd(Localization.all_measures, 'r') as fd:
                all_data = fd.asdf()
        else:
            with UniversalFd(data_src, 'r') as fd:
                all_data = fd.asdf()
        
        # Drop the measures to exclude (other than the
        # measure to be predicted):
        if to_exclude is not None:
            # Drop the cols in place     
            all_data.drop(axis='columns', columns=to_exclude, inplace=True)
            
        elif to_include is not None:
            cols_to_drop = set(all_data.columns) - set(to_include)
            all_data.drop(axis='columns', columns=cols_to_drop, inplace=True)
        
        self.positive_label = positive_label
        
        # Get the desired results for the classification,
        # such as is_daylight:
        if to_predict in all_data.columns:
            y = all_data[to_predict]
            # Data must exclude the predicted variable. But
            # if caller passed in a dataset, such as PCA-transformed
            # data, the predicted var isn't present, so tolerate that
            # absence:
            X = all_data.drop(to_predict, axis='columns', inplace=False)              
        else:
            # Get truth from the original measures:
            with UniversalFd(Localization.all_measures, 'r') as fd:
                all_measures_with_admins = fd.asdf()
            y = all_measures_with_admins[to_predict]
            X = all_data
        
        # Sanity: X and y must have the same length:
        if len(X) != len(y):
            raise ValueError(f"Length of X and y must match, but are {len(X)} and {len(y)}, respectively") 
        
        # Split into test/train; test set to be 25% of data,
        # with a random_state set to get reproducible results:
        
        # Get a StratifiedKFold instance:
        skf = StratifiedKFold(n_splits=n_fold,
                              shuffle=False, # True not recommended for SKF
                              random_state=None)
        
        n_samples  = len(X)
        n_features = len(X.columns)
        n_classes  = len(set(y))
        
        dummy_X = pd.DataFrame(np.zeros((n_samples, n_features)), columns=X.columns)
                                
        # Returns a generator with n_splits 2-tuples:
        # Each tuple has as its first element indices 
        # into the data for training set rows, and
        # the second element is indices to get a corresponding
        # test set:
         
        trn_tst_idxs_it = skf.split(dummy_X, y)
        
        if classifier is not None:
            clf = classifier
        elif n_classes == 2:
            clf = LogisticRegression(max_iter=200)
        else:
            raise NotImplementedError(f"Predicting multiple classes not implemented; feature {to_predict} has {n_classes} distinct values.")

               
        # Place for ClassificationResult instances
        # from each fold: 
        scores = []

        # Collect the prediction probabilities for each fold:
        y_pred_probs = []
        
        for fold_num, (train_idxs, test_idxs) in enumerate(trn_tst_idxs_it):
            X_train, X_test = X.iloc[train_idxs], X.iloc[test_idxs]
            y_train, y_test = y.iloc[train_idxs], y[test_idxs]            

            if class_imbalance_fix is not None:
                self.log.info(f"Oversampling: {class_imbalance_fix.value}")
                sm = SMOTE(random_state=1066, sampling_strategy=class_imbalance_fix.value)
                # Oversample as requested; the SMOTEOption passed in
                # is an Enum whose values are the proper strings for
                # the sm.fit_resample() method:
                X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

                X_train = X_train_resampled
                y_train = y_train_resampled

            self.log.info(f"Fitting model to fold {fold_num}")
            clf.fit(X_train, y_train)
            
            self.log.info(f"Using model to predict model in fold {fold_num}")
            
            # Predict class membership:
            y_pred = clf.predict(X_test)
            
            # Predict class probabilities:
            y_pred_proba = clf.predict_proba(X_test)
            
            # Make a df from the two y_pred_proba columns,
            # and the current fold's y_test.
            # Occasionally, the length of the returned
            # y_pred_proba will be one more than 
            # the length of X_test (and y_test). Chop
            # off the additional, and make into a df.
            # The column names will be the class labels
            # used by the classifier:
            
            probs_df = pd.DataFrame(y_pred_proba[:len(X_test), :], 
                                    columns=clf.classes_, 
                                    index=X_test.index)
            # For binary classification, at least for Logistic 
            # Regression, the column names will be boolean:
            # np.bool. To make the columns easier to access,
            # we turn those bools into strings: 'True', 'False'
            
            # Add this fold's y_test truth column:
            # The new column will have name 'y_true':
            saved_cols = list(probs_df.columns)
            probs_df = pd.concat([probs_df, y_test], axis='columns')
            saved_cols.append('y_true')
            probs_df.columns = saved_cols 
            y_pred_probs.append(probs_df)

            # Note the results for this fold:
            scores.append(ClassifictionResult(y_test, y_pred, 
                                              classes_=clf.classes_, 
                                              class_label_map=class_label_map,
                                              comment=f"Scores fold{fold_num}; imbalance fix: {class_imbalance_fix}",
                                              cm_normalize=cm_normalization
                                              ))

        # Get a ClassifictionResult instance
        # that holds the means of all measurements
        # from the folds. A comment will be constructed
        # from the comments injected in each fold score,
        # unless the caller provided one:
        self.log.info(f"Computing mean of results from all {n_fold} folds")
        mean_scores_obj = ClassifictionResult.mean(scores, comment=comment)
        
        # Result Metrics Computations
        res_dict = {'mean_scores' : mean_scores_obj}    
        
        # Draw a confusion matrix:
        mean_conf_mat = mean_scores_obj.conf_mat
        _cm_df, cm_fig = self.make_confusion_matrix(mean_conf_mat,
                                                    clf,
                                                    cm_title, 
                                                    class_label_map)
        
        # Save the confusion matrix image:
        
        fig_fname = f"clf_conf_matrix_fig_{timestamp}.png"
        cm_fig.savefig(os.path.join(self.dst_dir, fig_fname))
        
        pr_dict, pr_fig = self.make_precision_recall(y_pred_probs,
                                                     clf, 
                                                     pr_title=pr_title, 
                                                     plot_chance_level=False)
        # Add results or precision-recall analysis:
            # 'pr_df'     : pr_df,
            # 'pos_label' : pr_curve.pos_label,
            # 'prevalence_pos_label' : pr_curve.prevalence_pos_label,
            # 'average_precision'    : pr_curve.average_precision,
            # 'fig'                  : fig
        
        res_dict['prec_rec_results'] = pr_dict
        
        # Save the figure:
        pr_curve_fname = f"clf_pr_curve_{timestamp}.png"
        pr_fig.savefig(os.path.join(self.dst_dir, pr_curve_fname))
        
        clf_res_fname = f"clf_classification_eval_{timestamp}.json"
        self.log.info(f"Saving classification result to {clf_res_fname}")
        mean_scores_obj.to_json(os.path.join(self.dst_dir, clf_res_fname))
        
        return res_dict

    #------------------------------------
    # make_confusion_matrix
    #-------------------
    
    def make_confusion_matrix(self, 
                              confusion_matrix,
                              classifier,
                              cm_title=None,
                              class_label_map=None
                              ):
        '''
        Takes one or more confusion matrices, picks one
        at random, and creates a confusion matrix figure.
        
        Returns the confusion matrix, and the figure.  
        
        :param confusion_matrix: confusion matrix as produced
            by sklearn.confusion_matrix
        :type confusion_matrix: [ndarray[n_classes, n_classes]]
        :param classifier: classifier object that computed results
        :type classifier: sklearn.sklearn.linear_model.<classifier>
        :param cm_title: title for top of confusion matrix figure
        :type cm_title: optional[str]
        :param class_label_map: map from data values to human readable labels.
            Example: {True : 'terrorist', False: 'civilian'}
        :type class_label_map: optional[dict[str : str]]
        :returns a dataframe with the confusion matrix data, and the Figure object
        :rtype tuple[pd.DataFrame, Figure]
        '''
        # If confusion matrices is a list of cms, 
        # pick one at random:
        confusion_matrix

        cm_fig = ConfusionMatrixDisplay(confusion_matrix.to_numpy())

        cm_fig.plot(values_format='', cmap='Blues')
        
        fig = cm_fig.figure_
        if cm_title is not None:
            fig.suptitle(cm_title)        

        # Set the class labels of the confusion matrix:
        if class_label_map is not None:
            ax = fig.gca()
            cf_label_objs = ax.get_xmajorticklabels()
            data_values    = classifier.classes_
            encoded_values = {data_value  : encoding 
                              for encoding, data_value
                              in enumerate(data_values)}
            
            for data_value in data_values:
                try:
                    # From the data value to the desired label
                    # in the conf matrix:
                    target_label = class_label_map[data_value]
                    # The numeric class encoding, which is currently
                    # the (string formatted) label used in the conf 
                    # matrix viz:
                    encoding = str(encoded_values[data_value])
                    try:
                        # Among the conf matrix label text objects,
                        # find the one that is a string of the encoding
                        
                        txt_obj = next(filter(lambda cf_label_obj: cf_label_obj.get_text() == encoding,
                                              cf_label_objs))
                        txt_obj.set_text(target_label)
                    except StopIteration:
                        raise ValueError(f"Cannot find confusion matrix tick labeled {data_value}")
                    
                except KeyError:
                    self.log.warn(f"Data value {data_value} has no mapping to human label in {class_label_map}. Retaining raw value")
                    continue

            # Change the figure labels:            
            ax.set_xticklabels(cf_label_objs)
            ax.set_yticklabels(cf_label_objs)
            # Same for the conf mat df columns and index:
            confusion_matrix.columns = [cm_label.get_text() for cm_label in cf_label_objs]
            confusion_matrix.index = [cm_label.get_text() for cm_label in cf_label_objs]
            
            fig.show()          
        return confusion_matrix, fig
        
    #------------------------------------
    # make_precision_recall 
    #-------------------
    
    def make_precision_recall(self,
                              y_pred_probs,
                              classifier,
                              curve_label=None, 
                              plot_chance_level=True,
                              pr_title=None,
                              ):
        '''
        For one or more binary classifications: draw precision/recall 
        curves for each into the same plot. 
        
        The y_pred_probs is expected to be a list of dataframes.
        Each df must have the form:
        
		   	          False      True      y_true
		   	0      0.002854  0.997146       False
		   	1      0.003423  0.996577       False
		   	2             ...
		   	        
        where y_true are the true values of the predicted
        variable. That column is the truth data for the 
        probabilities in the False/True columns.
        
        Draw the PR chart(s), all in one figure:
        
        Returns a dict:
           {
            'pr_df'                : pr_df,
            'pos_label'            : pr_curve.pos_label,
            'prevalence_pos_label' : pr_curve.prevalence_pos_label,
            'average_precision'    : average of the average precisions
                                     across all folds
            'fig'                  : <PR-figure>
           }
        
        The positive_label is the class of main interest. In bat chirps,
        when trying to predict whether a chirp was recorded during the 
        day, then the was-at-daytime class would be the positive class.
        Other examples: presence of a disease, or is-spam-message.   
        
        The prevalence_pos_label is the percentage of the dataset
        for which the outcome is the positive label. For example, if
        the positive label is 1 ('daytime'), and prevalence_pos_label
        is 0.0972, then the true 'daytime' outcome is only 10%, i.e. 
        the data are imbalanced.   
        
        :param y_pred_probs: dataframes, each with probabilities
            for negative and positive class membership, plus
            a column with truth.
        :type y_pred_probs: list[pd.DataFrame]
        :param classifier: the classifier object that produced
            the probabilities. Used to find number of classes.
        :type classifier: sklearn.Classifier
        :param curve_label: label for the PR curve
        :type curve_label: optional[str]
        :param plot_chance_level: whether to plot a horizontal curve
            at the precision you would get for the positive class
            by guessing.
        :type plot_chance_level: bool
        :param pr_title: title above the figure
        :type pr_title: optional[str]
        :return a dict with all pr-curve numeric info, and the matplotlib Figure object
        :rtype tuple[dict[str : any], Figure] 
        '''

        num_classes  = len(classifier.classes_)
        class_labels = classifier.classes_

        # Find the class label for the positive class:
        if num_classes > 2 and self.positive_label is None:
            msg = f"For multi-class results (this one is {num_classes} classes), must provide positive_label when calling classify()"
            raise ValueError(msg)
        
        elif num_classes == 2 and self.positive_label is None:
            # Guess that the positive label is 1:
            if True in class_labels:
                # Find the position of np.True_ in class_labels,
                # which is an np_array without the Python list's index().
                true_pos = list(class_labels).index(True)
                positive_label = class_labels[true_pos]
            else:
                # Pick the last of the probability columns:
                positive_label = class_labels[-1]
                self.log.warn(f"No positive-class label provided; choosing last in {classifier.classes_}")
        
        # The [:,1] selects the probs of the second class:
        # If plot_chance_level=True, draws horizontal line at precision
        # you would get by guessing the majority class. With class imbalance, 
        # that won't be .5, 
        
        precision_vals = []
        # Go through each df with probabily/truth info, and
        # make a pr_curve object:
        fig, ax = plt.subplots()
        for prediction in y_pred_probs:
            
            y_true = prediction.y_true
            y_prob = prediction.loc[:,positive_label]
            pr_curve = PrecisionRecallDisplay.from_predictions(
                y_true, y_prob,
                #pos_label=True, 
                name=curve_label, 
                plot_chance_level=plot_chance_level,
                ax=ax # Draw all curves on one figure
                )
            precision_vals.append(pd.Series(pr_curve.precision))
        if pr_title:
            fig.suptitle(pr_title)

        # Get mean of precision values, and draw final curve
        # in black:
        # Get the shortest of the precision series:
        size = min([len(ser) for ser in precision_vals])
        precs_df = pd.concat([ser[0:size] for ser in precision_vals], axis='columns')
        prec_mean_y = precs_df.mean(axis='columns')
        recs_x      = pr_curve.recall[:size]
        ax.plot(recs_x, prec_mean_y, color='black')
        fig.show()
        
        # The overall ave prec is the mean of 
        # precisions of the mean curve:
        avg_prec_overall = prec_mean_y.mean()

        # Create dict with the means for saving.
        recs_ser  = pd.Series(recs_x, name='recall') 
        precs_ser = pd.Series(prec_mean_y, name='mean_precision')
        pr_df = pd.DataFrame({'recall' : recs_ser, 'precision' : precs_ser})
        
        res_df = {
            'pr_df'                : pr_df,
            'pos_label'            : pr_curve.pos_label,
            'prevalence_pos_label' : pr_curve.prevalence_pos_label,
            'average_precision'    : avg_prec_overall,
            'fig'                  : fig
            }

        return res_df, fig
        

    
    #------------------------------------
    # _make_custom_class_labels
    #-------------------
    
    def _make_custom_class_labels(self, clf, target_labeling):
        '''
        Given a mapping from data values, return labels that
        should be used for visualizations. Example:
        
        Given: 
            {False : 'Nighttime',
             True  : 'Daytime'
             }
              
        or:
            {'large'  : 'father',
             'medium: : 'mother',
             'small'  : 'child'
             }
             
        The classifier's internals will be:
          classes_  = ['large', 'medium', 'small']
          encodings = [0,1,2]
          
        Return a dict that maps internal, encoded
        class designations (0, 1, 2, ...) to the 
        desired labels ('Nighttime', 'father', ...)
        
            {0  : 'father',
             1  : 'mother', 
             2  : 'child'}
        
        :param clf: classifier
        :type clf: Classifier
        :param target_labeling:
        :type target_labeling:
        :return Dict from numeric class labels to target 
            labels: {0 : 'father', 1 : 'mother', ...}
        :rtype dict[union[str, int] : str]
        '''
        
        encoder = LabelEncoder()
        # Get one data value example for each class:
        # These might be [True, False], or ['small', 'medium', 'large']
        # in the classifier's internal order:
        data_values = clf.classes_
        
        # For each class get the corresponding internal
        # classifier class labels: [0,1, 2, 3,...]
        encodings = encoder.fit_transform(data_values)
        
        from_encoded = {encoding : target_labeling[data_values[i]]
                        for i, encoding 
                        in enumerate(encodings)}
        return from_encoded
    
            
# TODO:

#   the 0th class probabilities has one more result
#   then y_true. The other four probs lists are fine.

def get_timestamp():
    # Find any existing results in Localization.analysis_dst:
    clf_files = Utils.find_file_by_timestamp(Localization.analysis_dst,
                                             prefix='clf_', 
                                             suffix='.joblib',
                                             latest=True)
    if len(clf_files) > 0:
        # Use the timestamp of the first found prior-run result file:
        timestamp = Utils.extract_file_timestamp(clf_files[0])
    else:
        timestamp = None
    return timestamp

# ------------------------ Main ------------
if __name__ == '__main__':

    log = LoggingService()
    
    #******************
    # df_raw = pd.read_feather('/tmp/df_raw.feather')
    # fid_map_file = os.path.join(Localization.measures_root, 'split_filename_to_id.csv')
    # data_calc = DataCalcs(fid_map_file=fid_map_file)
    # df_with_rectime = data_calc.add_recording_datetime_and_more(df_raw)
    # df = data_calc._add_trig_cols(df_with_rectime, 'rec_datetime')
    
    #******************
    # -------------------------------------------------------------------
    # Create one DF from all demeaned split files. They
    # are all of the form splitnn.feather, with nn being
    # a number. Then create a second df, which will contain
    # the unscaled values. Returned is a dict:
    #   res_dict: {df_scaled : <combined data from all (scaled) split files
    #                          with columns, like is_daylight added.
    #              df_orig   : <unscaled, original SonoBat data, including
    #                          columns that were not included (and therefore
    #                          scaled) in the split files.
    
    # split_file_pat = re.compile(f"^split[\\d]+.feather")
    # split_files = []
    # for maybe_split_fname in os.listdir(Localization.measures_root):
    #     match = split_file_pat.search(maybe_split_fname)
    #     if match is None:
    #         continue
    #     else:
    #         split_files.append(os.path.join(Localization.measures_root, maybe_split_fname))
    #
    # # Sort the split files by split number (which is taken
    # # from the split file name):
    # split_files.sort(key=lambda fname: int(re.search(r"\d+", fname)[0]))
    #
    # # Is there a file called timestamp.txt at the source dir?
    # # If so, it contains a timestamp string of when the split
    # # files were created, and we use it in filenames that we
    # # create as we concat the splits:
    #
    # timestamp_path = f"{Localization.measures_root}/timestamp.txt"
    # if os.path.exists(timestamp_path):
    #     with open(timestamp_path, 'r') as fd:
    #         timestamp = fd.read()
    # else:
    #     timestamp = None
    #
    # #df_sources = ['/Users/paepcke/quatro/home/vdesai/data/training_data/all/splits/split20.feather',
    # #              '/Users/paepcke/quatro/home/vdesai/data/training_data/all/splits/split21.feather'
    # #              ]
    # # The split files have a col called 'level_0', which contains
    # # row number that runs across all split files. Make that
    # # column the pandas index. That column 
    # ma = MeasuresAnalysis(action=Action.CONCAT, 
    #                       df_sources=split_files, 
    #                       dst_dir=os.path.dirname(Localization.all_measures), 
    #                       prefix='scaled',
    #                       idx_columns='level_0',
    #                       out_file_type=FileType.FEATHER,
    #                       augment=True,
    #                       timestamp=timestamp
    #                       )
    # res_dict = ma.experiment_result
    # df_out_file = res_dict['out_file']
    # df = res_dict['df']
    # df_orig = DataCleaner.recover_orig_from_scaled_data(Localization.scaler, df)
    # # Save this unscaled version as well:
    # all_measures_path = Path(df_out_file)
    # # Use the same time stamp for the original (descaled)
    # # df as was used for the scaled on
    # timestamp = Utils.extract_file_timestamp(all_measures_path)
    # df_orig_path = all_measures_path.parent / Path(f'orig_chirps_{timestamp}.feather')
    # df_orig.to_feather(df_orig_path)
    # # Save as .csv as well, b/c Tableau can't read .feather:
    # df_orig_path_csv = Path(df_orig_path).with_suffix('.csv')
    # df.orig.to_csv(df_orig_path_csv)
    #
    #
    # res_dict['df_orig'] = df_orig
    # #print(res_dict)
    # print('Done')
    # sys.exit()

    # -------------------------------------------------------------------
    # Explore how different chirps are within one 10sec recording:
    
    
    # Make repeatable
    # ma = MeasuresAnalysis(action=Action.CHIRP_SIMILARITY)
    # random.seed = 1066
    # random.randint()
    
        
    # -------------------------------------------------------------------
    # Descaling and re-meaning a StandardScaler treated df:
    # src_file = Localization.all_measures
    # with UniversalFd(src_file, 'r') as fd:
    #     df_scaled = fd.asdf()
    # df_orig = DataCleaner.recover_orig_from_scaled_data(Localization.scaler, df_scaled)
    #
    # additional_cols = ['file_id','chirp_idx','split',
    #                    'rec_datetime','is_daytime','sin_hr',
    #                    'cos_hr','sin_day',
    #                    'cos_day','sin_month',
    #                    'cos_month','sin_year','cos_year']
    # additionals_df = pd.DataFrame(df_scaled[additional_cols])
    # df_orig = pd.concat([df_orig, additionals_df], axis='columns')
    #
    # dst_fpath   = Path(Localization.all_measures_descaled)
    # dst_feather = dst_fpath.with_suffix('.feather')
    # dst_csv     = dst_fpath.with_suffix('.csv')
    #
    # log.info(f"Saving reconstituted data as .feather to {dst_feather}...")
    # df_orig.to_feather(dst_feather)
    # log.info(f"Saving reconstituted data as .csv to {dst_csv}...")    
    # df_orig.to_csv(dst_csv)
    #
    # print('Done')
    # sys.exit()
    
    # -------------------------------------------------------------------   
    # Stats of data: a df like:
    #                      numFiles   numChirps    meanChirpsPerFile
    #     Timing              
    #     Nighttime
    #     Daytime
    #     Total

    # ma = MeasuresAnalysis(action=None)
    # gp = ma.all_measures_with_admins.groupby(by='file_id')
    #
    # # Nums Overall:
    #
    # num_files_total = len(gp.groups)
    # num_chirps_total = len(ma.all_measures_with_admins)
    # min_chirp_seq_len_total = gp.size().min()
    # max_chirp_seq_len_total = gp.size().max()
    # mean_chirp_seq_len_total = gp.size().mean()
    #
    # # Nums Nighttime:
    # # Get groups of measures df like:
    # #    {False : [3,5,6...],
    # #      True : [10,50,30...]
    # gp_light = ma.all_measures_with_admins.groupby(by='is_daytime')
    # # The night indexes: is_daytime is False:
    # night_idxs = gp_light.groups[False]
    # day_idxs   = gp_light.groups[True]
    # night_df   = ma.all_measures_with_admins.iloc[night_idxs, :]
    # day_df     = ma.all_measures_with_admins.iloc[day_idxs, :]
    #
    # num_chirps_night = len(night_df)
    # num_chirps_day   = len(day_df)
    #
    # gp_night_df      = night_df.groupby(by='file_id')
    # num_files_night  = len(gp_night_df.groups)
    # min_night_chirp_seq_len = gp_night_df.size().min()
    # max_night_chirp_seq_len = gp_night_df.size().max()
    # mean_night_chirp_seq_len = gp_night_df.size().mean()
    #
    # gp_day_df      = day_df.groupby(by='file_id')
    # num_files_day  = len(gp_day_df.groups)
    # min_day_chirp_seq_len = gp_day_df.size().min()
    # max_day_chirp_seq_len = gp_day_df.size().max()
    # mean_day_chirp_seq_len = gp_day_df.size().mean()
    #
    # # All together:
    # data_summary = pd.DataFrame(
    #     {'numFiles'     	: [num_files_night, num_files_day, num_files_total],
    #      'numChirps'    	: [num_chirps_night, num_chirps_day, num_chirps_total],
    #      'meanChirpSeqLen'  : [mean_night_chirp_seq_len, mean_day_chirp_seq_len, mean_chirp_seq_len_total],
    #      'minChirpSeqLen'   : [min_night_chirp_seq_len, min_day_chirp_seq_len, min_chirp_seq_len_total],
    #      'maxChirpSeqLen'   : [max_night_chirp_seq_len, max_day_chirp_seq_len, max_chirp_seq_len_total],
    #      },
    #     index=['Nighttime', 'Daytime', 'Total'],
    #     )
    # data_summary.index.name='Timing'
    # tbl_txt = data_summary.to_markdown()
    # timestamp = Utils.file_timestamp()
    # tbl_fname = f"data_summmary_tbl_{timestamp}.txt"
    # with open(os.path.join(Localization.analysis_dst, tbl_fname), 'w') as fd:
    #     fd.write(tbl_txt)
    # sys.exit()

    # -------------------------------------------------------------------
    # Print correlation of everyone against is_daytime:
    # mc = MeasuresAnalysis(action=None)
    # df = mc.all_measures_with_admins.copy()
    # df.rec_datetime = df.rec_datetime.apply(lambda dt_str: Utils.datetime_from_timestamp(dt_str).hour)
    #
    # corrs_with_is_daytime = DataCalcs.correlate_all_against_one(df, 'is_daytime')
    # print(corrs_with_is_daytime)

    # -------------------------------------------------------------------
    # Histogram of all daytime chirps, hour by hour:
    
    # mc = MeasuresAnalysis(action=None)
    # df = mc.all_measures_with_admins
    #
    # # All rec_daytime xformed to datetime objs:
    # rec_times = mc.all_measures_with_admins.rec_datetime.apply(
    #     lambda dtstr: Utils.datetime_from_timestamp(dtstr))
    #
    # daytime_idxs = df.loc[df.is_daytime].index
    # daytime_dates  = rec_times.iloc[daytime_idxs]
    # daytime_hrs    = daytime_dates.apply(lambda dt : dt.hour)
    # daytime_hrs.name = 'hour'
    # daytime_mins   = daytime_dates.apply(lambda dt : dt.minute)
    # daytime_mins.name = 'minute'
    # daytime_hr_min = pd.concat([daytime_hrs, daytime_mins], axis='columns')
    # hr_grouped = daytime_hr_min.groupby(by=['hour', 'minute'])
    # # Now:
    # # hr_grouped.groups
    # #{(19, 0): [15505, ...],
    # # (19, 1): [15090, ...],
    # #      ...
    # # (20, 1): [12758, ...],
    # # (20, 2): [13758, ...],
    # #      ...
    # # }
    #
    # # Get the lengths of each minute:
    # num_chirps_by_min = {key : len(vals) for key, vals in hr_grouped.groups.items()}
    # # Now have like:
    # #    {(19, 0): 105, (19, 1): 129, (19, 2): 158, (19, 3): 154,...,
    # #     (20, 0): 493, (20, 1): 447, (20, 2): 579, (20, 3): 504, 
    # #     }
    #
    # times    = list(num_chirps_by_min.keys())
    # freqs    = list(num_chirps_by_min.values())
    # # Turn the tuples (19, 0), (20, 32), etc. into
    # # strings like '19:00', '20:32', etc.. The zfill(2)
    # # ensures leading zeroes if hours or minutes are
    # # only one digit long:
    # times    = [f"{str(hr_min[0]).zfill(2)}:{str(hr_min[1]).zfill(2)}" 
    #             for hr_min 
    #             in num_chirps_by_min.keys()]
    #
    # histo_df = pd.DataFrame(freqs, index=times, columns=['num_chirps'])
    # histo_df.index.name = 'chirp_time'
    #
    # fig = DataViz.simple_chart(histo_df, 
    #                            xlabel='Time', 
    #                            ylabel='Chirp Activity', 
    #                            kind='bar', 
    #                            title="Daylight Bat Hourly Activity")
    # ax = fig.gca()
    # # Thin out the x axis ticks to only be when
    # # the hour changes:
    # ax.xaxis.set_major_locator(DataViz.hour_xticker())
    # sys.exit()

    # -------------------------------------------------------------------   
    # Print the most recent classification result:
    # MeasuresAnalysis(Action.PRINT_CLF_RESULTS)
    # sys.exit()

    # -------------------------------------------------------------------   
    # Create a correlation heatmap for all the
    # numeric measurements:
    
    # with UniversalFd(Localization.all_measures, 'r') as fd:
    #     df = fd.asdf()
    #     df.drop(['rec_datetime', 'species', 'is_daytime', 
    #              'freq_mean', 'file_id', 'chirp_idx', 'split',  
    #              'sin_hr', 'cos_hr', 'sin_day', 
    #              'cos_day', 'sin_month', 'cos_month', 'sin_year', 
    #              'cos_year'], axis='columns', inplace=True)
    # corr = df.corr()
    # fig = DataViz.heatmap(corr, 
    #                       #title='Measures Correlations',
    #                       width_height=(10,10),
    #                       #xlabel_rot=45,
    #                       color_legend=False,
    #                       axis_label_fontsize=24, 
    #                       #save_file=save_file
    #                       )
    # ax = fig.gca()
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xlabel('Chirp Measures[$\\it{1..n}$]', fontsize=24)
    # ax.set_ylabel('Chirp Measures[$\\it{1..n}$]', fontsize=24)
    # save_file = '/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/AnalysisResults/ResearchWriteups/Figures/cross_corr_heatmap.png'
    # fig.savefig(save_file)    
    #
    #
    # print('Done')
    # sys.exit()
    
    # -------------------------------------------------------------------   
    # Create a barchart of importance of features:
    # with UniversalFd('/Users/paepcke/Project/Wildlife/Bats/VarunExperimentsData/AnalysisResults/PCA/feature_importance_2024-06-02T13_29_54.csv', 'r') as fd:
    #     df = fd.asdf()
    # fig = DataViz.simple_chart(df.impact_contribution_perc, 
    #                            ylabel='Variance Explained', 
    #                            title='Feature Importance', 
    #                            kind='bar')
    # print(fig)
    

    # -------------------------------------------------------------------   
    # Run PCA: Three option: (1) unconditionally run a 
    # brand new PCA with all float-typed measures. 
    # (2) Run an existing PCA again, but with the 23 more
    # influential features only. (3) Make a new PCA, if
    # one does not already exist, else use the existing one.
    # [Don't remember which dir it looks in, but likely
    #  the default dst_dir]
    #
    # You can pass a string as a 'comment' str. It will be
    # saved in a file sibling of the PCA instance.
     
    # Force a new PCA to be made:
    #
    # Include all non-administrative measures, and 
    # as many components as there are measures:
    ma = MeasuresAnalysis(Action.PCA)
    pca_result = ma.experiment_result
    # Write result file set to Localization.analysis_dst:
    pca_result.save()
    res_dict = ma.data_calcs.pca_needed_dims(pca_result, variance_threshold=0.9)
    num_comps,\
    sufficient_features,\
    feature_powers,\
    explained_var = res_dict.values()
    # Save the features_powers for vizualizing how 
    # feature power adds up: 
    timestamp = Utils.timestamp_from_datetime(pca_result.timestamp)

    feat_pwr_fname = f"pca_feature_powers_{timestamp}.csv"
    feat_pwr_path  = os.path.join(Localization.analysis_dst, feat_pwr_fname)
    feature_powers.to_csv(feat_pwr_path)
    
    suff_feats_fname = f"pca_sufficient_features_{timestamp}.csv"
    suff_feats_path  = os.path.join(Localization.analysis_dst, feat_pwr_fname)
    sufficient_features.to_csv(suff_feats_path)
    
    sys.exit()

    # comment = ('PCA analysis to create 23 components, which was previously '
    #            'determined to cover 90% of the variance. Excluded is '
    #            'is_daytime, because we will want to predict that with the '
    #            'components. Total of 297,476 chirps. In-features: 115.'
    #            )
    # # Include all non-administrative measures that  
    # # are not time related, other than TimeInFile:
    # analysis = MeasuresAnalysis(
    #     Action.PCA,
    #     n_components=23,
    #     to_exclude=['is_daytime'],
    #     comment=comment
    #     )
    # sys.exit()
    
    
    # Make a new PCA, or use an existing one if available:
    # analysis = MeasuresAnalysis(Action.PCA_ANALYSIS)
    # res = analysis.experiment_result
    
    # -------------------------------------------------------------------    
    # # Train to predict is_daytime with top22 measures:
    #
    # timestamp = get_timestamp()        
    # clf = Classification()
    #
    # inclusions = [ 
    #             'TimeInFile', 'PrecedingIntrvl', 'CallsPerSec',
    #             'CallDuration', 'Fc', 'HiFreq', 'LowFreq',
    #             'Bndwdth', 'FreqMaxPwr', 'PrcntMaxAmpDur',
    #             'TimeFromMaxToFc', 'FreqKnee', 'PrcntKneeDur',
    #             'StartF', 'EndF', 'DominantSlope', 'SlopeAtFc',
    #             'StartSlope', 'EndSlope', 'SteepestSlope',
    #             'LowestSlope', 'TotalSlope', 'HiFtoKnSlope',
    #             'is_daytime' # The var to predict
    #             ]
    #
    # res_dict = clf.classify(
    #     'is_daytime',
    #     #to_exclude=exclusions,
    #     to_include=inclusions,
    #     class_label_map={True : 'Daytime',
    #                      False: 'Nighttime'},
    #     cm_title="Daytime Prediction: Confusion Matrix",
    #     pr_title="Daytime Prediction: Precision/Recall Tradeoff",
    #     timestamp=timestamp,
    #     class_imbalance_fix=SMOTEOption.MINORITY
    #     )
    #----------------------------------------------
    
    # Train to predict is_daytime with top23 measures PLUS recording time:
    # See commented-out sin/cos entries to vary with or without time info:
    
    # timestamp = get_timestamp()        
    # clf = Classification()
    #
    # # inclusions = [ 
    # #             'TimeInFile', 'PrecedingIntrvl', 'CallsPerSec',
    # #             'CallDuration', 'Fc', 'HiFreq', 'LowFreq',
    # #             'Bndwdth', 'FreqMaxPwr', 'PrcntMaxAmpDur',
    # #             'TimeFromMaxToFc', 'FreqKnee', 'PrcntKneeDur',
    # #             'StartF', 'EndF', 'DominantSlope', 'SlopeAtFc',
    # #             'StartSlope', 'EndSlope', 'SteepestSlope',
    # #             'LowestSlope', 'TotalSlope', 'HiFtoKnSlope',
    # #             #'sin_hr', 'cos_hr',
    # #             'is_daytime' # The var to predict
    # #             ]
    # inclusions = [ 
    #             'CallsPerSec', 'CallDuration', 'Fc', 'Bndwdth', 
    #             'PrcntMaxAmpDur', 'FreqKnee', 'DominantSlope',
    #             #'sin_hr', 'cos_hr',
    #             'is_daytime' # The var to predict
    #             ]
    #
    # res_dict = clf.classify(
    #     'is_daytime',
    #     #to_exclude=exclusions,
    #     to_include=inclusions,
    #     class_label_map={True : 'Daytime',
    #                      False: 'Nighttime'},
    #     cm_title="Daytime Prediction: Confusion Matrix",
    #     pr_title="Daytime Prediction: Precision/Recall Tradeoff",
    #     timestamp=timestamp,
    #     class_imbalance_fix=SMOTEOption.MINORITY,
    #     cm_normalization=None,
    #     # NOTE: not logistic regression here:
    #     classifier=RandomForestClassifier(
    #         max_depth=6,
    #         min_samples_leaf=5,
    #         n_jobs=5
    #         ),
    #     n_fold=3,
    #     #****comment="Predict daytime rec. Sin/Cos time provided. Imbalance fix: SMOTE.Minority",
    #     #****comment="Top 23 features. Predict daytime rec. Cos/sin NOT provided Imbalance fix: SMOTE.Minority",
    #     #****comment="Top 23 features. Predict daytime rec. Cos/sin NOT provided. Random Forest. Imbalance fix: SMOTE.Minority",
    #     comment="Small selection of top23. Folds: 3. Predict daytime. Cos/sin NOT provided. Random Forest. Imbalance fix: SMOTE.Minority"
    #     #***************
    #     #n_fold=2
    #     #***************
    #     )
    # # Place two files into the Localization.analysis_dst dir:
    # # A .csv file with the scores as a table. This file cannot
    # # be used for recovering the scores. Just for spreadsheets.
    # # The second file is a printable .txt file withe a human
    # # readable table with the classification results. The scores
    # # will be read from the json files that clf.classify placed
    # # in Localization.analysis_dst:
    # MeasuresAnalysis(Action.PRINT_CLF_RESULTS)
    
    #----------------------------------------------
    
    # Train to predict is_daytime with the PCA transformed measures.
    
    # timestamp = get_timestamp()        
    # clf = Classification()
    #
    # # Imbalanced vs. balanced data:
    # res_dict = clf.classify(
    #     'is_daytime',
    #     class_label_map={True : 'Daytime',
    #                      False: 'Nighttime'},
    #     cm_title="Daytime Prediction PCA Components: Confusion Matrix",
    #     pr_title="Daytime Prediction PCA Components: Precision/Recall Tradeoff",
    #     timestamp=timestamp,
    #     cm_normalization=None,
    #     #comment="Top 23 components. Predict daytime. Data imbalanced.",
    #     comment="Top 23 components. Predict daytime. Imbalance fix: SMOTE.Minority.",
    #     class_imbalance_fix=SMOTEOption.MINORITY,
    #     data_src=Localization.pca_xformed # USE THE PCA-TRANSFORMED DATA!
    #     )
    #
    # # Place two files into the Localization.analysis_dst dir:
    # # A .csv file with the scores as a table. This file cannot
    # # be used for recovering the scores. Just for spreadsheets.
    # # The second file is a printable .txt file withe a human
    # # readable table with the classification results. The scores
    # # will be read from the json files that clf.classify placed
    # # in Localization.analysis_dst:
    # MeasuresAnalysis(Action.PRINT_CLF_RESULTS)
    #
    # print(clf)

    #----------------------------------------------
    # Find best cluster k for top 23, and make chart:
    # inclusions = [ 
    #                 #'TimeInFile', 
    #                 'PrecedingIntrvl', 'CallsPerSec',
    #                 'CallDuration', 'Fc', 'HiFreq', 'LowFreq',
    #                 'Bndwdth', 'FreqMaxPwr', 'PrcntMaxAmpDur',
    #                 'TimeFromMaxToFc', 'FreqKnee', 'PrcntKneeDur',
    #                 'StartF', 'EndF', 'DominantSlope', 'SlopeAtFc',
    #                 'StartSlope', 'EndSlope', 'SteepestSlope',
    #                 'LowestSlope', 'TotalSlope', 'HiFtoKnSlope',
    #                 #'sin_hr', 'cos_hr'
    #                 ]
    # ma = MeasuresAnalysis(action=None)
    #
    # X_df = ma.all_measures[inclusions].sample(10000)
    # k_silhouttes_df = DataCalcs.find_optimal_k(X_df, range(2,6))
    # sil_scores = k_silhouttes_df['silhouette_score']
    # best_row   = sil_scores.idxmax()
    # best_k     = k_silhouttes_df.loc[best_row, 'k']
    # kmeans    = k_silhouttes_df.loc[best_row, 'kmeans']
    #
    # # Get cluster labels
    # cluster_labels = kmeans.labels_
    #
    # #*****
    # from sklearn.manifold import TSNE
    # #*****
    # # Define desired lower dimension (e.g., 2 for visualization)
    # tsne_obj = TSNE(n_components=2, 
    #                 init='pca', 
    #                 #perplexity=30,
    #                 perplexity=100,
    #                 metric='cosine',
    #                 n_jobs=8,
    #                 random_state=3
    #                 )
    #
    # # Apply t-SNE to the data
    # embedded_data = tsne_obj.fit_transform(X_df)
    # tsne_df = pd.DataFrame(embedded_data, columns=['tsne_x', 'tsne_y'])
    # DataViz.plot_tsne(tsne_df, cluster_labels, title='Cluster Top Features', show_plot=True)
    # print('Done')
