'''
Created on May 28, 2024

@author: paepcke
'''
from data_calcs.data_calculations import DataCalcs, PerplexitySearchResult, \
    ChirpIdSrc, FileType, Localization
from data_calcs.data_viz import DataViz
from data_calcs.universal_fd import UniversalFd
from data_calcs.utils import Utils
from datetime import datetime
from enum import Enum
from logging_service.logging_service import LoggingService
from pathlib import Path
from tempfile import NamedTemporaryFile
import itertools
import os
import pandas as pd
import random
import shutil
import sys
import time


# Members of the Action enum are passed to run
# to specify which task the program is to perform: 
class Action(Enum):
    HYPER_SEARCH    = 0
    PLOT_SEARCH_RES = 1
    ORGANIZE        = 2
    CLEAR_RESULTS   = 3
    SAMPLE_CHIRPS   = 4
    EXTRACT_COL     = 5
    CONCAT          = 6
    PCA             = 7


class MeasuresAnalysis:
            
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, action, **kwargs):
        '''
        Initializes various constants. Then, if action is 
        one of the Action enum members, executes the requested
        experiment.
        
        Each experiment generates a dictionary with relevant
        results. These results are stored in the MeasuresAnalysis
        experiment_result attribute. The action that identifies
        the experiment done is in the action attribute. 
        
        For each type of experiment there may be required and/or
        optional keyword arguments. Provide them all as kwargs,
        though the args without a default below are mandatory:
        
            PCA:              n_components
                returns       {'pca' : pca, 
		                       'weight_matrix' 	 : weight_matrix, 
		                       'xformed_data'  	 : xformed_data,
		                       'pca_file'      	 : pca_dst_fname,
		                       'weights_file'  	 : weight_fname,
   	                           'xformed_data_fname' : xformed_data_fname                
		                       }
		   
            
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
        self.fid_map_file = 'split_filename_to_id.csv'
        self.data_calcs = DataCalcs()
        self.data_calcs = DataCalcs(measures_root=Localization.measures_root,
                                    inference_root=Localization.inference_root,
                                    fid_map_file=self.fid_map_file
                                    )
        
        # All measures from 10 split files combined
        # into one df, and augmented with rec_datetime,
        # the recording time's sin/cos transformations for
        # hour, day, month, and year, and the is_daylight
        # column. This df includes the split number, file_id,
        # And a few other administrative data that should not
        # be used for training:
        
        all_measures_with_admins_fname = Localization.all_measures
        all_measures_with_admins = pd.read_feather(all_measures_with_admins_fname)
        
        # Take out the administrative data:
        self.all_measures = all_measures_with_admins.drop(columns=['file_id', 'split','rec_datetime'])

        
        if action == Action.PCA:
            self.log.info("Computing PCA")
            res = self.pca_action(**kwargs)
            
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
            
            res = self._concat_files(**kwargs)
            
        elif action == Action.CLEAR_RESULTS:
            self.log.info("Cleanung up TSNE hyper parameter search debris")
            res = self.remove_search_res_files()

        
        # Save the results, usually packaged as a dict:
        self.experiment_result = res
        
    #------------------------------------
    # pca_action
    #-------------------
    
    def pca_action(self, n_components=None):
        '''
        Runs PCA on all chirp data. Saves PCA object, weight matrix,
        and the transformed original data.
         
        Returns dict:
        
               {'pca' : pca, 
                'weight_matrix' 	 : weight_matrix, 
                'xformed_data'  	 : xformed_data,
                'pca_file'      	 : pca_dst_fname,
                'weights_file'  	 : weight_fname,
                'xformed_data_fname' : xformed_data_fname                
                }
                
        THe PCA object will have an attribute create_time. It
        can be used to timestamp both the PCA object itself, and
        related analysis files.
        
        :param n_components: number of components to which the
            data is to be reduced
        :type n_components: union[None, int]
        :return: dict with pc, weight matrix, and transformed data, 
            as well as the file names where they were saved.
        :rtype dict[str : any]
        '''
        if n_components is None:
            n_components = len(self.all_measures.columns)
            
        # PCA returns a dict with keys 'pca', 'weight_matrix',
        # and 'xformed_data'. Also saves the PCA, returning the
        # file path to the saved pca in a dict:
        pca, weight_matrix, xformed_data, pca_save_file = self.data_calcs.pca_computation(
            df=self.all_measures, 
            n_components=n_components,
            dst_dir=self.analysis_dst
            ).values()

        # Path where PCA was save is of the form:
        # <dir>/pca_20240528T161317.259849.joblib.
        # Get the timestamp:
        num_in_features  = pca.n_features_in_
        num_samples      = pca.n_samples_
        
        # Save the weights matrix with the same 
        # timestamp as when the PCA object was saved:
        weights_fname = Utils.mk_fpath_from_other(pca_save_file,
                                                  prefix='pca_weights_', 
                                                  suffix='.feather',
                                                  features=num_in_features,
                                                  samples=num_samples)

        self.log.info(f"Saving weights matrix to {weights_fname}")
        weight_matrix.to_feather(weights_fname)
        
        # Save the transformed data:
        xformed_data_fname = Utils.mk_fpath_from_other(pca_save_file,
                                                       prefix='xformed', 
                                                       suffix='.feather',
                                                       components=n_components,
                                                       samples=num_samples)
        self.log.info(f"Saving transformed data to {xformed_data_fname}")
        xformed_data.to_feather(xformed_data_fname)
        
        return {'pca' : pca, 
                'weight_matrix' 	 : weight_matrix, 
                'xformed_data'  	 : xformed_data,
                'pca_file'      	 : pca_save_file,
                'weights_file'  	 : weights_fname,
                'xformed_data_fname' : xformed_data_fname
                }
    
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
    # _sample_chirps
    #-------------------
    
    def _sample_chirps(self, num_samples=None, save_dir=None):
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
                      include_index=False
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
            to be used as names for the respective index, rather than
            as a column name. Only used for .csv and .csv.gz
        :type idx_columns: union[None | list[str]]
        :param dst_dir: directory where combined file is placed. No saving, if None
        :type dst_dir: union[None | str]
        :param out_file_type: if saving to disk, which file type to use 
        :type out_file_type: FileType
        :param prefix: prefix to place in destination file name
        :type prefix: union[None | str]
        :param augment: if True, add columns for recording time, and
            sin/cos of recording time for granularities HOURS, DAYS, MONTHS, and YEARS 
        :type augment: bool
        :return: a dict with fields 'df', and 'out_file'
        :rtype dict[str : union[None | str]
        '''

        if type(df_sources) == str:
            df_sources = [df_sources]
            
        if prefix is None:
            prefix = ''
        
        supported_file_extensions = [file_type.value for file_type  in FileType]
        
        # Ensure that idx_columns are the same length as df_sources:
        if idx_columns is not None:
            if len(idx_columns) != len(df_sources):
                raise ValueError(f"A non-None idx_columns arg must be same length as df_sources (lenght {len(df_sources)}, not {len(idx_columns)}")
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
            filename_safe_dt = Utils.file_timestamp()
            fname = os.path.join(dst_dir, f"{prefix}_chirps_{filename_safe_dt}{out_file_type.value}")
        else:
            fname = None
        
        dfs  = []
        cols = None
        for path, idx_col_nm in zip(df_sources, idx_columns):
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
            df_with_rectime = data_calc.add_recording_datetime(df_raw)
            df = data_calc._add_trig_cols(df_with_rectime, 'rec_datetime')  
            df.reset_index(drop=True, inplace=True)
        else:
            df = df_raw
 
        if fname is not None:
            df.to_csv(fname, index=include_index)
            self.log.info(f"Concatenated {len(df_sources)} bat measures files with total of {len(df)} rows (chirps) to {fname}")

        res_dict = {'df' : df, 'out_file' : fname}
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
    # pca_report
    #-------------------
    
    def pca_report(self, pca_result=None, pca_fname=None, pca_weights_matrix=None, pca_weights_matrix_fname=None):
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
                
                      importance_analysis_{pca_timestamp}.csv   

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
           
           
        For input, must provide either the pca_result object, or the pca_fname 
        where the pca is stored. 
           
        :param pca_result: PCA result object
        :type pca_result: union[None, sklearn.PCA
        :param pca_fname: file where PCA object is stored
        :type pca_fname: union[None, str]
        :param pca_weights_matrix: the weight matrix df computed by the PCA
        :type pca_weights_matrix: optional[str]
        :param pca_weights_matrix_fname: file where pca's weights matrix df is stored 
        :type pca_weights_matrix_fname: union[None, str]
        '''
        
        
        # Get the PCA:
        if pca_result is None:
            if pca_fname is None:
                raise ValueError("Provide either pca__result object, or pca_fname where pca object was saved")
            pca_result = DataCalcs.load_pca(pca_fname)
            
        # Get the weight matrix
        if pca_weights_matrix is None:
            if pca_weights_matrix_fname is None:
                raise ValueError("Provide either pca_weights_matrix, or pca_weights_matrix_fname")
            with UniversalFd(pca_weights_matrix_fname, 'r') as fd:
                weights = fd.read()
        else:
            weights = pca_weights_matrix


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
        
        importance_analysis = pd.concat([max_importance_feature_per_component,
                                         max_weights,
                                         loadings,
                                         direction
                                         ],axis=1)

        # For the file name, use the same timestamp as was used for the PCA file:
        pca_timestamp = Utils.timestamp_from_datetime(pca_result.create_date)
        important_features_fname = f"importance_analysis_{pca_timestamp}.csv"
        importance_analysis.to_csv(os.path.join(self.dst_dir, important_features_fname))
        
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
        explain_amount.to_csv(os.path.join(self.dst_dir, cum_expl_fname))
        
        # Create a chart to show the increase in explained variance
        # as number of components is increased:
        
        fig = DataViz.simple_chart(explain_amount.cumPercExplained, 
                                   xlabel='Components Deployed', 
                                   ylabel='Total % Variance Explained' 
                                   )
        fig_fname = f"variance_explained_{pca_timestamp}.png"
        fig.savefig(os.path.join(self.dst_dir, fig_fname))
        
        self.log.info("Done PCA")
        
            
# ------------------------ Main ------------
if __name__ == '__main__':

    # PCA all data
    #action = Action.PCA
    #analysis = MeasuresAnalysis(action)
    #res = analysis.experiment_result
    
    interpretations = ResultInterpretations()
    # Get list of saved PCA object file names
    # in the destination dir:
    pca_files = Utils.find_file_by_timestamp(interpretations.dst_dir, 
                                             prefix='pca_', 
                                             suffix='.joblib',
                                             latest=True)
    if len(pca_files) == 0:
        print(f"No pca files in {interpretations.dst_dir}. Aborting")
        sys.exit(1)
    
    pca_fname = pca_files[0]
    # Get the timestamp when the PCA object was created:
    str_timestamp = Utils.extract_file_timestamp(pca_fname)
    
    # Get the weights matrix that was saved along with
    # the PCA object (i.e. that has the same timestamp):    
    weights_fnames = Utils.find_file_by_timestamp(interpretations.dst_dir,
                                                  timestamp=str_timestamp, 
                                                  prefix='pca_weights_',
                                                  latest=True
                                                  )

    if len(weights_fnames) == 0:
        print(f"No pca weight matrix in {interpretations.dst_dir}. Aborting")
        sys.exit(1)
                
    pca_weights_matrix_fname = weights_fnames[0]
    
    pca_exploration = interpretations.pca_report(pca_fname=pca_fname,
                                                 pca_weights_matrix_fname=pca_weights_matrix_fname)
    
