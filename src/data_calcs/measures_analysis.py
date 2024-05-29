'''
Created on May 28, 2024

@author: paepcke
'''
from data_calcs.data_calculations import DataCalcs, PerplexitySearchResult, \
    ChirpIdSrc, FileType
from data_calcs.data_viz import DataViz
from data_calcs.universal_fd import UniversalFd
from data_calcs.utils import Utils
from enum import Enum
from logging_service.logging_service import LoggingService
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime
import os
import pandas as pd
import random
import shutil
import time

from data_calcs.data_calculations import Localization

# Members of the Action enum are passed to run
# to specify which task the program is to perform: 
class Action(Enum):
    HYPER_SEARCH  = 0
    PLOT          = 1
    ORGANIZE      = 2
    CLEAR_RESULTS = 3
    SAMPLE_CHIRPS = 4
    EXTRACT_COL   = 5
    CONCAT        = 6
    PCA           = 7


class MeasuresAnalysis:
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self):
        
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
        activities = Activities()
        
        # Task: obtain PCA information about all measures:
        
        # PCA returns a dict with keys 'pca', 'weight_matrix',
        # and 'xformed_data':        
        pca, weight_matrix, _xformed_data, pca_save_file = activities.run(
            Action.PCA,
            **{'df' : self.all_measures,
               'n_components' : len(self.all_measures.columns)
              }
            ).values()

        # Path where PCA was save is of the form:
        # <dir>/pca_20240528T161317.259849.joblib.
        # Get the timestamp:
        timestamp = Utils.extract_file_timestamp(pca_save_file)
        pca_path = Path(pca_save_file)
        num_in_features = pca.n_features_in_
        num_samples     = pca.n_samples_
        weight_fname = pca_path.parent.joinpath(f"pca_weights_{timestamp}_{num_in_features}features_{num_samples}samples.feather")
        weight_matrix.to_feather(weight_fname)
        
        
        print(pca)
    
        
    #------------------------------------
    # _mk_fpath_from_other
    #-------------------
    
    def _mk_fpath_from_other(self, 
                             other_fname, 
                             prefix='',
                             suffix='',
                             timestamp_from_other=False, 
                             **name_components):
        '''
        Given a filename with, or without a timestamp, a desired
        prefix and suffix (fname extension incl. leading period),
        create a new filename. The name_components is an optional
        dict containing as keys filename fragments, whose values
        are the number of those items are to be named in the 
        filename.
        
        Example:
            other_fname == my_file
        
        :param other_fname:
        :type other_fname:
        :param prefix:
        :type prefix:
        :param suffix:
        :type suffix:
        :param timestamp_from_other:
        :type timestamp_from_other:
        '''
        
        if not isinstance(other_fname, Path):
            other_fpath = Path(other_fname)
        else:
            other_fpath = other_fname
            
        if timestamp_from_other:
            timestamp = Utils.extract_file_timestamp(other_fname)
        else:
            timestamp = Utils.file_timestamp()

        fname = f"{prefix}{timestamp}"
        for name, quantity in name_components.items():
            fname += f"_{quantity}{name}"
        full_fpath = other_fpath.parent.join_path(f"{fname}{suffix}")
        return full_fpath
            
        
        
class Activities:

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self):
        
        self.log = LoggingService()
        self.res_file_prefix = 'perplexity_n_clusters_optimum'
        self.data_calcs = DataCalcs()
        self.fid_map_file = 'split_filename_to_id.csv'
        
        self.data_dir = '/tmp'
        
        
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
        filename_safe_dt = Utils.file_timestamp()
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
    # run
    #-------------------
    
    def run(self, actions, **action_specific_kwargs):
        '''
        Perform one task using class DataCalcs methods.
        Keyword arguments are passed to the executing 
        methods, if appropriate. Some of these kwargs 
        are specific to the invocation of a single Action, 
        others are relevant to every invocation of that Action.
        
        Example: for plot_search_results() we need to pass the
            search results to plot each time. But the destination
            directory for saving may be shared across many calls.
            
        The action run specific kwargs should be passed in to
        this method. This method obtains the fixed kwargs from
        _make_kwargs().
        
        The action that required specific kwargs passed into this
        method are:
        
            Action.SAMPLE_CHIRPS: num_samples
            Action.PLOT         : search_results
            Action.EXTRACT_COL  : df_src, 
                                  col_name
            Action.CONCAT       : df_sources
            Action.PCA          : df
                                  n_components


        :param actions: one or more Actions to perform. If
            multiple actions are specified, they are executed
            in order.
        :type actions: union[Action | list[Action]]
        '''

        if type(actions) != list:
            actions = [actions]

        # Get a list of kwarg dicts:
        action_kwargs = [self._make_kwargs(action) for action in actions]
        action_kwargs.append(action_specific_kwargs)
        merged_kwargs = {}
        
        for one_dict in action_kwargs:
            merged_kwargs.update(one_dict)

        for action in actions:
            if action == Action.ORGANIZE:
                return self.organize_results()
                
            elif action == Action.HYPER_SEARCH:
                # Returns PerplexitySearchResult instance:
                return self.hyper_parm_search(repeats=1)
                
            elif action == Action.CLEAR_RESULTS:
                self.remove_search_res_files()
                
            elif action == Action.PLOT:
                return self._plot_search_results(merged_kwargs)
                
            elif action == Action.SAMPLE_CHIRPS:
                # Possible kwarg: num_chirps, which is
                # the number of chirp.
                # Returns {'df' : x,  'save_file' : y}
                return self._sample_chirps(merged_kwargs)
                
            elif action == Action.EXTRACT_COL:
                # Returns dict {'col', 'out_file'}
                return self._extract_column(merged_kwargs)

            elif action == Action.CONCAT:
                # Returns {'df' : x, 'out_file': y}
                return self._concat_files(merged_kwargs)

            elif action == Action.PCA:
                # Returns {'pca' : x, 'weight_matrix' : y, 'transformed_data' : z, 'pca_save_file' : f}
                return DataCalcs().pca_computation(**merged_kwargs)
            else:
                raise ValueError(f"Unknown action: {action}")
            
    #------------------------------------
    # _make_kwargs
    #-------------------
    
    def _make_kwargs(self, action, **kwargs):
        '''
        
        Extra args:
            Action.SAMPLE_CHIRPS: num_samples
            Action.PLOT         : search_results
            Action.EXTRACT_COL  : df_src, 
                                  col_name
            Action.CONCAT       : df_sources
            Action.PCA          : df
                                  n_components
 
        :param action: action for which kwargs are to be constructed
        :type action: Action
        '''
        
        proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        if action in [Action.HYPER_SEARCH, Action.PLOT]:
            kws = {'dst_dir' : os.path.join(proj_dir, 'results/hyperparm_searches')}
            kws.update(kwargs)
        elif action == Action.ORGANIZE:
            kws = {}
        elif action == Action.SAMPLE_CHIRPS:
            chirp_samples = os.path.join(proj_dir, 'results/chirp_samples/'),
            # Add number of samples:
            kws = {'df_file'       : os.path.join(chirp_samples, '_50000_chirps_20240524T090814.107027.csv')}
            kws.update(kwargs)
        elif action == Action.EXTRACT_COL:
            kws = {'dst_dir' : os.path.join(proj_dir, 'results/chirp_samples'),
                   'prefix'  : 'col_extract_'
                   }
            kws.update(kwargs)
        elif action == Action.CONCAT:
            kws = {'prefix'     : 'concat_',
                   'dst_dir' : os.path.join(proj_dir, 'results/chirp_samples')}
            kws.update(kwargs)
        elif action == Action.PCA:
            # Have pca_dump() in data_calculations add the proper suffix:
            kws = {'dst_fname'    : os.path.join(proj_dir, f"results/chirp_samples/pca_{Utils.file_timestamp()}"), 
                   'columns'      : None}
            kws.update(kwargs)
        else:
            raise ValueError(f"Action is not recognized: {action}")
    
        return kws    
            
            
# ------------------------ Main ------------
if __name__ == '__main__':

    MeasuresAnalysis()
        