'''
Created on Jul 11, 2024

@author: paepcke

Compute regression performance for transformers predicting
the SonoBat measures of the 34 variables that explain the 
most variance.

NOTE: 'transformer' and 'model' are used interchangeably

Main class is : ModelPerformance

To run (already done in __main__ section)
 ***** Test save and restore!!!!

    pred_analyzer = PredictionAnalyzer()
    
    # Grab transformer prediction results from files:
    pred_analyzer.import_predictions(timestamp='2024-06-25T12_55_03')
    
    # Compute all performance results, and save them
    # to Localization.prediction_performance_dir
    pred_analyzer.analyze_transformer_outputs()

To use a finished instance:
 
pred_analyzer.performance_scaled.model_means(): RMSE 
numbers are percentages of 1SD (?)

	                 RMSE_scaled_model_mean  ...  max_abs_diff_scaled_model_mean
	measure                                  ...                                
	TimeInFile                     0.826222  ...                        5.140086
	PrecedingIntrvl                1.870262  ...                       62.416629
	HiFreq                         0.384448  ...                        4.564543
	Bndwd                          0.676643  ...                        6.965542
	FreqMaxPw                      0.407437  ...                        3.203328
	             ...
	  
pred_analyzer.performance_descaled.model_means():

	                 RMSE_descaled_model_mean  ...  max_abs_diff_descaled_model_mean
	measure                                    ...                                  
	TimeInFile                   3.171576e+03  ...                      6.082238e+03
	PrecedingIntrvl              1.894670e+02  ...                      3.892119e+03
	HiFreq                       2.323248e+01  ...                      8.512715e+01
	Bndwd                        1.608922e+01  ...                      6.311623e+01
	FreqMaxPw                    1.566956e+01  ...                      4.854528e+01
                         ...		

More disaggregated, results for all models individually:

self.performance_descaled.RMSE
                    bats_transformer_seed_21.ckpt.log  ...  bats_transformer_seed_60.ckpt.log
    measure                                             ...                                   
    TimeInFile                            3.185921e+03  ...                       3.145835e+03
    PrecedingIntrvl                       1.972092e+02  ...                       1.913769e+02
    HiFreq                                2.311183e+01  ...                       2.322837e+01
    Bndwd                                 1.593117e+01  ...                       1.609169e+01
    FreqMaxPw                             1.563338e+01  ...                       1.569473e+01
                         ...	

self.performance_descaled.min_abs_diff
	                 bats_transformer_seed_21.ckpt.log  ...  bats_transformer_seed_60.ckpt.log
	measure                                             ...                                   
	TimeInFile                                0.179887  ...                           0.141861
	PrecedingIntrvl                           0.007380  ...                           0.001235
	HiFreq                                    0.001599  ...                           0.000146
	Bndwd                                     0.001052  ...                           0.000071
	FreqMaxPw                                 0.000534  ...                           0.000095
	PrcntMaxAmpD                              0.000394  ...                           0.000280
	FreqKnee                                  0.000242  ...                           0.000736
                         ...	

self.performance_descaled.max_abs_diff
	                 bats_transformer_seed_21.ckpt.log  ...  bats_transformer_seed_60.ckpt.log
	measure                                             ...                                   
	TimeInFile                            6.115768e+03  ...                       5.834503e+03
	PrecedingIntrvl                       3.883725e+03  ...                       3.881582e+03
	HiFreq                                8.511811e+01  ...                       8.490618e+01
	Bndwd                                 6.467831e+01  ...                       6.208690e+01
	FreqMaxPw                             4.776356e+01  ...                       4.739970e+01
                          ...	
	                       
'''

from data_calcs.data_calculations import (
    Localization,
    FileType,
    DataCalcs)
from data_calcs.data_cleaning import (
    DataCleaner)
from data_calcs.measures_analysis import (
    MeasuresAnalysis,
    Action)
from data_calcs.universal_fd import (
    UniversalFd)
from data_calcs.utils import (
    Utils)
from logging_service.logging_service import (
    LoggingService)
from sklearn.metrics import (
    root_mean_squared_error)
from sklearn.preprocessing._data import (
    StandardScaler)
import joblib
import os
import pandas as pd
import re
from numba.core.types import none

# -------------------------- ModelPerformance ------------

class ModelPerformance:
    '''
    Contains performance measures for one transformer.
    An instance holds for each variable:
        RMSE, 
        mean_abs_diff, 
        min_abs_diff, 
        max_abs_diff, 
        mean_perc_abs_diff
    
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, which, timestamp):
        '''
        :param timestamp: timestamp to use with generated files
        :type timestamp: str
        :param init_all: 
        '''

        self.log = LoggingService()
                
        if which not in ['scaled', 'descaled']:
            raise ValueError(f"Argument 'which' must be 'scaled', or 'descaled', not {which}")
        
        self.which = which
        self.timestamp = timestamp
        self.res_each_model = {}
        
        return
    
    #------------------------------------
    # compute_all
    #-------------------
    
    def compute_all(self):
        '''
        Initializes self.res_each_model, which is a dict mapping
        model_ids to dataframes that contain:
        
                model_id1: 
                         var1        rmse_var1   min_abs_diff_var1    max_abs_diff_var1
                         var2        rmse_var2   min_abs_diff_var2    max_abs_diff_var2
                                         ...
                model_id2: 
                         var1        rmse_var1   min_abs_diff_var1    max_abs_diff_var1
                         var2        rmse_var2   min_abs_diff_var2    max_abs_diff_var2
                                         ...
                                   ...                         
        
        Also initializes self.means_across_all_models:
        
	                       RMSE_scaled_model_mean  ...  max_abs_diff_scaled_model_mean
	        measure                                  ...                                
	        TimeInFile                     0.826222  ...                             0.0
	        PrecedingIntrvl                1.870262  ...                             0.0
	        HiFreq                         0.384448  ...                             0.0
	        Bndwd                          0.676643  ...                             0.0
	        FreqMaxPw                      0.407437  ...                             0.0
                        ...
                                       
        :param which: whether to compute all over scaled, or descaled
        :type which: str['scaled' | 'descaled']
        '''
                         
        # Iterate through all the prediction files. Each
        # iteration gets a composite of the true values, and
        # the predicted ones of one model:
        #             TimeInFile_truth  TimeInFile_pred...  chirp_idx
        #     0                  769.0  ... 780.0   	        4
        #     1                  944.0  ... 940.0   	        5
        #     2                 1042.0  ... 1045.0  	        6
        #     3                 1207.0  ... 1200.0  	        7
        #           ...                    ...
        
        # Each cols_matcher is a dict mapping truth col names to 
        # their respective prediction col names in the match df: 
        #    {'TimeInFile_truth': 'TimeInFile_pred', 
        #     'PrecedingIntrvl_truth': 'PrecedingIntrvl_pred',
        #                  ...
        #    }    
        for df_matched, cols_matcher in MatchedDataIterator(self.which, self.timestamp):
            
            # The model id is constant within one prediction file,
            # so, just grab the first one:
            model_id = df_matched['model_id'].iloc[0]
            self.log.info(f"Computing RMSE, etc. {self.which} for model {model_id}")
            # Create a dataframe:
            #            RMSE, min_abs_diff, max_abs_diff
            #     var1   
            #     var2        ...
            #     ...
            
            # Get a Series of RMSE results, one for each model:
            #      RMSE_ser
            #      TimeInFile         0.817667
            #      PrecedingIntrvl    1.871458
            #      HiFreq             0.385425
            #      Bndwd              0.676340
            #      FreqMaxPw          0.407145
            #                ...
            #     Name: RMSE_scaled, dtype: float64
            
            # Note the the series name includes whether the
            #      result comes from the scaled or unscaled
            #      truth and prediction sources.
                
            RMSE_ser          = self.compute_RMSE(df_matched, cols_matcher)
            RMSE_ser.name     = f"RMSE_{self.which}"

            # Same for min_abs_diff...            
            min_abs_diff_ser  = self.compute_min_abs_diff(df_matched, cols_matcher)
            min_abs_diff_ser.name=f"min_abs_diff_{self.which}"

            # and max_abs_diff:
            max_abs_diff_ser  = self.compute_max_abs_diff(df_matched, cols_matcher)
            max_abs_diff_ser.name=f"max_abs_diff_{self.which}"

            performance = pd.DataFrame([RMSE_ser, min_abs_diff_ser, max_abs_diff_ser]).transpose()
            performance.index.name = 'measure'
            
            self.res_each_model[model_id] = performance.copy()
            
        # Find the min and max model number (the int in model ids)
        # of what we found. Used in __repr__():
        model_nums = [PredictionAnalyzer.model_id_num(model_id)
                      for model_id 
                      in self.res_each_model.keys()
                      ]
        self.min_model_num = min(model_nums)
        self.max_model_num = max(model_nums)

        #                RMSE_scaled_model_mean  ...  max_abs_diff_scaled_model_mean
        # measure                                  ...                                
        # TimeInFile                     0.826222  ...                             0.0
        # PrecedingIntrvl                1.870262  ...                             0.0
        # HiFreq                         0.384448  ...                             0.0
        # Bndwd                          0.676643  ...                             0.0
        # FreqMaxPw                      0.407437  ...                             0.0
        #                 ...

        self.means_across_all_models = self.model_means()

    #------------------------------------
    # save
    #-------------------
    
    def save(self, dst_dir=None):
        
        if dst_dir is None:
            dst_dir = Localization.prediction_performance_dir
            
        all_models_mean_fname = f"performance_means_{self.which}_{self.timestamp}.csv"
        self.means_across_all_models.to_csv(os.path.join(dst_dir, all_models_mean_fname))
        for model_id, perf_df in self.res_each_model.items():
            fname = f"perf_model_{self.which}_{model_id}_{self.timestamp}.csv"
            fpath = os.path.join(dst_dir, fname)
            perf_df.to_csv(fpath)
        
    #------------------------------------
    # load
    #-------------------
    
    @classmethod
    def load(cls, which, timestamp, src_dir=None):

        self = ModelPerformance(which, timestamp)

        # Note: the model numbers (21-60) are harcoded here;
        #       better to put those numbers more globally,
        #       or (even better) discovered:
        
        self.min_model_num = 21
        self.max_model_num = 60
        
        # The number range of model names:
        model_num_range = range(self.min_model_num, 1+self.max_model_num)
        
        if src_dir is None:
            src_dir = Localization.prediction_performance_dir

        self.res_each_model = {}
            
        all_models_mean_fname = f"performance_means_{self.which}_{self.timestamp}.csv"
        self.log.info(f"Loading overall mean performance result from {all_models_mean_fname} in {src_dir}")
        all_models_mean_fpath = os.path.join(src_dir, all_models_mean_fname)
        self.means_across_all_models = pd.read_csv(all_models_mean_fpath)
        self.means_across_all_models.set_index('measure', drop=True, inplace=True)

        # Pattern to pull model id from filename:
        pat = re.compile(fr"perf_model_{which}_bats_transformer_seed_([^.]*).*")
        
        for fname in Utils.filename_iterator(src_dir, 
                                             prefix=f"perf_model_{self.which}_bats_transformer_seed_", 
                                             num_range=model_num_range, # Output model numbers 
                                             suffix=f".ckpt.log_{self.timestamp}.csv" 
                                             ):
            # Extract the model ID:
            match = pat.search(fname)
            if match is None:
                raise FileNotFoundError(f"Cannot extract model id from file '{fname}'")
            model_id = int(match.group(1))
            fpath = os.path.join(src_dir, fname)
            self.log.info(f"Loading performance result of model {model_id} from {fname} in {src_dir}")
            model = pd.read_csv(fpath)
            model.set_index('measure', drop=True, inplace=True)
            self.res_each_model[model_id] = model

        # Initialize self.pred_truth_descaled_df:
        self.pred_truth_descaled_df = pd.read_feather(Localization.prediction_truth_descaled)
        # Init 
        return self

    #------------------------------------
    # RMSE
    #-------------------
    
    @property 
    def RMSE(self):
        '''
        Return a DataFrame with the RMSEs of all models
        for each variable:
           
        '''
        col_nm = f"RMSE_{self.which}"
        # Make a df whose cols are the RMSEs of vars
        # in one model. Columns are model_id values:
        model_ids = self.res_each_model.keys()
        perf_series = [self.res_each_model[model_id][col_nm]
                       for model_id 
                       in model_ids
                       ]
        df_with_all = pd.DataFrame(perf_series).T 
        df_with_all.columns = model_ids
        
        return df_with_all

    #------------------------------------
    # min_abs_diff
    #-------------------
    
    @property 
    def min_abs_diff(self):
        col_nm = f"min_abs_diff_{self.which}"
        # Make a df whose cols are the RMSEs of vars
        # in one model. Columns are model_id values:
        model_ids = self.res_each_model.keys()
        perf_series = [self.res_each_model[model_id][col_nm]
                       for model_id 
                       in model_ids
                       ]
        df_with_all = pd.DataFrame(perf_series).T 
        df_with_all.columns = model_ids
        
        return df_with_all

    #------------------------------------
    # max_abs_diff
    #-------------------
    
    @property 
    def max_abs_diff(self):
        col_nm = f"max_abs_diff_{self.which}"
        # Make a df whose cols are the RMSEs of vars
        # in one model. Columns are model_id values:
        model_ids = self.res_each_model.keys()
        perf_series = [self.res_each_model[model_id][col_nm]
                       for model_id 
                       in model_ids
                       ]
        df_with_all = pd.DataFrame(perf_series).T 
        df_with_all.columns = model_ids
        
        return df_with_all

    #------------------------------------
    # model_means
    #-------------------
    
    def model_means(self):
        '''
        Compute the means of RMSE, max_abs_diff, etc over
        all models:
		                  RMSE_scaled_model_mean  ...  max_abs_diff_scaled_model_mean
		   measure                                  ...                                
		   TimeInFile                     0.826222  ...                             0.0
		   PrecedingIntrvl                1.870262  ...                             0.0
		   HiFreq                         0.384448  ...                             0.0
		   Bndwd                          0.676643  ...                             0.0
		   FreqMaxPw                      0.407437  ...                             0.0
		                   ...
		                   
		The column names will include 'scaled' (as above), or 'descaled'.
        '''
        
        # Compute a series:
        #
        #      var1  mean RMSE over all models
        #      var2  mean RMSE over all models
        #       ...
        # A list of RMSE series:
        col_nm = f'RMSE_{self.which}'
        
        rmses = [res[col_nm] for res in self.res_each_model.values()]
        rmses_df = pd.DataFrame(rmses).T
        model_ids = self.res_each_model.keys()
        rmses_df.columns = model_ids
        rmse_means = rmses_df.mean(axis='columns')
        rmse_means.name = f"{col_nm}_model_mean"

        # Same for the other performance measures:
        col_nm = f'min_abs_diff_{self.which}'
        
        minnies = [res[col_nm] for res in self.res_each_model.values()]
        minnies_df = pd.DataFrame(minnies).T
        model_ids = self.res_each_model.keys()
        minnies_df.columns = model_ids
        minnies_means = minnies_df.mean(axis='columns')
        minnies_means.name = f"{col_nm}_model_mean"

        col_nm = f'max_abs_diff_{self.which}'
        
        maxies = [res[col_nm] for res in self.res_each_model.values()]
        maxies_df = pd.DataFrame(maxies).T
        model_ids = self.res_each_model.keys()
        maxies_df.columns = model_ids
        maxies_means = maxies_df.mean(axis='columns')
        maxies_means.name = f"{col_nm}_model_mean"
        
        # Make a df from the means:
        means_df = pd.DataFrame([rmse_means, minnies_means, maxies_means]).T
        return means_df
        
    #------------------------------------
    # compute_RMSE
    #-------------------
    
    def compute_RMSE(self, df_matched, cols_matcher):
        '''
        Takes a df with columns for each predicted variable,
        both a truth and a predicted version:
        
                  var1_truth, var1_pred, var2_truth, var2_pred, ...
                  
        :param df_matched:
        :type df_matched:
        :param cols_matcher:
        :type cols_matcher:
        '''

        res = []
        for truth_col_nm in cols_matcher.keys():
            res.append(root_mean_squared_error(df_matched[truth_col_nm], 
                                               df_matched[cols_matcher[truth_col_nm]]
                                               ))
        ser_index = [truth_col_nm.strip('_truth') 
                     for truth_col_nm 
                     in cols_matcher.keys()
                     ]
        return pd.Series(res, index=ser_index)

    #------------------------------------
    # compute_min_abs_diff
    #-------------------
    
    def compute_min_abs_diff(self, df_matched, cols_matcher):

        res = []
        for truth_col_nm in cols_matcher.keys():
            res.append((df_matched[truth_col_nm] - df_matched[cols_matcher[truth_col_nm]]).abs().min())
        
        ser_index = [truth_col_nm.strip('_truth') 
                     for truth_col_nm 
                     in cols_matcher.keys()
                     ]
        return pd.Series(res, index=ser_index)

    #------------------------------------
    # compute_max_abs_diff
    #-------------------

    def compute_max_abs_diff(self, df_matched, cols_matcher):

        res = []
        for truth_col_nm in cols_matcher.keys():
            res.append((df_matched[truth_col_nm] - df_matched[cols_matcher[truth_col_nm]]).abs().max())
        
        ser_index = [truth_col_nm.strip('_truth') 
                     for truth_col_nm 
                     in cols_matcher.keys()
                     ]
        return pd.Series(res, index=ser_index)

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        the_str = f"<ModelPerformance (models {self.min_model_num}-{self.max_model_num}) at {hex(id(self))}>"
        return the_str

    #------------------------------------
    # __str__
    #-------------------
    
    def __str__(self):
        return self.__repr__()

# -------------------------- Class MatchedDataIterator ------------

class MatchedDataIterator:
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, which, timestamp):
        
        self.timestamp = timestamp
        if which == 'scaled':
            with UniversalFd(Localization.prediction_truth_scaled, 'r') as fd:
                self.df_truth = fd.asdf()
        elif which == 'descaled':
            with UniversalFd(Localization.prediction_truth_descaled, 'r') as fd:
                self.df_truth = fd.asdf()
        else:
            raise ValueError(f"The 'which' argument must be 'scaled', or 'descaled', not {which}")
        
        self.relevant_cols = DataCalcs.predicted_col_names + ['file_id', 'cntxt_sz']
        
        self.fname_iter = self.matched_data_iter(which)
        
    #------------------------------------
    # matched_data_iter
    #-------------------
    
    def matched_data_iter(self, which):
        
        if which == 'scaled':
            fname_iter = Utils.filename_iterator(Localization.predictions_scaled_dir, 
                                                      prefix='bats_transformer_seed_',
                                                      num_range=range(21,61), 
                                                      suffix='.ckpt.feather', 
                                                      )
        else:
            # Wants descaled data
            fname_iter = Utils.filename_iterator(Localization.predictions_descaled_dir,
                                                 prefix='predictions_descaled_',
                                                 num_range=range(21,61), 
                                                 suffix=f"_{self.timestamp}.feather", 
                                                 )
            
        return fname_iter

    #------------------------------------
    # __next__
    #-------------------
    
    def __next__(self):

        
        df_pred_fpath = next(self.fname_iter)
        with UniversalFd(df_pred_fpath, 'r') as fd:
            df_pred = fd.asdf()
             
        df_matched = pd.merge(self.df_truth[self.relevant_cols], 
                              df_pred, 
                              on=['file_id', 'cntxt_sz'], 
                              suffixes=['_truth', '_pred'])
        cols_matcher = {col_nm : col_nm.replace('_truth', '_pred')
                        for col_nm
                        in df_matched.columns 
                        if col_nm.endswith('_truth')
                        } 
        
        return (df_matched, cols_matcher)
    
    #------------------------------------
    # __iter__
    #-------------------
    
    def __iter__(self):
        return self

    

# -------------------------- Class PredictionAnalyzer ------------

class PredictionAnalyzer:
    '''
    Works with the transformer prediction files to see how
    well the transformers predict, and how their variances 
    differ for the same input.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self):
        '''
        Constructor
        '''
        self.log = LoggingService()
        
        self.pred_truth_descaled_df = None
        self.preds_descaled_dfs     = None
        self.scaled_fnames          = None
        self.descaled_fnames        = None
        
        self.performance_scaled   = none
        self.performance_descaled = none

    #------------------------------------
    # load
    #-------------------
    
    @classmethod
    def load(cls, timestamp, src_dir=None):
        '''
        Creates a PredictionAnalyzer instance with all
        performance measures in place, i.e. self.performance_scaled,
        and self.performance_descaled
        
        :param timestamp: timestamp of workflow (encoded in all related file names)
        :type timestamp: str
        :param src_dir: directory where prediction results are stored
        :type src_dir: optional[str]
        :return a fully initialized instance
        :rtype PredictionAnalyzer
        '''
        self = PredictionAnalyzer()
        self.performance_scaled = ModelPerformance.load(which='scaled', timestamp=timestamp, src_dir=src_dir)
        self.performance_descaled = ModelPerformance.load(which='descaled', timestamp=timestamp, src_dir=src_dir)
        
        # Set self.scaled_fnames self.descaled_fnames: 
        self._gather_prediction_files()

        return self

    #------------------------------------
    # analyze_transformer_outputs
    #-------------------
    
    def analyze_transformer_outputs(self):
        '''
        Computes RMSE, min_abs_diff, and max_abs_diff
        for each transformer's output, for each of their
        variables. Creates
         
                   self.performances,
                    
        which is a list of of ModelPerformance instances, 
        one for each model.
        
        Also creates the aggregate:
        
                   self.mean_performance,
        
        which is a Series, whose index are the SonoBat variables,
        and whose colums are MeanRMSE, MeanMinAbsDiff, and MeanMaxAbsDiff.
        Those values are means across all transformers.
        '''
        
        self.performance_scaled   = ModelPerformance('scaled', self.timestamp)
        self.performance_scaled.compute_all()
        self.performance_descaled = ModelPerformance('descaled', self.timestamp)
        self.performance_descaled.compute_all()
        
        self.performance_scaled.save()
        self.performance_descaled.save()
        
    #------------------------------------
    # import_predictions
    #-------------------
    
    def import_predictions(self, timestamp=None):
        '''
        Two main tasks:
           1. Create descaled versions of each scaled predition file,
              unless those versions already exist.
           2. Create one descaled file from the prediction
              toolchain's prediction_truth_values.feather file.
           3. Initialize the following instance vars: 

                    self.pred_truth_descaled_df   # One df truth from all models
                    self.scaled_fnames            # list of full paths to scaled fnames
                    self.descaled_fnames          # list of full paths to descaled fnames
                    self.preds_descaled_dfs       # list of descaled prediction dfs
                    
        Scaled transformer prediction files are expected to 
        be of the form bats_transformer_seed_42.ckpt.feather, 
        with nn being the transformer number.

        The generated files will all include a timestamp.
        For descaled predictions:
            predictions_descaled_{model_num}_{timestamp}.feather")
        and for the truth:
            prediction_truth_descaled_{timestamp}.feather
            
        like:
        
            predictions_descaled_62_2024-06-25T12_55_03.feather
        and
            prediction_truth_descaled_2024-06-25T12_55_03.feather
        
             
        in Localization.measures_root}/timestamp.txt
        
        NOTE: the transformer toolchain output files that are csv formatted inside,
              but have extension '.log'. The bash script <proj-root>/bash/rename_prediction_files.bash
              is used to rename them all to have .csv extension.
              The script  <proj-root>/bash/prediction_files_to_feather.bash is then used
              to create .feather files.
        
        The files look like:
        
        ,Unnamed: 0,TimeIndex,TimeInFile,PrecedingIntrvl,HiFreq,Bndwdth,...LdgToFcAmp,file_id,cntxt_sz,model_id
        0,0,1.702441,0.33496338,0.33018303,1.6207018,1.3135769,1.8231528,1.4286407,1.7472501,1.3470422,1.6221832,1.7523462,1.6188947,1.7112585,1.6973218,1.7737527,
            1.8365705,1.1933901,1.8794438,1.8902324,1.5931046,1.5791124,0.6963413,1.660197,-0.038168818,-0.056345016,-0.043458775,-0.06680888,0.077796675,
            -0.080624774,13023.0,8.0,bats_transformer_seed_21.ckpt.log
        
        1,1,1.702441,1.7943119,0.5975144,1.5512433,1.5456356,1.6200428,1.6924435,1.6597524,1.4275432,1.5580229,1.6413761,1.5333059,1.6334242,1.6119692,1.5654896,1.6198841,
            1.4922111,1.4876903,1.3369187,1.5389187,1.6520755,0.9609698,1.6141444,-0.030378342,-0.080755904,-0.06160508,-0.04614529,0.07999599,-0.03201635,9483.0,
            13.0,bats_transformer_seed_21.ckpt.log
        
        2,2,1.702441,3.4016354,0.8019287,1.9089952,2.021113,1.709298,1.6764234,1.7276975,1.3758863,1.97589,1.6965662,1.970968,1.8022838,1.8223938,1.6510321,1.73663,
            2.0870585,1.5121526,1.5576175,1.5454394,1.5120811,0.96669364,1.6825845,-0.007453665,-0.041472033,-0.05017312,-0.025630653,0.02266039,-0.04213579,2261.0,
            21.0,bats_transformer_seed_21.ckpt.log
                    ...
        
        '''
        # Is there a file called timestamp.txt at the truth dir?
        # If so, it contains a timestamp string of when the split
        # (truth) files were created, and we use it in filenames that we
        # create as we concat the prediction files:
    
        if timestamp is None:
            # Find or make a timestamp:    
            timestamp_path = f"{Localization.measures_root}/timestamp.txt"
            if os.path.exists(timestamp_path):
                with open(timestamp_path, 'r') as fd:
                    timestamp = fd.read().strip()   
            else:
                timestamp = Utils.file_timestamp()
                
        self.timestamp = timestamp

        # Get Truth:
        # Do we have the descaled prediction truth cached?
        if os.path.exists(Localization.prediction_truth_descaled):
            self.log.info(f"Loading prediction truth descaled from {Localization.prediction_truth_descaled}")
            with UniversalFd(Localization.prediction_truth_descaled, 'r') as fd:
                self.pred_truth_descaled_df = fd.asdf() 
        else:
            # Load the truth values:
            self.log.info(f"Loading prediction truth from {Localization.prediction_truth_scaled}")
            with UniversalFd(Localization.prediction_truth_scaled, 'r') as fd:
                df_pred_truth = fd.asdf()
            self.log.info(f"Done saving.")
                
            self.pred_truth_descaled_df = self._descale_and_save(df_pred_truth,
                                                                 Localization.prediction_truth_scaler, 
                                                                 Localization.prediction_truth_descaled)

        # Get Predictions:

        # Set self.scaled_fnames self.descaled_fnames: 
        self._gather_prediction_files()
        
        # Descale scaled predictions if needed:
        if len(self.scaled_fnames) != len(self.descaled_fnames):
            # We need the scaler that handles just the actually
            # predicted columns of the truth data:
            if os.path.exists(Localization.prediction_output_scaler):
                output_scaler = joblib.load(Localization.prediction_output_scaler)
            else:
                # Create a scaler from the truth, but for
                # only the actually predicted columns:
                output_scaler = self._create_output_scaler(self.pred_truth_descaled_df)
                # Save it for future fast retrieval:
                self.log.info(f"Saving output scaler to {Localization.prediction_output_scaler}")
                joblib.dump(output_scaler, Localization.prediction_output_scaler)
            self.preds_descaled_dfs = self._descale_predictions(output_scaler, 
                                                                self.scaled_fnames, 
                                                                self.timestamp)
        else:
            # Lazy loading done when computing performance
            pass
            # Load the predictions, which are nicely saved in files:
            # self.preds_descaled_dfs = []
            # for fpath in self.descaled_fnames:
            #     self.log.info(f"Loading descaled prediction {Path(fpath).name} ({fpath})")
            #     with UniversalFd(fpath, 'r') as fd:
            #         self.preds_descaled_dfs.append(fd.asdf())

    #------------------------------------
    # _gather_prediction_files
    #-------------------
    
    def _gather_prediction_files(self):
        '''
        Set self.scaled_fnames and self.descaled_fnames to be lists
        of prediction files
        '''

        # Find all scaled prediction files to know how many
        # there are, and how many descaled ones should therefore
        # be available:
        self.scaled_fnames   = [os.path.join(Localization.predictions_scaled_dir, scaled_fname)
                                for scaled_fname in os.listdir(Localization.predictions_scaled_dir)
                                if scaled_fname.startswith('bats_transformer_seed_') and \
                                   scaled_fname.endswith('.feather')]
        # Descaled prediction files are of the form:
        #    bats_transformer_seed_descaled_63_2024-06-25T12_55_03.feather
        # only the model number (63 in this case) changes:
        self.descaled_fnames = [os.path.join(Localization.predictions_descaled_dir, descaled_fname)
                                for descaled_fname in os.listdir(Localization.predictions_descaled_dir)
                                if descaled_fname.startswith('predictions_descaled_') and \
                                   descaled_fname.endswith(f"_{self.timestamp}.feather")]


    #------------------------------------
    # model_id_num
    #-------------------
    
    @staticmethod
    def model_id_num(model_id):
        '''
        Given a model identifier string, extract and return
        the integer that is embeded in the name:
        Names are like bats_transformer_seed_42.ckpt.log
        
        :param model_id: the id from which to extract the int
        :type model_id: str
        :return the integer that is embedded in the model id
        :rtype int
        '''
        
        match = re.match(r'[^0-9]*([0-9]*).*', model_id)
        if match is None:
            raise ValueError(f"Could not find integer in model_id {model_id}")
        model_num = int(match.group(1))
        return model_num
            


    #------------------------------------
    # _create_output_scaler
    #-------------------
    
    def _create_output_scaler(self, truth_df_descaled):  # @DontTrace
        '''
        Extracts just the columns that are predicted
        by transformers from the given truth df, and 
        creates a new scaler fitted for just those cols.
        
        Returns the fitted scaler. 
        
        :param truth_df_descaled: descaled truth values for all columns
        :type truth_df_descaled:  pd.DataFrame
        :return a fitted scaler
        :rtype sklearn.preprocessing.StandardScaler
        '''
        
        df = truth_df_descaled[DataCalcs.predicted_col_names]
        scaler = StandardScaler()
        scaler.set_output(transform = "pandas")
        scaler.fit(df)
        return scaler
        
    #------------------------------------
    # _descale_predictions
    #-------------------

    def _descale_predictions(self, scaler, scaled_fnames, timestamp):
        '''
        Given a list of absolute paths to scaled 
        prediction files, load each file into a df, 
        descale that df, and append that descaled 
        prediction df to
        
             self.preds_descaled_dfs.  
        
        also write the df to predictions_descaled_{model_num}_descaled.feather
        in directory Localization.predictions_descaled_dir.
    
        :param scaler: scaler to use for descaling predictions
        :type scaler: sklearn.preprocessing.StandardScaler
        :param scaled_fnames: list of full paths to scaled
            predictions
        :type scaled_fnames: list[str]
        :param timestamp: timestamp to include in names 
            of generated files
        :type timestamp: str
        :return the list of descaled prediction dfs
        :rtype list[pd.DataFrame] 
        '''

        self.preds_descaled_dfs = []
        # Regex to get number from like bats_transformer_seed_52.ckpt.feather:
        predict_file_pat = re.compile(f"[^0-9]*([0-9]+)\\.ckpt\\.feather")

        for scaled_file in scaled_fnames:
            
            model_num = predict_file_pat.match(scaled_file).group(1)
            
            descaled_path = os.path.join(Localization.predictions_descaled_dir,
                                         f"predictions_descaled_{model_num}_{timestamp}.feather")
            
            self.log.info(f"Loading {scaled_file}... ")
            with UniversalFd(scaled_file, 'r') as fd:
                df_scaled = fd.asdf()
            
            # Clarify some datatypes:
            df_scaled.file_id  = df_scaled.file_id.astype(int)
            df_scaled.cntxt_sz = df_scaled.cntxt_sz.astype(int)
            #df_scaled.model_id = df_scaled.model_id.astype(str)
            
            # Get just the scaled cols that are floating pt nums.
            # The df.dtypes returns a Series whose index
            # are col names, and the values are data types:
            float_cols = df_scaled.dtypes.loc[df_scaled.dtypes == float].index
              
            # Descale this df:
            self.log.info(f"Descaling {len(float_cols)} scaled cols to {scaled_file}... ")
            df_descaled_raw = DataCleaner.recover_orig_from_scaled_data(
                scaler, 
                df_scaled[float_cols])
            
            # Add file_id, cntxt_sz, and model_id columns into the descaled data:
            df_descaled_raw['file_id'] = df_scaled.file_id
            df_descaled_raw['cntxt_sz'] = df_scaled.cntxt_sz
            df_descaled_raw['model_id'] = df_scaled.model_id
            
            # Clean the df, adding column chirp_idx, removing
            # other cols:
            df_descaled = self._clean_prediction_df(df_descaled_raw)
            self.preds_descaled_dfs.append(df_descaled)
            
            # Save this descaled output:
            self.log.info(f"Saving descaled df at {descaled_path}")
            df_descaled.to_feather(descaled_path)
            self.log.info(f"Done saving.")
        
        return self.preds_descaled_dfs

    #------------------------------------
    # _descale_and_save
    #-------------------
    
    def _descale_and_save(self, df, scaler, dst_path):
        '''
        The provided df is asssumed to be a dataframe of
        all scaled predictions, or of all scaled truth
        values. Descales the df, and saves it in dst_path.
        
        Returns the descaled df.
        
        :param df: dataframe to descale
        :type df: pd.DataFrame
        :param scaler: scaler to use for descaling
        :type scaler: sklearn.proprocessing.StandardScaler
        :param dst_path: destination of descaled result. Must
            be full path, including extension of .csv, .feather,
            or .csv.gz
        :type dst_path: str
        :return the descaled dataframe
        :rtype pd.DataFrame
        '''
        
        # Now descale the predictions, and save them:
        self.log.info(f"Descaling... ")
        df_descaled = DataCleaner.recover_orig_from_scaled_data(scaler, df)
        self.log.info(f"Saving descaled df at {dst_path}")
        with UniversalFd(dst_path, 'w') as fd:
            fd.write(df_descaled)
        self.log.info(f"Done saving.")
        return df_descaled

    #------------------------------------
    # _consolidate_predictions
    #-------------------
    
    def _consolidate_predictions(self):
        '''
        Reaches into the dir of the transformer predicton output
        files. Concatenates the files into a dataframe, and returns
        that dataframe.
        
        :returns dataframe of all transformers' predictions
        :rtype pd.DataFrame 
        '''
        
        predict_file_pat = re.compile(f"[^0-9]*([0-9]+)\\.ckpt\\.feather")
        predict_files = []
        for maybe_predict_fname in os.listdir(Localization.inference_root):
            match = predict_file_pat.search(maybe_predict_fname)
            if match is None:
                continue
            else:
                predict_files.append(os.path.join(Localization.inference_root, maybe_predict_fname))
        
        # Sort the split files by split number (which is taken
        # from the split file name):
        predict_files.sort(key=lambda fname: int(re.search(r"\d+", fname)[0]))
        
        # Is there a file called timestamp.txt at the truth dir?
        # If so, it contains a timestamp string of when the split
        # (truth) files were created, and we use it in filenames that we
        # create as we concat the prediction files:
        
        timestamp_path = f"{Localization.measures_root}/timestamp.txt"
        if os.path.exists(timestamp_path):
            with open(timestamp_path, 'r') as fd:
                timestamp = fd.read().strip()   
        else:
            timestamp = None
        
        ma = MeasuresAnalysis(action=Action.CONCAT, 
                              df_sources=predict_files, 
                              dst_dir=os.path.dirname(Localization.all_measures), 
                              prefix='prediction',
                              #******idx_columns='level_0',
                              out_file_type=FileType.FEATHER,
                              augment=False,
                              timestamp=timestamp
                              )
        df_pred = ma.experiment_result['df']
        # Clean the df, adding column chirp_idx, removing
        # other cols:
        df_pred = self._clean_prediction_df(df_pred)
        return df_pred

        
    #------------------------------------
    # _clean_prediction_df 
    #-------------------
    
    def _clean_prediction_df(self, df_pred):
        
        # Make col 'row_num', which mirrors the index
        # be the index, removing it as a col:
        
        df_pred.index.name = 'chirp_num'
        # There are two cols: "Unnamed: 0" and "Unnamed: 0.1",
        # which are a mystery. Delete one, and rename the other:

        # Add the chirp index within its sequence to ease
        # matching against ground truth data: The context size
        # is the number of chirps in the same sequence that are
        # used to predict the chirp being focused on in each
        # row. Since chirp indexing is zero-based, we just
        # replicate that col with the name chirp_idx:
        df_pred['chirp_idx'] = df_pred['cntxt_sz']
        
        # Make ints for chirp_idx and cntxt_sz:
        df_pred.chirp_idx = df_pred.chirp_idx.astype(int)
        df_pred.cntxt_sz = df_pred.cntxt_sz.astype(int)
        df_pred.file_id   = df_pred.file_id.astype(int)
        
        # The TimeIndex column is a constant (1.702441).
        # So remove that col:
        df_pred.drop(['TimeIndex'], axis='columns', inplace=True)
        
        return df_pred
        
# ------------------------ Main ------------
if __name__ == '__main__':

    # Load pre-computed transformer model performance results
    # from saved files:
    pred_analyzer = PredictionAnalyzer().load('2024-06-25T12_55_03')
    
    # ----------------------
    # Analyze all transformer model performance results from scratch:
    
    # pred_analyzer = PredictionAnalyzer()
    # pred_analyzer.import_predictions(timestamp='2024-06-25T12_55_03')
    # pred_analyzer.analyze_transformer_outputs()
    
    print('Done')
    
