'''
Created on Jul 11, 2024

@author: paepcke

NOTE: 'transformer' and 'model' are used interchangeably
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
from pathlib import (
    Path)
from sklearn.metrics import (
    mean_squared_error)
from sklearn.preprocessing._data import (
    StandardScaler)
import joblib
import os
import pandas as pd
import re

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
    
    def __init__(self, model_id, df_truth, df_pred):
        
        self.df_truth  = df_truth
        self.df_pred   = df_pred
        self.model_id  = model_id
        self.mean_RMSE = None
        self.var_names = DataCalcs.predicted_col_names
        
        relevant_cols = self.var_names + ['file_id', 'cntxt_sz']
        df_matched = pd.merge(df_truth[relevant_cols], 
                              df_pred, 
                              on=['file_id', 'cntxt_sz'], 
                              suffixes=['_truth', '_pred'])
        cols_matcher = {col_nm : col_nm.replace('_truth', '_pred')
                        for col_nm
                        in df_matched.columns 
                        if col_nm.endswith('_truth')
                        } 
        
        # Create a dataframe:
        #            RMSE, min_abs_diff, max_abs_diff
        #     var1   
        #     var2        ...
        #     ...
        self.performance = pd.DataFrame([
            self.compute_RMSE(df_matched, cols_matcher),
            self.compute_min_abs_diff(),
            self.compute_max_abs_diff()
            ], index=self.var_names)
        self.performance.index.name = 'variable'

    #------------------------------------
    # RMSE
    #-------------------
    
    @property 
    def RMSE(self):
        return self.performance['RMSE']

    #------------------------------------
    # min_abs_diff
    #-------------------
    
    @property 
    def min_abs_diff(self):
        return self.performance['min_abs_diff']

    #------------------------------------
    # max_abs_diff
    #-------------------
    
    @property 
    def max_abs_diff(self):
        return self.performance['max_abs_diff']
        
    #------------------------------------
    # compute_RMSE
    #-------------------
    
    def compute_RMSE(self, df_matched, cols_matcher):

        res = []
        for truth_col_nm in cols_matcher.keys():
            # The 'squared=False' causes the result 
            # be RMSE, instead of MSE:
            res.append(mean_squared_error(df_matched[truth_col_nm], 
                                          df_matched[cols_matcher[truth_col_nm]], 
                                          squared=False))
        return pd.Series(res, index=cols_matcher.keys(), name='RMSE')

    #------------------------------------
    # compute_min_abs_diff
    #-------------------
    
    def compute_min_abs_diff(self):

        res = []
        for var_name in self.var_names:
            res.append((self.df_pred[var_name] - self.df_truth[var_name]).abs().min())
        return pd.Series(res, index=self.var_names, name='min_abs_diff')

    #------------------------------------
    # compute_max_abs_diff
    #-------------------
    
    def compute_max_abs_diff(self):

        res = []
        for var_name in self.var_names:
            res.append((self.df_pred[var_name] - self.df_truth[var_name]).abs().max())
        return pd.Series(res, index=self.var_names, name='min_abs_diff')

# -------------------------- PredictionAnalyzer ------------
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
        
        self.performances = []
        for pred_descaled_df in self.preds_descaled_dfs:
            # The model id col in the descaled prediction
            # dfs is constant within one df:
            model_id = pred_descaled_df.model_id.iloc[0]
            self.performances.append(ModelPerformance(model_id, self.pred_truth_descaled_df, pred_descaled_df))
            
        print('foo')
        
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
                    self.preds_descaled_dfs # list of descaled prediction dfs
                    
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
                                   descaled_fname.endswith(f"_{timestamp}.feather")]
        
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
                                                                timestamp)
        else:
            # Load the predictions, which are nicely saved in files:
            self.preds_descaled_dfs = []
            #***********
            #for fpath in self.descaled_fnames:
            for fpath in self.descaled_fnames[:3]:
            #***********
                self.log.info(f"Loading descaled prediction {Path(fpath).name} ({fpath})")
                with UniversalFd(fpath, 'r') as fd:
                    self.preds_descaled_dfs.append(fd.asdf())

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
    
    pred_analyzer = PredictionAnalyzer()
    pred_analyzer.import_predictions(timestamp='2024-06-25T12_55_03')
    pred_analyzer.analyze_transformer_outputs()
    
    print('Done')
    
