'''
Created on Jul 11, 2024

@author: paepcke
'''

from data_calcs.data_calculations import (
    Localization,
    FileType, DataCalcs)
from data_calcs.data_cleaning import (
    DataCleaner)
from data_calcs.measures_analysis import (
    MeasuresAnalysis,
    Action)
from data_calcs.universal_fd import (
    UniversalFd)
from data_calcs.utils import (
    Utils)
import pandas as pd
from pathlib import (
    Path)
from sklearn.metrics import mean_squared_error
import os
import re
from logging_service.logging_service import LoggingService

# -------------------------- ModelPerformance ------------

class ModelPerformance:
    '''
    Contains performance measures for one transformer.
    An instance holds the RMSE for every continuous variable.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, model_id, df_truth, df_pred, var_names):
        
        self.df_truth  = df_truth
        self.df_pred   = df_pred
        self.var_names = var_names
        self.model_id  = model_id
        self.mean_RMSE = None
        
        # Dict whose keys are SonoBat variable names.
        # The values are that variable's RMSE value 
        self.rmse_dict = {}
        
    #------------------------------------
    # compute_RMSE
    #-------------------
    
    def compute_RMSE(self):

        for var_name in self.var_names:
            # The 'squared=False' causes the result 
            # be RMSE, instead of MSE:
            self.rmse_dict[var_name] = mean_squared_error(self.df_truth[var_name], 
                                                          self.df_pred[var_name], 
                                                          squared=False)

    #------------------------------------
    # mean_RMSE
    #-------------------
    
    def mean_RMSE(self):

        # Have it cached?
        if self.mean_RMSE is not None:
            return self.mean_RMSE
            
        # Have we computed the RMSEs yet?
        if len(self.rmse_dict) == 0:
            # Nope, do it now:
            self.compute_RMSE()
            
        # Compute the mean...
        self.mean_RMSE = pd.Series(self.rmse_dict.values()).mean()
        # ...cache it
        return self.mean_RMSE


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
        self.df_pred            	= None
        self.df_pred_descaled   	= None
        self.df_pred_truth      	= None
        self.df_pred_truth_descaled = None

    #------------------------------------
    # analyze_transformer_outputs
    #-------------------
    
    def analyze_transformer_outputs(self):
        
        performances = []
        sb_vars = DataCalcs.dataset_names
        
        # Partition the predictions into separate
        # dataframes for each model output:
        preds_grp = self.df_pred_descaled.groupby(by='model_id')
        
        # For each continuous var, compute
        # the Root Mean Square Error between
        # predicted and the true values:
        for model_pred_df in preds_grp:
            performances.append(ModelPerformance('foo', 
                                                 self.df_pred_truth_descaled, 
                                                 model_pred_df,
                                                 sb_vars
                                                 ))
         
        print('foo')
        
    #------------------------------------
    # import_predictions
    #-------------------
    
    def import_predictions(self):
        '''
        Create one DF from all (scaled) transformer prediction files.
        They are expected to be of the form bats_transformer_seed_42.ckpt.feather, 
        with nn being the transformer number.
        
        Two cases: (1) the transformer output files have been gathered into
                       one df before, and were saved.
                   (2) the output files were never gathered into one df before.
                   
        In case one, the location of the df is expected in Localization.predictions.
        We load it, and stick it into self.df_pred.
        
        In case two, we load all the constituent .feather files, do some cleanup,
        and save into Localization.predictions.
        
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
        
        Sets instance vars 
                    self.df_pred_descaled 
                    self.df_pred_truth_descaled
                    self.model_ids
        '''
        # Do we have a descaled version of the predictions in a file?
        if os.path.exists(Localization.predictions_descaled):
            self.log.info(f"Loading descaled prediction df from {Localization.predictions_descaled}")
            with UniversalFd(Localization.predictions_descaled, 'r') as fd:
                self.df_pred_descaled = fd.asdf()
        elif os.path.exists(Localization.predictions):
            # We at least have one consolidated file of all the 
            # (scaled) predictions:
            with UniversalFd(Localization.predictions, 'r') as fd:
                df_pred = fd.asdf()
            self.df_pred_descaled = self._descale_and_save(df_pred, Localization.predictions_descaled)
        else:
            # No, we do not have a descaled version, nor do we
            # have a consolidated file of all predictions:        
            # Have to collect the files from their staging dir:        
            df_pred = self._consolidate_predictions()                        
            self.df_pred_descaled = self._descale_and_save(df_pred, Localization.predictions_descaled)

        # Now we have self.pred_descaled. Next, get 
        # the descaled truth values:

        # Do we have the descaled prediction truth cached?
        if os.path.exists(Localization.prediction_truth_descaled):
            self.log.info(f"Loading prediction truth descaled from {Localization.prediction_truth}")
            with UniversalFd(Localization.prediction_truth_descaled, 'r') as fd:
                self.df_pred_truth_descaled = fd.asdf() 
        else:
            # Load the truth values:
            self.log.info(f"Loading prediction truth from {Localization.prediction_truth}")
            with UniversalFd(Localization.prediction_truth, 'r') as fd:
                df_pred_truth = fd.asdf()
                
            self.df_pred_truth_descaled = self._descale_and_save(df_pred_truth, Localization.prediction_truth_descaled)

        # List of model ids, i.e. transformers:
        self.log.info(f"Finding all model ids")
        self.model_ids = self.df_pred.model_id.unique()

    #------------------------------------
    # _descale_and_save
    #-------------------
    
    def _descale_and_save(self, df, dst_path):
        '''
        The provided df is asssumed to be a dataframe of
        all scaled predictions, or of all scaled truth
        values. Descales the df, and saves it in dst_path.
        
        Returns the descaled df.
        
        :param df: dataframe to descale
        :type df: pd.DataFrame
        :param dst_path: destination of descaled result. Must
            be full path, including extension of .csv, .feather,
            or .csv.gz
        :type dst_path: str
        :return the descaled dataframe
        :rtype pd.DataFrame
        '''
        
        # Now descale the predictions, and save them:
        self.log.info(f"Descaling... ")
        df_descaled = DataCleaner.recover_orig_from_scaled_data(
            Localization.prediction_scaler, df)
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
        
#***********                
#         ***** Bombs here:
#   File "/Users/paepcke/EclipseWorkspacesNew1/bats/src/transformer_prediction_calcs/prediction_analysis.py", line 295, in <module>
#     pred_analyzer.import_predictions()
#   File "/Users/paepcke/EclipseWorkspacesNew1/bats/src/transformer_prediction_calcs/prediction_analysis.py", line 225, in import_predictions
#     df_pred = self._clean_prediction_df(df_pred)
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/paepcke/EclipseWorkspacesNew1/bats/src/transformer_prediction_calcs/prediction_analysis.py", line 266, in _clean_prediction_df
#     df_pred.set_index('row_num', drop=True, inplace=True)
#   File "/Users/paepcke/anaconda3/envs/bats/lib/python3.12/site-packages/pandas/core/frame.py", line 6122, in set_index
#     raise KeyError(f"None of {missing} are in the columns")
# KeyError: "None of ['row_num'] are in the columns"
#***********        
        df_pred.set_index('row_num', drop=True, inplace=True)
        # There are two cols: "Unnamed: 0" and "Unnamed: 0.1",
        # which are a mystery. Delete one, and rename the other:
        df_pred.drop(['Unnamed: 0.1'], axis=1, inplace=True)
        df_pred.rename(columns={'Unnamed: 0' : 'MysteryCol'}, inplace=True)
        
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
    pred_analyzer.import_predictions()
    pred_analyzer.analyze_transformer_outputs()
    
