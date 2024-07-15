'''
Created on Jul 9, 2024

@author: paepcke
'''

from logging_service.logging_service import (
    LoggingService)
import pandas as pd

class TableauPrepper:
    '''
    classdocs
    '''

    measure_types = {}
    measure_types['exp_fit'] = [
        'HiFtoFcAmp', 'HiFtoFcDmp', 'KnToFcAmp', 'KnToFcDmp', 'HiFtoKnExpAmp',
        'HiFtoKnDmp', 'HiFtoUpprKnAmp', 'HiFtoUpprKnExp', 'HiFtoKnAmp','HiFtoKnExp', 
        'HiFtoFcExp', 'UpprKnToKnAmp', 'UpprKnToKnExp', 'KnToFcExp', 'LdgToFcAmp',
        'LdgToFcExp', 'LnExpA_StartAmp', 'LnExpB_StartAmp', 'AmpStartLn60ExpC', 'LnExpA_EndAmp',
        'LnExpB_EndAmp', 'AmpEndLn60ExpC', 'PreFc250Residue', 'PreFc500Residue', 'PreFc1000Residue',
        'PreFc3000Residue', 'KneeToFcResidue', 'HiFtoFcExpAmp', 'KnToFcExpAmp',
        'AmpStartLn60ExpC',
        ]
    
    measure_types['amplitudes'] = [
        'PrcntMaxAmpDur', 'Amp1stQrtl', 'Amp2ndQrtl', 'Amp3rdQrtl', 'Amp4thQrtl',
        'Amp1stMean', 'Amp2ndMean', 'Amp3rdMean', 'Amp4thMean', 'AmpVariance',
        'AmpMoment', 'AmpGausR2', 
        ]  
    
    measure_types['frequencies'] = [
        'FreqMaxPwr', 'FreqCtr', 'FFwd32dB', 'FBak32dB', 'FFwd20dB',
        'FBak20dB', 'FFwd15dB', 'FBak15dB', 'FFwd5dB','FBak5dB',
        'PreFc250', 'PreFc500','PreFc1000', 'PreFc3000', 'StartF',
        'HiFreq','UpprKnFreq', 'FreqKnee', 'FreqLedge', 'Fc',
        'EndF','LowFreq', 'RelPwr2ndTo1st', 'RelPwr3rdTo1st', 'HiFminusStartF',
         
        ]
    measure_types['bandwidths'] = [
        'Bndwdth', 'Bndw32dB', 'Bndw20dB', 'Bndw15dB', 'Bndw5dB',
        'FcMinusEndF',  
        ]
    
    measure_types['durations'] = [
        'TimeFromMaxToFc', 'DurOf32dB', 'DurOf20dB', 'DurOf15dB', 'DurOf5dB',
        'PrcntMaxAmpDur', 'PrecedingIntrvl', 'PrcntKneeDur', 'CallsPerSec',
        'CallDuration', 'KnToFcDur', 
        ]
    
    measure_types['slopes'] = [
        'AmpK@start', 'AmpK@end', 'AmpKurtosis', 'AmpSkew', 'EndSlope',
        'SteepestSlope','StartSlope', 'HiFtoUpprKnSlp', 'HiFtoKnSlope','DominantSlope',
        'KneeToFcSlope','TotalSlope', 'CummNmlzdSlp', 'SlopeAtFc','LdgToFcSlp',
        'LowestSlope',  'Kn-FcCurvinessTrndSlp',
        ]
    
    measure_types['knee_ledges'] = [
        'LedgeDuration', 'LedgeDuration', 'Kn-FcCurviness', 'Kn-FcCurvinessTrndSlp',
        'meanKn-FcCurviness', 'KnToFcAmp', 'KnToFcDmp', 'HiFtoKnExpAmp', 'UpprKnToKnAmp', 
        'UpprKnToKnExp', 'KnToFcExp', 'LdgToFcAmp', 'LdgToFcExp', 'KneeToFcResidue', 
        'KnToFcExpAmp', 'UpprKnFreq', 'FreqKnee', 'FreqLedge', 'PrcntKneeDur', 'KnToFcDur',
        'HiFtoUpprKnSlp', 'HiFtoKnSlope', 'KneeToFcSlope' 
        ]
    
    measure_types['quality'] = [
        'Quality', 'MinAccpQuality','Max#CallsConsidered',
        ]
    
    #------------------------------------
    # Constructor
    #-------------------


    def __init__(self, df):
        '''
        Constructor
        '''
        
        self.log = LoggingService()
        self.df = df
        df_rows = []
        for measure_type, measure_cols in self.measure_types.items(): 
            df_rows = self.add_level(df_rows, measure_type, measure_cols)

        res_df = pd.concat(df_rows, axis='rows')
        #*******res_df = res_df.reset_index(drop=True)
        res_df.index.name = 'chirp_num'
        
        self.df = res_df
        
    #------------------------------------
    # add_level
    #-------------------
    
    def add_level(self, df_rows, level_nm, measure_nms):
        
        df_list = []
        
        for measure_nm in measure_nms:
            # Get the level name column ('Amplitude', 'Frequency', ...):
            level_nm_ser   = pd.Series([level_nm]*len(self.df))
            measure_nm_ser = pd.Series([measure_nm]*len(self.df))
            try:
                df_list.append(pd.DataFrame({'Measure_Type'  : level_nm_ser, 
                                             'Measure_Name'  : measure_nm_ser, 
                                             'Measure_Value' : self.df[measure_nm]}))
                                            
            except KeyError:
                self.log.err(f"While adding level {level_nm}, measure {measure_nm}: measure column missing in df.")
                continue
        
        df_rows.extend(df_list)                
        return df_rows
            