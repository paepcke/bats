
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import Signature

import argparse
import pandas as pd
import numpy as np
import os

import plotly.express as px
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


#------------------------------------
# create_scale_info
#-------------------

def create_scale_info(sig_values):
    '''
    Given sig_values, provide the scale_info used to normalize the sig_values.
    NOTE: Code take from test_signatures.py.

    :param sig_values: Computed metrics from .wav file, size (time_frames, num_metrics=4) 
    :type sig_vales: pd.DataFrame()
    :returns scale_info: used for normalization by Signature class
    :type scale_info: dict of dicts
    '''
    flatness   = sig_values.flatness
    continuity = sig_values.continuity
    pitch      = sig_values.pitch
    freq_mod   = sig_values.freq_mod

    flatness_mean   = flatness.mean()
    continuity_mean = continuity.mean()
    pitch_mean      = pitch.mean()
    freq_mod_mean   = freq_mod.mean()

    scale_info = {
        'flatness' : {'mean' : flatness_mean,
                        'standard_measure' : np.abs((flatness - flatness_mean).median()) 
                        },
        'continuity' : {'mean' : continuity_mean,
                        'standard_measure' : np.abs((continuity - continuity_mean).median()) 
                        },
        'pitch' : {'mean' : pitch_mean,
                    'standard_measure' : np.abs((pitch - pitch_mean).median()) 
                    },

        'freq_mod' : {'mean' : freq_mod_mean,
                    'standard_measure' : np.abs((freq_mod - freq_mod_mean).median())
                    }
        }

    return scale_info


#------------------------------------
# create_signatures
#-------------------

def create_signatures(dir):
    '''
    Give directory of .wav files, create Signature for each and save
    the sig_values as a .csv.

    :param dir: directory of .wav files
    :type dir: string
    :returns signatures: list of Signatures
    :type signatures: list of Signature class
    '''
    print('Creating signatures')
    signatures = []
    for audio in os.listdir(dir):
        fname = os.path.join(dir, audio) 
        sp_flat = SignalAnalyzer.spectral_flatness(spec_src=fname)
        sp_cont = SignalAnalyzer.spectral_continuity(audio=fname)
        pitch = SignalAnalyzer.harmonic_pitch(spec_src=fname)
        freq_mod = SignalAnalyzer.freq_modulations(spec_src=fname)
        species = fname[-8:-4]

        sig_values = pd.concat([sp_flat, sp_cont[1], pitch, freq_mod], axis=1)
        scale_info = create_scale_info(sig_values) 
        signature = Signature(species, sig_values, scale_info)
        signatures.append(signature)

        # saving sig_values for possible later use
        sig_values.to_csv(dir[:-1] + '_sig_values/' + fname[15:-4] + '.csv')    

    return signatures


#------------------------------------
# create_dataset_from_wav
#-------------------

def create_dataset_from_wav(dir):
    '''
    Given a directory of .wav, consolidate signature values
    into a 3D array for clustering.

    :param dir: directory of .wav file
    :type dir: string
    :returns X: signature data organized with size 
        (num_examples, time_frames * num_metrics=4)
    :type X: np.array
    :returns Y: labels for signature data organized with size
        (num_examples, 1) 
    :type Y: np.array
    '''
    print('Creating dataset')
    X, Y = np.array([]), np.array([])

    signatures = create_signatures(dir)
    X = np.array([signature.sig for signature in signatures])
    Y = np.array([signature.species for signature in signatures])

    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    return X, Y


#------------------------------------
# create_dataset_from_csv
#-------------------
def create_dataset_from_csv(dir):
    '''
    Given a directory of .csv files (sig_values), consolidate signature values
    into a 3D array for clustering.

    :param dir: directory of  .csv file
    :type dir: string
    :returns X: signature data organized with size 
        (num_examples, time_frames * num_metrics=4)
    :type X: np.array
    :returns Y: labels for signature data organized with size
        (num_examples, 1) 
    :type Y: np.array
    '''
    print('Creating dataset')
    X, Y = [], []

    for csv in os.listdir(dir):
        fname = os.path.join(dir, csv) 
        X.append(pd.read_csv(fname)['pitch'])
        Y.append(fname[-8:-4])

    X, Y = np.array(X), np.array(Y)
    X = X.reshape((X.shape[0], X.shape[1]))
    return X, Y


#------------------------------------
# cluster_dataset
#-------------------

def cluster_dataset(X, Y):
    '''
    Given an array of sig_values from multiple recordings and an array
    of species labels, perform tSNE to cluster data and save plot.

    :param X: signature data organized with size 
        (num_examples, time_frames * num_metrics=4)
    :type X: np.array
    :param Y: labels for signature data organized with size
        (num_examples, 1) 
    :type Y: np.array
    '''
    X_embedded = TSNE(n_components=2,
                      learning_rate='auto',
                      init='random',
                      perplexity=5).fit_transform(X)
    fig = px.scatter(X_embedded, x=0, y=1, color=Y)
    fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('ftype')
    args = parser.parse_args()

    dir, ftype = args.dir, args.ftype

    X, Y = None, None
    if ftype == "wav":
        X, Y = create_dataset_from_wav(dir)
    if ftype == "csv":
        X, Y = create_dataset_from_csv(dir)

    cluster_dataset(X, Y)


if __name__ == "__main__":
    main()