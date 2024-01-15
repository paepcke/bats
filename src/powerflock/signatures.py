'''
Created on Dec 23, 2021

@author: paepcke
'''

import json
from pathlib import Path

from experiment_manager.experiment_manager import JsonDumpableMixin
import librosa

from data_augmentation.utils import Utils, Interval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SpectralTemplate(JsonDumpableMixin):
    '''
    Hold the spectral centroid timelines (signatures)
    of all calls in one recording.
    
    Usage:
        o <inst>.signatures    : list of Signature instances with the 
                                 frequency-of-max-energy timeline
        o iter(<inst>)         : return iterator over signatures
        o <inst>[n]            : return the nth signature
        o len(<inst>)          : number of signatures
        o <inst>.sig_lengths   : list of lengths of the signatures
        o <inst>.mean_sig      : signature that is the mean of all
                                 signatures' frequencies at each
                                 time.
        o <inst>.as_time(<signature>): sigs are pd.Series whose index are times
                                 as fractional secs. This method returns an array of 
                                 those times.
                                
    Also available as (read-only) properties:
        o sample_rate          : sample rate
        o hop_length           : samples between frames
        o n_fft                : related to number of frequency bands 
                                
    '''
    # ~88 time frames per second:
    hop_length = 256
    # 1025 frequency bins:
    n_fft = 2048
    sr = 22050
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 signatures, 
                 rec_fname=None,
                 sr=22050,
                 hop_length=256,
                 n_fft=2048
                 ):
        '''
        
        :param signatures: list of Signature instances
        :type signatures: [Signature]
        :param rec_fname: path to file from which signatures 
            were created
        :type rec_fname: str
        :param sr: sample rate with which sigs were created
        :type sr: int
        :param hop_length: spectrogram slide width
        :type hop_length: float
        :param n_fft: spectrogram window width
        :type n_fft: float
        '''

        # Save the hop length and frequency bin-count-relevant
        # parameters so that clients wanting to create signatures
        # for audio clips to match against this template will
        # create proper sigs:
        
        self._hop_length = hop_length
        self._n_fft      = n_fft
        
        self.signatures = signatures
        self.rec_fname = rec_fname
        self.sr = sr

        # Computed lazily:
        self.cached_mean_sig = None
        
        # About application of bandpass filtering
        # of test audio when matching its snippets to
        # any of this template's sigs: if any of the
        # sigs were created with a species-specific bandpass
        # filter applied, then create an Interval that
        # comprises the max bandwidth from among this template's
        # constitutent sigs:
        
        sigs_with_bandpass = sum([1 if sig.bandpass_filter is not None else 0
                                  for sig
                                  in self.signatures
                                  ])
        if sigs_with_bandpass > 0:
            self.bandpass_filter = self.frequency_band()
        else:
            self.bandpass_filter = None

    #------------------------------------
    # json_dumps
    #-------------------
    
    def json_dumps(self):
        '''
        Return a json string holding the contents
        of this SpectralTemplate instance, ready to json.dump() into
        a file. To create a new SpectralTemplate instance,
        pass the string to class method SpectralTemplate.json_loads()
        '''
        
        # Only save what we need, e.g. not the audio:
        recovery_dict = {

            "hop_length" : self.hop_length,
            "_n_fft"     : self._n_fft,
            "rec_fname"  : self.rec_fname,
            "sr"         : self.sr,
            "cached_mean_sig" : None,
            }  
        # Jsonize the list of Signature instances:
        json_sigs = [sig.json_dumps()
                     for sig
                     in self.signatures]

        recovery_dict['json_sig_list'] = json_sigs

        return json.dumps(recovery_dict)

    #------------------------------------
    # json_loads
    #-------------------
    
    @classmethod
    def json_loads(cls, jstr):
        '''
        Given a json formatted SpectralTemplate instance string
        return a materialized SpectralTemplate.

        :param jstr: unevaluated json string 
        :type str: {str : str}
        :return a new SpectralTemplate instance
        :rtype SpectralTemplate
        '''
        try: #*******template_jdict unexpectedly is a string!!!!
            template_jdict = json.loads(jstr)
            json_sigs_list = template_jdict['json_sig_list']
            signatures = [Signature.json_loads(json_sig)
                          for json_sig in json_sigs_list]
        except Exception as e:
            raise ValueError(f"Could not read signature list from string ({repr(e)})")

        materialized_inst = SpectralTemplate(
            signatures=signatures,
            rec_fname=template_jdict['rec_fname'],
            sr=template_jdict['sr'],
            hop_length=template_jdict['hop_length'],
            n_fft=template_jdict['_n_fft']
            )

        return materialized_inst

    #------------------------------------
    # json_dump
    #-------------------

    def json_dump(self, fname):
        '''
        write SpectralTemplate instance to
        file.
        
        :param fname: destination path
        :type fname: str
        '''
        
        with open(fname, 'w') as fd:
            #json.dump(self.json_dumps(), fd)
            fd.write(self.json_dumps())

    #------------------------------------
    # json_load
    #-------------------
    
    @classmethod
    def json_load(cls, fname):
        '''
        Load SpectralTemplate instance from file
        Read fname content as a json structure
        from which a SpectralTemplate instance can be
        retrieved. Return a new SpectralTemplate instance
        with the retrieved state.
        
        :param fname: json file name
        :type fname:str
        :return a fully initialized SpectralTemplate instance
        :rtype SpectralTemplate
        '''
        with open(fname, 'r') as fd:
            jstr = fd.read()
        template = cls.json_loads(jstr)
        return template

    #------------------------------------
    # mean_sig
    #-------------------

    @property
    def mean_sig(self):
        if self.cached_mean_sig is not None:
            return self.cached_mean_sig
        
        max_sig_len = max(self.sig_lengths)
        longest_seq = [sig 
                       for sig 
                       in self.signatures 
                       if len(sig) == max_sig_len
                       ][0]
        df = pd.DataFrame()
        for sig in self.signatures:
            # Number of needed pads
            sig_len = len(sig)
            pad_len = max_sig_len - sig_len
            # Get pad value for this signature:
            mean_freq = sig.mean()
            
            # Col names for the padding columns
            # will match the corresponding col names
            # in the longes sig: 
            col_names = longest_seq.index[sig_len :]
            
            # Pads are the mean of the sig:
            pad_series = pd.Series([mean_freq]*pad_len, 
                                   index=col_names)
            
            padded_sig      = sig.append(pad_series)
            padded_sig.name = sig.name
            # Now sig is as long as the longest seq,
            # so can append to df without generating
            # NaNs:
            df = df.append(padded_sig)
        
        # Finally: take the column-wise mean,
        # i.e. the mean of frequencies at each
        # time frame, generating a single pd.Series:
        mean_sig = df.mean(axis=0)
        mean_sig.name = 'mean_sig'
        self.cached_mean_sig = mean_sig
        return mean_sig

    #------------------------------------
    # frequency_band
    #-------------------
    
    def frequency_band(self):
        '''
        Finds the lower and higher frequency
        bounds of each signature, and returns
        an Interval such that 'low_val' is the lowest
        frequency, and 'high_val' is the highest
        frequency from among all signature's bandwidths  
        
        :returns the lowest frequency and the highest 
            frequency from among all signatures in the template
        :rtype Interval
        '''
        
        lowest_freq = min([sig.freq_interval['low_val']
                           for sig
                           in self.signatures
                           ])
        high_freq   = max([sig.freq_interval['high_val']
                           for sig
                           in self.signatures
                           ])
        
        return Interval(lowest_freq, high_freq + 1)

    #------------------------------------
    # as_time
    #-------------------
    
    def as_time(self, signature):
        '''
        For each element of the signature, return
        the wallclock time. The last entry corresponds
        to the duration of the clip from which the signature
        was extracted. 
        
        :param signature: signature for which to 
            return times
        :type signature: pd.Series
        :return array of fractional seconds
        :rtype [float]
        '''
        
        return librosa.frames_to_time(np.arange(len(signature)),
                                      hop_length=self.hop_length)


    #------------------------------------
    # duration
    #-------------------
    
    def duration(self, signature):
        '''
        Return the duration of the signature in
        fractional seconds.
        
        :param signature: signature to analyze
        :type signature: pd.Series
        :return time duration
        :rtype float
        '''
        
        duration = librosa.frames_to_samples(
            len(signature), hop_length=self.hop_length) / self.sample_rate
        return duration


    #------------------------------------
    # sig_lengths
    #-------------------
    
    @property
    def sig_lengths(self):
        
        return [len(sig) for sig in self.signatures]

    #------------------------------------
    # recording_fname
    #-------------------
    
    @property
    def recording_fname(self):
        return self.rec_fname

    #------------------------------------
    # sample_rate
    #-------------------
    
    @property
    def sample_rate(self):
        return self.sr
    
    #------------------------------------
    # hop_length
    #-------------------
    
    @property
    def hop_length(self):
        return self._hop_length

    #------------------------------------
    # n_fft
    #-------------------
    
    @property
    def n_fft(self):
        return self._n_fft

    #------------------------------------
    # get_sig
    #-------------------
    
    def get_sig(self, sig_id):
        '''
        Return member signature, given signature id
        
        :param sig_id: desired signatature's ID
        :type sig_id: Any
        :return: the requested signature
        :rtype: Signature
        :raise KeyError if no signature of given name is found
        '''
        try:
            return self.sig_dict[sig_id]
        except AttributeError:
            # First call to this method; build the dict:
            self.sig_dict = {sig.sig_id : sig for sig in self.signatures}
            return self.sig_dict[sig_id]

    #------------------------------------
    # __eq__
    #-------------------
    
    def __eq__(self, other):
        
        # Test equality of the signature:
        sig_equalities = [sig == other_sig
                          for sig, other_sig
                          in zip(self.signatures, other.signatures)
                          ]
        if sum(sig_equalities) != len(self.signatures):
            return False
        
        if [self.hop_length, self.rec_fname, self.sr, self.bandpass_filter] != \
           [other.hop_length, other.rec_fname, other.sr, other.bandpass_filter]:
            return False
        
        return True
        
    #------------------------------------
    # __getitem_
    #-------------------

    def __getitem__(self, idx):
        return self.signatures[idx]

    #------------------------------------
    # __len__
    #-------------------
    
    def __len__(self):
        return len(self.signatures)
    
    #------------------------------------
    # __iter__
    #-------------------
    
    def __iter__(self):
        
        return iter(self.signatures)

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        return f"<SpectralTemplate ({len(self.signatures)} sigs) {hex(id(self))}>"

    #------------------------------------
    # __str__
    #-------------------
    
    def __str__(self):
        return self.__repr__()
    
# -------------------------- Class Signature -----------

class Signature(JsonDumpableMixin):
    '''
    Instances hold the spectral signature of a
    single bird call. Information includes the
    following properties:
    
          o sig: a pd.DataFrame:
            Each row corresponds to one timeframe, and
            is a 4-tuples of:
            
              flatness,     # measure of 'peakiness'
              continuity    # percentage of frequencies
                            # that are part of a contour
              pitch         # fundamental frequency
              freq_mod      # frequency modulation 
               
            time. The index are the fractional seconds
            into the call for each value.
            
                        flatness    continuity   pitch   freq_mod
                 Time
                0.01456   2402
                0.02912   3725
                      ...
                      ...
          o fname: path to the recording that contains the
            bird call from which the signature was determined.
          o start_idx: index into the recording at which the
            call started
          o end_idx: index into the recording at which the
            call ended
          o sr: sample rate of the recording extract from which
            the call was taken. By default this is 22050, the librosa
            default to which the librosa.load() function resamples
            
    '''
    
    # Code for np.inf and -np.inf when rendering to json:
    JSON_INF     = '!infinity'
    JSON_NEG_INF = '!infinity_neg'

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self,
                 species, 
                 sig_values,
                 scale_info=None,
                 sr=22050, 
                 start_idx=0, 
                 end_idx=None, 
                 fname=None,
                 sig_id='na',
                 audio=None,
                 freq_interval=None,
                 bandpass_filter=None,
                 extract=True
                 ):
        '''
        The sig_values is the actual signature: the
        four spectral measures at each timeframe
        Each index value is expected to be a time into 
        the call relative to the beginning of the call 
        (i.e. not relative to the start of the recording 
        from which the call was lifted. Columns are expected
        to be ['flatness', 'continuity', 'pitch', 'freq_mod'].  
        
        
        NOTE ABOUT scale_info: if provided it is assumed that the
                   the passed in sig df has been normalized with
                   the given values. If this is not the case, pass
                   None, and call method normalize_self() with the
                   scale_info on the instance.
                   
        If provided, the scale_info consists of a dict with four entries,
        one each for flatness, continuity, pitch, and freq_mod.
        Each of the corresponding values is a nested dict holding
        the mean of the vocalizations across all calibration samples,
        and the median distance from the mean:
        
               {'flatness' : {'mean' : 10,
                              'standard_measure' : 2},
                'continuity':{'mean' : ...,
                              'standard_measure' : ...}
                              ...
                }
                
        This information is used in the norm_to_sig() method.

        The start_idx/end_idx are the indices to the clip samples
        in the full recording from which the clip is lifted.
        The fname is the path to a file; format up to the caller;
        may or may not include hostname.
        
        :param sig_values: spectral measures flatness,
            continuity, pitch, and frequency modulation.
        :type sig_values: pd.DataFrame
        :param scale_info: scale factors derived from median
            distance from mean of pitch, frequency modulation,
            spectral flatness, and spectral continuity.
        :type scale_info: pd.Series
        :param sr: sample rate
        :type sr: int
        :param start_idx: index into the full recording's samples
            array where the call begins
        :type start_idx: int
        :param end_idx: index into the full recording's samples
            array where the call ends
        :type end_idx: int
        :param species: species that voiced the call
        :type species: str
        :param fname: path to full recording
        :type fname: str
        :param sig_id: optionally some identifier of this
            signature that is meaningful to the caller
        :type sig_id: Any
        :param audio: optionally, the audio from which the 
            signature was computed
        :type np.ndarray
        :param freq_interval: dict providing lowest, and highest 
            frequency of the signature, and the frequency steps
            with which the signature's spectrogram was created.
        :type freq_interval: {str : float}
        :param bandpass_filter: if not None: an interval that defines
            the bandpass filter that was applied before computing
            this signature.
        :type bandpass_filter: {None | Interval}
        :param extract: if True, signature was created by
            not just applying a bandpass filter, but 
            clipping out the spectrogram portions that
            are outside the filter
        :type extract: bool
        '''

        # Until we know better, assume that the
        # passed-in signature values are raw:
        self.normalized = False
        
        if type(sig_values) == pd.DataFrame:
            self.sig = sig_values
        else:
            raise TypeError(f"The spectral values must be a pd.DataFrame, not {type(sig_values)}")

        if scale_info is None:
            # Leave signature values raw when
            # normalize_self is called:
            self.scale_info = None
        else:
            if type(scale_info) != dict:
                raise TypeError(f"The scale factors must be a dict, not {type(scale_info)}")
            self.scale_info = scale_info
            self.normalize_self(scale_info)

        if fname is not None:
            rec_fname = Path(fname).name
        else:
            rec_fname = 'fname_na'
        self.sig.name = f"{rec_fname}_call_{sig_id}"
        self.sr = sr
        self.fname = fname
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.species = species
        self.sig_id  = sig_id
        self.audio = audio
        self.freq_interval = freq_interval
        if freq_interval is not None:
            self.freq_span = freq_interval['high_val'] - freq_interval['low_val']
        else:
            self.freq_span = None
        self.bandpass_filter = bandpass_filter
        self.extract = extract

    #------------------------------------
    # match_probabilities
    #-------------------
    
    def match_probabilities(self, other):
        '''
        Given another Signature, compute the pairwise
        feature vector distances. Apply a sigmoid, and
        return
         
             1 - sigmoid(distance(self, other))
             
        :param other: signature against which other is to be compared
        :type other: Signature
        :returns: a Series of probabilities over time. Each element
            is the probability that self and others are matched at
            the element's time.
        :rtype: pd.Series
        '''

        res = self._distances_to_probs(self._distance(other))
        return res
    
    #------------------------------------
    # _distance
    #-------------------
    
    def _distance(self, other):
        '''
        Compute Euclidean distance between the
        sig df of this Signature instance and another.
        The computation is the square root of the sum
        of squared feature distances. We have four features 
        here: flatness, continuity, pitch, freq_mod.
        
                1
          sqrt(--- * sum_1..n((feature_i(sig1) - feature_i(sig2))**2))
                n

        n=4 (the four features). The division by 2
        in the implementation below is the 1/n taken
        out of the sqrt: sqrt(1/4) == 1/2
        
        :param other: Signature instance from which the
            distance is to be computed
        :type other: Signature
        :returns a series of probabilities by time; each
            element is the probability that other is a
            match to signature self.
        :rtype pd.Series 
        '''
        
        # The .values is required because the
        # index of self.sig and other.sig will likely differ.
        # Without the .value we get NaN whenever an index
        # is mismatched:
        dist_over_time = np.mean(np.sqrt(np.nansum((self.sig.values - other.sig.values)**2, axis=1)) / 2.)
        return dist_over_time 

    #------------------------------------
    # _distances_to_probs
    #-------------------
    
    def _distances_to_probs(self, distances):
        '''
        Given a Series of distances between two signatures,
        return the corresponding probabilities that the
        two signatures are a match. The index of the given
        distances is assumed to be time, and that index is
        retained in the result. However, the computation does
        not depend on the index.
        
        Strategy: apply a sigmoid to each distance, and return
        1-sigmoid(distance).
        
        :param distances: the pairwise distances of feature
            vectors, for example as returned by distance()
        :type distances: pd.Series
        :returns: a Series with distances replaced by probabilities
            of being a match
        :rtype: pd.Series
        '''
        # Apply a sigmoid to squeeze the distances
        # into the [0,1] probabilities range:
        probs = 1. / (1+np.exp(-distances))
        # The *smaller* the distance that higher the probability:
        return 1 - probs


    #------------------------------------
    # norm_to_sig
    #-------------------
    
    def norm_to_sig(self, val, measure_type):
        '''
        Given a number, pd.Series, pd.DataFrame, or
        np.ndarray, element-wise normalize the value.
        
        The formula used is:
        
                    val - overall_mean
                    -------------------
                    mean_dist_from_mean 

        Where overall_mean is the mean of values across all
        calibration vocalizations, and median_dist_from_mean is
        also computed across all calbration samples.
        
        The median distance from the mean is what is used in
        Ofer Tchernichovski, Fernando Nottebohm, Ching Elizabeth Ho, Bijan Pesaran &
        Partha Pratim Mitra: "A procedure for an automated measurement of song similarity"
        One can use mean distance from the mean instead, i.e. the standard deviation.
        
        The measure_type must be one of 'flatness', 'continuity', 
        'pitch', or 'freq_mod'. The value is used to select the
        proper overall_mean and median_dist_from_mean. 
        
        :param val: value to be normalized
        :type val: {number | pd.Series | pd.DataFrame | np.ndarray}
        :param measure_type: which kind of measure is to be normalized
        :type measure_type: {'flatness', 'continuity', 'pitch', 'freq_mod', 'energy_sum'}
        :return element-wise normalized value
        :rtype {number | pd.Series | pd.DataFrame | np.ndarray}
        '''
        
        try:
            measure_info = self.scale_info[measure_type]
        except KeyError:
            raise ValueError(f"Measure type must be one of 'flatness', 'continuity', 'pitch', 'freq_mod', or 'energy_sum', not {measure_type}")
        res = (val - measure_info['mean']) / measure_info['standard_measure']
        return res
        
    #------------------------------------
    # normalize_self
    #-------------------
    
    def normalize_self(self, scale_info):
        '''
        
        '''
        if self.normalized:
            raise RuntimeError("Already normalized")
        
        self.scale_info = scale_info
        
        normed_flatness   = self.norm_to_sig(self.sig['flatness'], measure_type='flatness')
        normed_continuity = self.norm_to_sig(self.sig['continuity'], measure_type='continuity')
        normed_pitch      = self.norm_to_sig(self.sig['pitch'], measure_type='pitch')
        normed_freq_mod   = self.norm_to_sig(self.sig['freq_mod'], measure_type='freq_mod')
        # normed_energy_sum = self.norm_to_sig(self.sig['energy_sum'], measure_type='energy_sum')
        
        new_sig = pd.DataFrame({'flatness' : normed_flatness,
                                'continuity' : normed_continuity,
                                'pitch' : normed_pitch,
                                'freq_mod' : normed_freq_mod #,
                                # 'energy_sum' : normed_energy_sum
                                })
        self.sig = new_sig
        self.normalized = True
        
    #------------------------------------
    # as_walltime
    #-------------------
    
    def as_walltime(self):
        '''
        Returns a copy of the sig dataframe with
        the index relative to the start of the 
        entire recording from which the call clip
        was taken.
        
        :return: copy of signature DataFrame with new index
        :rtype: pd.DataFrame
        '''
        
        sig_times = self.sig.index
        new_index = sig_times + librosa.samples_to_time(self.start_idx, self.sr)
        sig_copy  = self.sig.copy()
        sig_copy.index = new_index
        return sig_copy
    
    #------------------------------------
    # as_frames
    #-------------------
    
    def as_frames(self):
        '''
        Returns a copy of the sig df with
        the index replaced with the enumeration
        of spectral frames for each value (i.e. 1..len(sig)).
        
        :return: copy of signature df with new index
        :rtype: pd.DataFrame
        '''
        
        sig_copy  = self.sig.copy()
        sig_copy.index = np.arange(len(self.sig))
        return sig_copy

    #------------------------------------
    # duration
    #-------------------

    def duration(self):
        '''
        Returns the duration of the vocalization that
        generated the signature.
        
        :return: duration of underlying call in fractional seconds
        :rtype: float
        '''
        return self.sig.index[-1]

    #------------------------------------
    # plot
    #-------------------
    
    def plot(self, ax=None):
        '''
        Plot the signature as five lines. The x axis 
        will be labeled with time into signature.
        
        The y axis will be unitless, since the values
        are scaled to the medium distance of each measure's
        deviation from the mean in example calls.
         
        :param ax: optional existing Axes instance to use
        :type ax: plt.Axes
        '''
        
        if ax is None:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        else:
            self.ax = ax
        for col in self.sig.columns:
            self.ax.plot(self.sig[col], label=col)
        self.fig.legend()
        self.fig.show()

    #------------------------------------
    # json_dumps
    #-------------------
    
    def json_dumps(self):
        '''
        Return a json string holding the contents
        of this instance, ready to json.dump() into
        a file. To create a new Signature instance,
        pass the string to class method Signature.json_loads()
        '''
        
        # Only save what we need, e.g. not the audio.
        # Cannot use form df.to_json() for self.sig, 
        # because df.to_json() does not handle np.inf
        # or -np.inf values. So, if necessary, make a
        # copy of self.sig, and replace those values with
        # ones that we then revert back to np.inf/-np.inf:
        
        has_inf     = self.sig[self.sig == np.inf].any().any()
        has_neg_inf = self.sig[self.sig == -np.inf].any().any()
        if has_inf or has_neg_inf:
            sig_measure = self.sig.copy()
        else:
            sig_measure = self.sig
        if has_inf:
            sig_measure.replace(np.inf, self.JSON_INF, inplace=True)
        if has_neg_inf:
            sig_measure.replace(-np.inf, self.JSON_NEG_INF, inplace=True)

        recovery_dict = {
            "sig" : sig_measure.to_json(),
            "scale_info" : self.scale_info,
            "sr" : self.sr,
            "fname" : self.fname,
            "start_idx" : self.start_idx,
            "end_idx" : self.end_idx,
            "species" : self.species,
            "sig_id" : self.sig_id,
            "normalized" : self.normalized,
            "freq_interval" : None if self.freq_interval is None else self.freq_interval.json_dumps(),
            "freq_span" : self.freq_span,
            "bandpass_filter" : None if self.bandpass_filter is None else self.bandpass_filter.json_dumps()
            }
        if hasattr(self, 'usable'):
            recovery_dict['usable'] = self.usable
        if hasattr(self, 'prominence_threshold'):
            recovery_dict['prominence_threshold'] = self.prominence_threshold
        if hasattr(self, 'prob_threshold'):
            recovery_dict['prob_threshold'] = self.prob_threshold
            
        return json.dumps(recovery_dict)

    #------------------------------------
    # json_loads
    #-------------------
    
    @classmethod
    def json_loads(cls, jstr):
        '''
        Given a json string straight from reading
        from a json file, return an instance of Signature.

        :param jstr: unevaluated json string 
        :type jstr: str
        :return a new Signature instance
        :rtype Signature
        '''
        
        jdict = json.loads(jstr)
        try:
            sig = pd.DataFrame(json.loads(jdict['sig']))
            # Replace the np.inf and -np.inf codes that may
            # have been used in self.json_dumps() to denote
            # np.inf and -np.inf:
            
            sig.replace(cls.JSON_INF, np.inf, inplace=True)
            sig.replace(cls.JSON_NEG_INF, -np.inf, inplace=True)
            
            # The index dtype will have defaulted to Object.
            # Cast it to float64:
            sig.index = sig.index.astype('float64') 
        except Exception as e:
            raise ValueError(f"Could not read signature df from string ({repr(e)})")

        freq_interval = jdict['freq_interval'] 
        if freq_interval is not None:
            freq_interval = Interval(*list(freq_interval.values()))
        
        bandpass_filter = jdict['bandpass_filter']
        if bandpass_filter is not None:
            bandpass_filter = Interval(*list(bandpass_filter.values()))

        # Set scale_info to None when creating the
        # instance to avoid re-normalizing a sig that
        # was already normalized before saving:
        
        scale_info =jdict['scale_info']
        normalized = jdict['normalized']

        sig_instance = Signature(
            jdict['species'],
            sig,
            scale_info=None,
            sr=jdict['sr'],
            start_idx=jdict['start_idx'],
            end_idx=jdict['end_idx'],
            fname=jdict['fname'],
            sig_id=jdict['sig_id'],
            freq_interval=freq_interval,
            bandpass_filter=bandpass_filter
            )
        
        sig_instance.normalized = normalized
        sig_instance.scale_info = scale_info
        if not normalized:
            sig_instance.normalize_self(scale_info)
            
        # Was the sig calibrated?
        try:
            sig_instance.usable = jdict['usable']
            sig_instance.prominence_threshold = jdict['prominence_threshold']
            sig_instance.prob_threshold = jdict['prob_threshold']
        except KeyError:
            # No, it wasn't; fine:
            pass

        return sig_instance


    #------------------------------------
    # json_dump
    #-------------------

    def json_dump(self, fname):
        
        with open(fname, 'w') as fd:
            json.dump(self.json_dumps(), fd)

    #------------------------------------
    # json_load
    #-------------------
    
    @classmethod
    def json_load(cls, fname):
        '''
        Read fname content as a json structure
        from which a Signature instance can be
        retrieved. Return a new Signature instance
        with the retrieved state.
        
        :param fname: json file name
        :type fname:str
        :return fully initialized Signature instance
        :rtype Signature
        '''
        with open(fname, 'r') as fd:
            json_str = json.load(fd)
        inst = cls.json_loads(json_str)
        return inst

    #------------------------------------
    # __len__
    #-------------------
    
    def __len__(self):
        return len(self.sig)
    
    #------------------------------------
    # __iter__
    #-------------------
    
    def __iter__(self):
        return iter(self.sig)
    
    #------------------------------------
    # __getitem__
    #-------------------
    
    def __getitem__(self, key):
        
        if type(key) == float:
            # Key is assumed to be a time in fractional seconds:
            return self.sig.loc[key] 
        elif type(key) == int:
            return self.sig[key]
        else:
            raise TypeError(f"Only floats and ints can index into sigs, not {key}")

    #------------------------------------
    # __eq__
    #-------------------
    
    def __eq__(self, other):

        # Number of attributes must match:
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            return False 
        
        # First check equality of corresponding scalars that are
        # guaranteed to be present:
        if [self.fname, self.sr,self.start_idx, self.end_idx, self.species,
            self.sig_id, self.audio, self.freq_span] != \
           [other.fname, other.sr,other.start_idx, other.end_idx, other.species,
            other.sig_id, other.audio, other.freq_span]:
            return False
        
        # Now the dataframes and Series:
        if not Utils.df_eq(self.sig, other.sig):
            return False
        if self.scale_info != other.scale_info:
            return False

        if self.bandpass_filter != other.bandpass_filter:
            return False
        
        return True

    #------------------------------------
    # __str__
    #-------------------

    def __str__(self):
        return f"<Signature ({self.species}-{self.sig_id}) {hex(id(self))}>"

    #------------------------------------
    # __repr__
    #-------------------

    def __repr__(self):
        return self.__str__()

    #------------------------------------
    # index
    #-------------------
    
    @property
    def index(self):
        '''
        Allow Signature instance to be used like
        a pd.Series by adding this index property.
        '''
        return self.sig.index

    #------------------------------------
    # name
    #-------------------
    
    @property
    def columns(self):
        '''
        Allow Signature instance to be used like
        a pd.DataFrame by adding this roperty.
        '''
        return self.sig.columns
    
# ------------------------ TemplateCollection --------------

class TemplateCollection(dict, JsonDumpableMixin):
    '''
    Simple dict mapping species names to a SpectralTemplate 
    instance:
             {'CMTOG' : <template for CMTOG>,
              'BANAS' : <template for BANAS>
                 ...
             }
    An instance of this class is created by quad_sig_calibration,
    and usually stored in an experiment's json directory under the
    key 'templates'. 
    
    Use instances like a dict. But json storage management is 
    an additional functionality. The class provides the:
        json_dump()
        json_load()
    methods used to store and retrieve instances in experiments
    (i.e. instances of ExperimentManager).
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, templates_dict):
        
        # This instance is itself a dict:
        self.update(templates_dict)
        
    #------------------------------------
    # json_load
    #-------------------
    
    @classmethod
    def json_load(cls, fname):
        '''
        Workhorse for reading a jsonized instance from file
        such a file.
        
        Reconstruct a dict of templates from
        the given json file. All SpectralTemplate
        instances will be reconstructed.
        The result be an instance of TemplateCollection,
        and will thus look like:
        
            {'CMTOG' : template_cmtog,
             'OtherSpec' : template_other_spec
             }

        :param fname: json file to load
        :type fname: str
        :return TemplateCollection instance
        :rtype TemplateCollection
        :raise FileNotFoundError
        '''
        with open(fname, 'r') as fd:
            dict_of_templates = json.load(fd)
        
        new_dict = {species : SpectralTemplate.json_loads(jstr)
                    for species, jstr
                    in dict_of_templates.items()
                    }

        return TemplateCollection(new_dict)

    #------------------------------------
    # json_dump
    #-------------------
    
    def json_dump(self, fname):
        '''
        Individually json encode each SpectralTemplate
        in self.values(), and write the entire dict to fname 
        as a json file.
        
        :param fname: destination path
        :type fname: str
        '''
        with open(fname, 'w') as fd:
            json.dump({key : val.json_dumps() for key,val in self.items()}, fd)
