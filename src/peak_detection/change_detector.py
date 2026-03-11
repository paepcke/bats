# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2025-11-30 16:45:08
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2025-12-08 10:29:22

from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from logging_service import LoggingService

class ChangeDetector:
    """
    Detects significant scene changes in a time-series of 'content_val' 
    by applying Gaussian smoothing and filtering peaks based on prominence.
    Usage:
      detector = SceneChangeDetector(scene_cur_values, vid_path)
      scenes = detector.detect_scenes()
      detector.frames() # to get the frame images for saving.
                        # Frames are ordered to correspond to
                        # the rows in the returned scenes df.
    """

    def __init__(
            self,
            time_series: pd.Series | str,
            sigma: Optional[int] = 3, 
            min_prominence: Optional[float] = None,
            min_height: Optional[float] = None,
            min_plateau: Optional[float] = None
            ) -> pd.DataFrame:
        """
        Initializes the detector with key parameters.
        The time_series_spec will be loaded into a df from file if the 
        path of a .csv is given. The loaded, or directly given
        df must include a column named as given by change_magnitude_col.
        Example, a df with columns:
            content_val    frame_number     timecode

        The pd.Series found by the above procedure will be available
        to callers in

              <scene-change-detector-instance>.time_series

        :param time_series_spec: either path to a .csv file, or a Pandas dataframe
        :param sigma: The standard deviation (spread) of the Gaussian filter. 
                         Higher sigma means more smoothing.
        :param min_prominence: Minimum required prominence for a peak 
                         to be considered a scene change. (Key for filtering spikes)
        :param min_height: Minimum absolute value a peak must reach. 
                         (Used to ensure the detected peak is high enough, like 5.0)
        :data_col: column in .csv file or dataframe that holds the frame change
                         quantification data.
        """
        self.sigma = sigma
        self.min_prominence = min_prominence
        self.min_height = min_height
        self.min_plateau = min_plateau
        self.smoothed_data = None
        self.scene_change_indices = None

        self.log = LoggingService()

        self.time_series = time_series
            
        # Determine the window size for the Gaussian kernel (e.g., 6*sigma + 1)
        # 3stds on left of distrib + 3stds on right of distrib ==> 6*sigma
        # The +1 makes the window odd to allow algnments around the mean:
        # self.window_size = int(6 * self.sigma + 1)

    def detect_changes(self) -> pd.DataFrame:
        """
        Processes the time series to detect scene changes.
        Returns dataframe like:
              
               frame_number  prominence  smoothed_content_val  content_vals      scene_frame
            0            27    4.783040              9.630631      9.620940    <img np.ndarray>
            1           294    7.560820              9.915100     10.103290    <img np.ndarray>
            4          1180    3.769875             14.477572     14.661802    <img np.ndarray>
            5          1278    5.455810             10.201112     10.268808    <img np.ndarray>

        Args:
            time_series (pd.Series): the movie's frame-by-frame change amount scores

        Returns:
            pd.DataFrame: A DataFrame containing the detected scene 
                change frames with associated change amount scores,
                and the scene image for each scene.
        """
        if not isinstance(self.time_series, pd.Series):
            raise TypeError("Input must be a pandas Series.")
            
        # Apply Smoothing
        self.smoothed_data = self._apply_gaussian_smoothing(self.time_series)
        
        # Find Peaks (using prominence and height filters)
        # find_peaks returns indices of detected peaks
        # The properties will be a dict with keys 
        #      ['peak_heights', 'prominences', 'left_bases', 'right_bases']
        # These are the same lengths as the scene indices, and correspond
        # to them. 
        # The indices point to scene changes:
        peak_height_constraint = self.min_height if self.min_height is not None \
                                                 else (-np.inf, np.inf)
        prominence_contraint = self.min_prominence if self.min_prominence is not None \
                                                   else (0, None)
        plateau_constraint = self.min_plateau if self.min_plateau is not None \
                                                   else (1, None)
        indices, properties = find_peaks(
            self.smoothed_data, 
            height=peak_height_constraint,
            prominence=prominence_contraint,
            plateau_size=plateau_constraint
        )
        
        self.scene_change_indices = indices
        
        # Make a result df like:
        #    idx  frame_number prominence height plateau  smoothed_val content_vals
        # Collect the rows that are scene changes.
        # This will be a pd.Series in which the index
        # values are the frame numbers, and the Series
        # values are the content_vals (i.e. frame change scores)
        scene_data = self.time_series.iloc[indices].copy()
        scenes = pd.DataFrame({
            'frame_number': scene_data.index,
            'prominence'  : properties['prominences'],
            'plateau'     : properties['plateau_sizes'],
            'smoothed_content_val': self.smoothed_data[indices],
            'content_vals': self.time_series[indices]
        })
        # The frame_number entries will have turned into floats.
        # Make them ints again:
        scenes['frame_number'] = scenes['frame_number'].astype('int64')
        msg = f"Found {len(scenes)} scenes"
        self.log.info(msg)
        return scenes

    def get_smoothed_values(self):
        """For clients of this class: the smoothed values array for plotting/analysis."""
        return self.smoothed_data
    
    def _apply_gaussian_smoothing(self, data):
        """Applies a 1D Gaussian filter to the input data."""
        
        smoothed = gaussian_filter(data, sigma=self.sigma, order=0, mode='reflect')
        return smoothed
