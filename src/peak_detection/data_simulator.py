 # **********************************************************
 #
 # @Author: Google Gemini
 # @Date:   2025-12-08 09:11:06
 # @File:   /Users/paepcke/VSCodeWorkspaces/test/src/data_simulator.py
 # @Last Modified by:   Andreas Paepcke
 # @Last Modified time: 2025-12-08 09:13:37
 #
 # **********************************************************

import numpy as np
import pandas as pd

class DataSimulator:
    """
    A class to generate a jittery Pandas Series of values in [0, 1] 
    with three defined, noisy peaks.
    """
    def __init__(self, 
                 N: int =100, 
                 peak_height: float = 0.8, 
                 base_noise_level: float = 0.1, 
                 peak_noise_level: float =0.15, 
                 random_seed: int =42):
        """
        Initializes the DataSimulator with configuration parameters.

        Args:
            N (int): The total number of data points to generate.
            peak_height (float): The maximum height of the Gaussian base of the peaks.
            base_noise_level (float): The maximum noise added to the non-peak sections.
            peak_noise_level (float): The maximum random jitter added to the peaks.
            random_seed (int): Seed for numpy's random generator for reproducibility.
        """
        self.N = N
        self.peak_height = peak_height
        self.base_noise_level = base_noise_level
        self.peak_noise_level = peak_noise_level
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def generate_series(self) -> pd.Series:
        """
        Generates the final Pandas Series with base jitter and three noisy peaks.

        Returns:
            pd.Series: The generated series named 'content_vals'.
        """
        
        # 1. Create the base 'jittery' data array (mostly low values)
        # Start with a low base value and add uniform noise
        data = np.random.uniform(0.05, self.base_noise_level, self.N)

        # 2. Define the three peak locations and widths
        # Location and width are hardcoded here based on the original request's structure
        start1, end1 = 15, 25  # Peak 1: ~10 points wide
        start2, end2 = 40, 55  # Peak 2: ~15 points wide
        start3, end3 = 82, 90  # Peak 3: ~8 points wide

        # 3. Generate the peaks and add them to the base data
        data = self._add_noisy_peak(data, start1, end1)
        data = self._add_noisy_peak(data, start2, end2)
        data = self._add_noisy_peak(data, start3, end3)

        # 4. Insert the array into a Pandas Series
        content_vals = pd.Series(data, name='content_vals')
        
        return content_vals

    def _add_noisy_peak(self, data, start, end):
        """Adds a Gaussian-like peak with added noise to a segment of the data."""
        segment_len = end - start
        if segment_len <= 0:
            return data

        # Create a smooth, centered peak using a Gaussian function
        x = np.linspace(-3, 3, segment_len)
        # Use peak_height and a standard deviation of 1.5 for the Gaussian shape
        gaussian_peak = self.peak_height * np.exp(-0.5 * (x/1.5)**2)

        # Add peak noise (jitter on the peak height)
        peak_jitter = np.random.uniform(-self.peak_noise_level, self.peak_noise_level, segment_len)
        noisy_peak = gaussian_peak + peak_jitter

        # Add the noisy peak to the data segment, ensuring values stay below 1
        # Use += to combine with existing base data and noise
        data[start:end] += noisy_peak
        data[start:end] = np.clip(data[start:end], 0, 1) # Keep values in [0, 1]

        return data

# --- Example Usage ---
# print("## 🚀 DataSimulator Class Example")
# print("---")

# # 1. Instantiate the class
# simulator = DataSimulator()

# # 2. Generate the series
# content_vals = simulator.generate_series()

# # 3. Display results
# print(f"Generated a Pandas Series with {len(content_vals)} points.")
# print("The first 10 values of 'content_vals':")
# print(content_vals.head(10))
# print("\nDescriptive Statistics:")
# print(content_vals.describe())
