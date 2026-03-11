#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2025-11-30 12:55:10
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2025-12-08 11:15:50

"""
MovieAnalyzer - Analyze content values in video files, generating
suggested representative frames for summarizing the movie.

This tool uses PySceneDetect to analyze video content and provides:
- Statistical analysis of content values
- Histogram visualization
- Time-series visualization with interactive thumbnail preview
- Scene count threshold calculation
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# PySceneDetect imports
import pandas as pd
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from scenedetect import open_video
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager

from logging_service import LoggingService

sys.path.append("./src")
from peak_detection.change_detector import ChangeDetector
from peak_detection.data_simulator import DataSimulator


class DataSeriesAnalyzer:
    """Analyze scene detection metrics for video files."""
    
    def __init__(self, 
                 data: np.ndarray | list | pd.Series,
                 scenecount_max_absolute: int | None = None, 
                 visuals: bool = True):
        """
        Initialize the MovieAnalyzer. Optionally show time charts of
        frame-by-frame changes. If scenecount_max is provided, no more
        than that number of scenes are identified. The reductions are 
        made by accepting only highly prominent frame change peaks.
        
        :param video_path: Path to the video file (.mp4, .mov, etc.)
        :param scenecount_max: optional limit on the number of scenes
        :param visuals: whether or not to show progress bars and charts
        """

        self.scenecount_max = scenecount_max_absolute
        self.visuals = visuals
        
        self.log = LoggingService()
        
        self.data = data
        self.smooth_data: pd.DataFrame = None
        self.scenes: pd.DataFrame = None
        
    def analyze(self, sigma=3) -> pd.DataFrame:
        """
        Run scene detection analysis on the video.
        Return a dataframe in which each row represents one 
        'important' scene. The goal is to find video frames
        that stand out from its environment, and are dissimilar
        from each other.

        Most importantly, rows include columns: frame_number and
        scene_frame. The scene_frame is raw image data of one scene
        Caller can use VideoUtils.show_frame(frame)
        or VideoUtils.frame_to_jpeg(frame, file_name, metadata)
        to save or display the raw frames.

        Details of the returned df:
               frame_number  prominence  smoothed_content_val  content_vals      scene_frame
            0            27    4.783040              9.630631      9.620940    <img np.ndarray>
            1           294    7.560820              9.915100     10.103290    <img np.ndarray>
            4          1180    3.769875             14.477572     14.661802    <img np.ndarray>
            5          1278    5.455810             10.201112     10.268808    <img np.ndarray>

        :returns dataframe with all needed scene information
        """
                
        # Create scene manager and stats manager. The scene manager
        # coordinates the movie processing:
        self.stats_manager = StatsManager()
        self.scene_manager = SceneManager(self.stats_manager)

        # Smooth these values, and find peaks and prominences
        self.scene_detector = ChangeDetector(self.data, sigma=sigma)

        # Obtain a df with just scene change pointer, e.g. for just scenes:
        #  idx    content_val  frame_number  timecode   prominence  smoothed_val
        #  159     12.602648       160       5.333333    7.562601     12.153559
        #                          ...
        scenes = self.scene_detector.detect_changes()

        # Are we to limit the number of scenes?
        if self.scenecount_max is not None and len(scenes) > self.scenecount_max:
            # Reduce the number of scenes by prioritizing high-prominence 
            # peaks in the frame-by-frame differences:

            # Keep the original accessible, but without all the images
            # They can be re-found from the 'frame_number' column:
            #self.all_scenes_no_img_copies = scenes.copy()
            self.all_scenes_no_img_copies = scenes.drop(columns=['scene_frame'])

            # Select the top N rows based on prominence
            # nlargest is generally faster/cleaner than sort_values().head().
            # Preserve the first found frame (therefore the [:1])
            reference_scene = scenes.iloc[0]
            # Find the largest prominences in the remaining scenese:
            subset = scenes.iloc[1:].nlargest(self.scenecount_max - 1, 'prominence')
            # Include the reference scene back:
            subset.loc[reference_scene['frame_number']] = reference_scene

            # Sort back by index (or frame_number) to restore temporal order
            scenes = subset.sort_index()

            # Reset index to have a 0,1,2,... index
            scenes = scenes.reset_index(drop=True)           
        
        self.scenes = scenes
        return scenes

    def get_smoothed_data(self):
        return self.scene_detector.get_smoothed_values()

        pass  
    def get_statistics(self, vals: pd.Series) -> dict:
        """
        Compute statistics on content values.

        Returns:
            Dictionary containing mean, median, std, min, max
        """
        if vals is None or len(vals) == 0:
            raise ValueError("No content values available. Run analyze() first.")
        
        stats = {
            'mean': np.mean(vals),
            'median': np.median(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'count': len(vals)
        }
        
        return stats
    
    def print_statistics(self, 
                         frame_content_data: pd.Series,
                         title: Optional[str] = None):
        
        """Print statistics to console."""
        stats = self.get_statistics(frame_content_data)
        title = "\n=== Content Value Statistics ===" \
            if title is None \
            else f"\n=== {title} ==="
                                                    
        print(title)
        print(f"Mean:     {stats['mean']:.2f}")
        print(f"Median:   {stats['median']:.2f}")
        print(f"Std Dev:  {stats['std']:.2f}")
        print(f"Min:      {stats['min']:.2f}")
        print(f"Max:      {stats['max']:.2f}")
        print(f"Frames:   {stats['count']}")
        print(f"Scenes:   {len(self.scenes)}")
    
    def plot_timeseries(
            self, 
            timeseries_data: List[pd.DataFrame] | pd.DataFrame,
            labels: Optional[List[str]] = None,
            interactive: bool = True
        ) -> Figure:
        """
        Create and display content values over time.
        
        Args:
            timeseries_data: the one or more series to plot
            labels: optional labels for color legend
            interactive: If True, enable interactive thumbnail preview
            save_path: Optional path to save the figure
        """
        
        if type(timeseries_data) != list:
            timeseries_data = [timeseries_data]

        if type(labels) != list:
            labels = [labels]            

        # Handle case where labels are not provided
        if labels is None:
            labels = [f"FrameDiffs {i+1}" for i in range(len(timeseries_data))]        

        # Check if lists match length (Optional safety check)
        if len(timeseries_data) != len(labels):
            raise ValueError(f"{len(timeseries_data)} DataFrames but {len(labels)} labels provided.")

        fig, ax = plt.subplots(figsize=(14, 6))

        for timeseries, label in zip(timeseries_data, labels):
            # Plot content values
            ax.plot(
                timeseries.index.values, 
                timeseries.values,
                linewidth=1, 
                alpha=0.8,
                label=label
                )
        
        # Mark detected scenes
        for idx, scene in self.scenes.iterrows():
            scene_time = scene['frame_number']
            ax.axvline(scene_time, color='r', alpha=0.3, linewidth=1, linestyle='--')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Content Change Score')
        # Generate a title based on the labels provided or generic text
        subtitle_text = ", ".join(labels) if labels and len(labels) < 4 else "Multiple Series"
        ax.set_title(f'Content Change: {subtitle_text} '
                     f'({len(self.scenes)} scenes detected)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if interactive:
            self._add_interactive_preview(fig, ax)
        # Show all plots at the end in main()
        #plt.show()
        return fig
    
    def save_stats_csv(self, output_path: str) -> None:
        """
        Save statistics to CSV file (similar to scenedetect CLI output).
        
        Args:
            output_path: Path for the output CSV file
        """
        if not self.data:
            raise ValueError("No content values available. Run analyze() first.")

        self.data.to_csv(output_path)       
        self.log.info(f"Stats saved to: {output_path}")

    def save_fig_and_data(self, fig_path, fig, **kwargs):
        '''
        Given a matplotlib figure, and associated data, save the figure
        as png, or other format, and save the data as .csv. The fig_path
        is the directory plus figure file. The extension determines the
        output format.
        Example:
             '/tmp/myMovieStatsTimes.png'

        The kwargs may provide any number of data arrays. Each
        kwarg key will be a column header. Corresponding value arrays
        will be columns. It is an error for the value arrays to have
        unequal lengths. Example for a **kwargs dict:
              {
                'content_val': [1,2,3],
                'timecode'   : [100,200,300]
              }

        The data will be saved in fig_path[without-ext].csv. In this example:
            /tmp/myMovieStatsStatsTimes.csv

        If fig_path or derived data destinations exist, they are overwritten\
        without warning.

        Directory will be created if needed.

        :param fig_path: path to figure save file
        :type fig_path: {str | Path}
        :param fig: Figure to save
        :type fig: Figure
        '''
        fig_path = Path(fig_path)
        dst_dir = Path(fig_path).parent 
        Path.mkdir(dst_dir, parents=True, exist_ok=True)

        # Save the plot
        fig.savefig(fig_path, dpi=150)

        # Save the data, if provided:
        if len(kwargs) > 0:
            data_path = fig_path.with_suffix('.csv')
            with open(data_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow(list(kwargs.keys()))
                writer.writerows(zip(*kwargs.values()))

def main():
    """Main entry point for the script."""

    log = LoggingService()
    #********** PUT REAL DATA HERE ***********
    data = DataSimulator().generate_series()    
    #********** PUT REAL DATA HERE ***********

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Exercise peak detection."
                                     )

    parser.add_argument('-c', '--charts',
                        action='store_true',
                        default=True,
                        help='whether to draw charts')

    args = parser.parse_args()
    try:
        analyzer = DataSeriesAnalyzer(data)
        # Get a df with columns:
        # 'frame_number', 'prominence', 'plateau', 'smoothed_content_val', 'content_vals'
        peaks_info = analyzer.analyze()
        
        # Print statistics
        if args.charts:
            analyzer.print_statistics(data, title="Raw Data")
            analyzer.print_statistics(analyzer.get_smoothed_data(), title="Smoothed Data")
            # Create visualizations
            log.info("\nGenerating visualizations...")
            # Time series with optional interactive preview
            fig: plt.Figure = analyzer.plot_timeseries(
                    [data, 
                     pd.Series(analyzer.get_smoothed_data())
                    ],
                    labels=['raw data', 'smoothed'],
                    interactive=False
            )
            log.info("Waiting for user to close charts...")
            plt.show(block=False)
            while input("Press q to quit...") not in ['q', 'Q']:
                        type=bool,
                        nargs='+',
    except Exception as e:
        log.err(f"Error: {e}")
    
    log.info("done.")


if __name__ == '__main__':
    main()
