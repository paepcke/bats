# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2025-12-01 14:43:53
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2025-12-08 09:41:48

import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
import cv2
from PIL import Image

# ------------------- Context Manager VideoStream -----------------

class VideoStream:
    '''
    Context manager for video streams such as a .mov file 
    or a camera. 
    Usage:
        with VideoStream(<video-file>) as cap:
            
    '''
    def __init__(self, source=0):
        """
        :param source: 0 for webcam, or a string path for a video file.
        """
        self.source = source
        self.cap = None

    def __enter__(self):
        # 1. Acquire the resource
        self.cap = cv2.VideoCapture(self.source)
        
        # 2. Check if opened successfully
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")
            
        # 3. Return the resource to be used in the 'as' clause
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 4. Release the resource (this runs even if the code crashes)
        if self.cap:
            self.cap.release()
        return False # propagate exceptions

# ------------------- Class VideoUtils -----------------

class VideoUtils:

    initialized_heic_reading = False

    @staticmethod
    def frame_to_jpeg(frame: np.ndarray, 
                      fpath: str, 
                      exif: Optional[dict[int, Any]] = None):
        '''
        Given a video frame, as returned by get_video_frame(), or cv2.read(),
        write the file to JPEG. The optional EXIF data will
        fields will be written

        :param frame: video frame to save as jpeg
        :type frame: np.ndarray
        :param fpath: destination file path
        :type fpath: str
        :raises IOError: on write failure
        '''
        # The function returns True if the file was successfully written, False otherwise.
        success = cv2.imwrite(fpath, frame)
        if not success:
            raise IOError(f"Could not save frame to file {fpath}")

    @staticmethod 
    def get_frame_fingerprint(frame: np.ndarray) -> np.ndarray:
        """
        Creates a color histogram fingerprint for a given frame.
        Using HSV is better than RGB as it separates luma (brightness) from chroma.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram: 
        # Channels [0, 1] = Hue and Saturation (we ignore Value/Brightness to be robust)
        # Bins [50, 60] = Granularity of the comparison
        # Ranges [0, 180, 0, 256] = Standard HSV ranges
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        # Normalize the histogram so image size doesn't matter
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def get_frame(path: str | Path, frame_num: int) -> np.ndarray:
        '''
        Given the path to a video, extract a given
        frame and return it. The frame_num start at 0.

        :param path: path to video file
        :param frame_num: the index into the frame sequence
        :return: the video frame
        :raise FileNotFoundError when file does not exist
        :raise IOError when cannot seek or read
        '''
        # Number of frames to undershoot if we overshoot
        # on first try in VFR video:
        SAFETY_BUF = 100

        if isinstance(path, Path):
            path = str(path)
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")
        
        with VideoStream(path) as cap:

            # Ensure frame_num not out of bound:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_num >= total_frames:
                msg = (f"File {path} only has {total_frames} frames; " 
                       f"requested frame {frame_num}")
                raise IndexError(msg)
            
            # Get to either the exact frame, or the next keyframe
            # below it (in most cases):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # Check where we actually landed
            # Note: get() returns the index of the NEXT frame to be decoded
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            success = False
            while True:
                # SCENARIO A: We landed exactly on target (Ideal)
                if current_pos == frame_num:
                    success = True
                    break
                elif current_pos < frame_num:
                    # SCENARIO B: we are at keyframe below target:
                    frames_to_skip = frame_num - current_pos
                    # 'Walk' from keyframe to target
                    for _ in range(frames_to_skip):
                        cap.grab() # Fast decode
                    success = True 
                    break
                else:
                    # In VFR file, and we overshot. Do it one more time,
                    # undershooting:
                    # Calculate a conservative target
                    # We aim 'SAFETY_BUF' frames early to avoid overshooting
                    seek_target = max(0, frame_num - SAFETY_BUF)
                    
                    # Perform the Seek (again):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                    
                    # Verify where we landed
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    # Still overshot. Handle the "Overshoot" (The VFR Disaster)
                    if current_pos > frame_num:
                        # We missed the target even with safety buf. Go back
                        # and walk all the way (expensive!)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        current_pos = 0
                        
                    # Walk the remaining distance ("The Neighborhood")
                    # This loop bridges the gap between our conservative seek and the target
                    frames_to_walk = frame_num - current_pos
                        
                    if frames_to_walk > 0:
                        for _ in range(frames_to_walk):
                            cap.grab()                    
                    success = True 
                    break

            if not success:
                raise IOError(f"Could not seek to frame {frame_num} in file {path}")
            
            success, frame = cap.read()
            if not success:
                raise IOError(f"Could not read frame {frame_num} from file {path}")
        return frame

    @staticmethod
    def show_frame(frame: np.ndarray):
        '''
        Show a frame array on the screen as an image.
        @see also show_file_frame()

        :param frame: frame to display
        :type frame: np.ndarray
        '''
        img = Image.fromarray(frame)
        img.show()

    @staticmethod
    def show_file_frame(path: str, frame_num: int):
        '''
        Given a file path and a 0-origin frame number,
        obtain the frame form the file, and display it
        in an OS-agnostic manner.

        If you have a frame in hand, use show_frame()
        instead of this method

        :param path: path to source file
        :type path: str
        :param frame_num: number of the frame to show
        :type frame_num: int
        '''
        frame = VideoUtils.get_frame(path, frame_num)
        VideoUtils.show_frame(frame)

    @staticmethod
    def read_img(fname: str) -> np.ndarray:
        '''
        Reads an image from file, and returns the 
        image data. Handles usual formats, including
        .heic.

        :param fname: full path to image
        :return: raw image data
        '''
        if Path(fname).suffix.lower() == '.heic' \
            and not VideoUtils.initialized_heic_reading:
            register_heif_opener()
        img = np.asarray(Image.open(fname))
        return img
    
    @staticmethod
    def resize_preserving_aspect(img, max_dim=640):
        """Resize so largest dimension is max_dim, preserving aspect ratio."""
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

