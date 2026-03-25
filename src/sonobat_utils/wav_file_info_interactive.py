#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-23 10:04:56
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-23 10:27:33

import argparse
import os
from pathlib import Path
import sys

from sonobat_utils.wav_file_info import WavInfo

class WavFileDisplay:

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, wav_file_paths: list[str | Path]):
        for wav_file in wav_file_paths:
            info = WavInfo.from_path(wav_file)
            print(info)


def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Print info about one or more .wav files"
                                     )
    parser.add_argument('wavfiles',
                        nargs="+",
                        help='One or more .wav file paths'
    )
    args = parser.parse_args()
    paths = [Path(p) 
             for p 
             in args.wavfiles
             if os.path.exists(p)
             ]
    if len(paths) != len(args.wavfiles):
        missing = set(args.wavfiles) - set(map(lambda p: str(p), paths))
        print(f"These .wav files are missing: {missing}")
        sys.exit(1)
    return paths

if __name__ == "__main__":
    paths = main()
    WavFileDisplay(paths)
