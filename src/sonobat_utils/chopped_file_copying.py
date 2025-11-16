#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2025-11-12 11:08:17
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2025-11-15 08:36:58

"""
Copy .wav files from Parsed_Files subdirectories to a parallel directory structure
with renamed files.
"""

import os
import sys
import re
import shutil
from pathlib import Path
from typing import List, Tuple
import argparse


class ChoppedFileCopier:
    """
    Copies .wav files from Parsed_Files subdirectories to a parallel directory
    structure with renamed files.
    """
    
    def __init__(self, src_root: Path, src_parsed_dir : Path, dst_root: Path, dst_suffix: str):
        """
        Initialize the copier.
        
        Args:
            src_root: Source root directory (e.g., /data/win_share/lake2)
            src_parsed_dir: of the SonoBat-generated parsed-files dir name (e.g. lake2_Parsed_Files)
            dst_root: Destination root directory (e.g., /data/win_share/choppedFiles)
            dst_suffix: Suffix to append to destination filenames (e.g., _2secs)
        """
        self.src_root = src_root
        self.src_parsed_dir = src_parsed_dir
        self.dst_root = dst_root
        self.dst_suffix = dst_suffix
        self.location_name = src_root.name
        self.uid = os.getuid()
        self.gid = os.getgid()
    
    def is_duplicate_file(self, filename: str) -> bool:
        """
        Check if a filename indicates a duplicate by having [<int>] before .wav
        Example: lake2_-20220103_012821[1].wav
        """
        pattern = r'\[\d+\]\.wav$'
        return bool(re.search(pattern, filename))
    
    def find_wav_files(self) -> List[Tuple[Path, str]]:
        """
        Find all valid .wav files to copy.
        Returns list of tuples: (source_file_path, date_directory_name)
        """
        wav_files = []
        
        # Iterate through potential date directories
        for item in self.src_root.iterdir():
            if not item.is_dir():
                continue
                
            # Check if directory name looks like a date (8 digits)
            if not re.match(r'^\d{8}$', item.name):
                continue
                
            date_dir = item.name
            
            # Construct path to Parsed_Files (no location subdirectory)
            parsed_files_dir = item / self.src_parsed_dir
            
            if not parsed_files_dir.exists() or not parsed_files_dir.is_dir():
                continue
                
            # Find .wav files directly in Parsed_Files (not in subdirectories)
            for wav_file in parsed_files_dir.iterdir():
                if not wav_file.is_file():
                    continue
                    
                if not wav_file.name.endswith('.wav'):
                    continue
                    
                # Skip duplicate files
                if self.is_duplicate_file(wav_file.name):
                    print(f"  Skipping duplicate: {wav_file}")
                    continue
                    
                wav_files.append((wav_file, date_dir))
        
        return wav_files
    
    def prepare_destinations(self, date_dirs: set) -> None:
        """
        Create all destination directories before copying.
        """
        self.dst_root.mkdir(parents=True, exist_ok=True)
        
        for date_dir in date_dirs:
            dest_dir = self.dst_root / f"{date_dir}{self.dst_suffix}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created destination directory: {dest_dir}")
    
    def copy_files(self, wav_files: List[Tuple[Path, str]]) -> None:
        """
        Copy all wav files to their destinations with renamed filenames.
        Uses shutil.copy() for speed and sets ownership to current user.
        """
        total = len(wav_files)
        
        for idx, (src_file, date_dir) in enumerate(wav_files, 1):
            # Create destination directory name
            dest_dir = self.dst_root / f"{date_dir}{self.dst_suffix}"
            
            # Create destination filename (insert suffix before .wav)
            src_name = src_file.name
            dest_name = src_name.replace('.wav', f"{self.dst_suffix}.wav")
            dest_path = dest_dir / dest_name
            
            # Copy the file (fast copy, doesn't preserve metadata)
            print(f"[{idx}/{total}] Copying: {src_file.name} -> {dest_path}")
            shutil.copy(src_file, dest_path)
            
            # Set ownership to current user:group
            os.chown(dest_path, self.uid, self.gid)
    
    def run(self, dry_run: bool = False) -> None:
        """
        Execute the copy operation.
        
        Args:
            dry_run: If True, show what would be copied without actually copying
        """
        print(f"Source root: {self.src_root}")
        print(f"Destination root: {self.dst_root}")
        print(f"Location name: {self.location_name}")
        print(f"Destination suffix: {self.dst_suffix}")
        print()
        
        # Find all .wav files to copy
        print("Finding .wav files...")
        wav_files = self.find_wav_files()
        
        if not wav_files:
            print("No .wav files found to copy.")
            return
        
        print(f"Found {len(wav_files)} .wav files to copy.")
        print()
        
        # Get unique date directories
        date_dirs = {date_dir for _, date_dir in wav_files}
        
        if dry_run:
            print("DRY RUN - Would create these destination directories:")
            for date_dir in sorted(date_dirs):
                print(f"  {self.dst_root / f'{date_dir}{self.dst_suffix}'}")
            print()
            print("DRY RUN - Would copy these files:")
            for src_file, date_dir in wav_files:
                dest_name = src_file.name.replace('.wav', f"{self.dst_suffix}.wav")
                print(f"  {src_file} -> {self.dst_root / f'{date_dir}{self.dst_suffix}' / dest_name}")
        else:
            # Create all destination directories
            print("Creating destination directories...")
            self.prepare_destinations(date_dirs)
            print()
            
            # Copy all files
            print("Copying files...")
            self.copy_files(wav_files)
            print()
            print(f"Done! Copied {len(wav_files)} files.")


def main():

    #sys.argv.append('--dry-run')
    #sys.argv.append('/data/win_share/lake2')
    #sys.argv.append('lake2_Parsed Files')
    #sys.argv.append('/data/win_share/lake2_choppedFiles')
    #sys.argv.append('_2secs')

    parser = argparse.ArgumentParser(
        description='Copy .wav files from Parsed_Files to parallel directory structure'
    )
    parser.add_argument('src_root', type=str, 
                       help='Source root directory (e.g., /data/win_share/lake2)')
    parser.add_argument('parsed_dir', type=str,
                       help='Auto-name of SonoBat-generated parsed-files directories. Default: "Parsed Files"',
                       default='Parsed Files')
    parser.add_argument('dst_root', type=str,
                       help='Destination root directory (e.g., /data/win_share/choppedFiles)')
    parser.add_argument('dst_suffix', type=str,
                       help='Suffix for all destination files (e.g., _2secs to indicate snippet lengths)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be copied without actually copying')
    
    args = parser.parse_args()
    
    src_root = Path(args.src_root)
    src_parsed_dir = Path(args.parsed_dir)
    dst_root = Path(args.dst_root)
    dst_suffix = args.dst_suffix
    
    # Create copier instance and run
    copier = ChoppedFileCopier(src_root, src_parsed_dir, dst_root, dst_suffix)
    copier.run(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
    