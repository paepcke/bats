#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2025-11-13 17:58:31
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2025-11-13 18:19:03

#!/usr/bin/env python3
"""
FileStager: Distribute date-named directories across multiple batch folders.
"""
import argparse
import shutil
from pathlib import Path
from typing import List


class FileStager:
    """Distributes date-named directories across batch input folders."""
    
    def __init__(self, source_dir: str, num_batches: int = 4, 
                 batch_base: str = "/data/win_share"):
        """
        Initialize the FileStager.
        
        Args:
            source_dir: Source directory containing date-named subdirectories
            num_batches: Number of batch folders to distribute to (default: 4)
            batch_base: Base path for batch folders (default: /data/win_share)
        """
        self.source_dir = Path(source_dir)
        self.num_batches = num_batches
        self.batch_dirs = [
            Path(batch_base) / f"batch{i}" / "input" 
            for i in range(1, num_batches + 1)
        ]
        
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
        for batch_dir in self.batch_dirs:
            if not batch_dir.exists():
                raise ValueError(f"Batch directory does not exist: {batch_dir}")
    
    def get_date_dirs(self, start_date: str = "20000101") -> List[Path]:
        """
        Get all date directories >= start_date.
        
        Args:
            start_date: Minimum date in YYYYMMDD format (default: 20000101)
            
        Returns:
            Sorted list of Path objects for date directories
        """
        date_dirs = []
        for item in sorted(self.source_dir.iterdir()):
            # Skip files and non-date directories
            if not item.is_dir():
                continue
            # Only keep directories that start with a digit and are >= start_date
            if item.name[0].isdigit() and item.name >= start_date:
                date_dirs.append(item)
        
        return date_dirs
    
    def distribute(self, start_date: str = "20000101", dry_run: bool = False):
        """
        Distribute date directories across batch folders.
        
        Args:
            start_date: Minimum date in YYYYMMDD format (default: 20000101)
            dry_run: If True, print actions without copying (default: False)
        """
        date_dirs = self.get_date_dirs(start_date)
        
        if not date_dirs:
            print(f"No directories found >= {start_date}")
            return
        
        print(f"Found {len(date_dirs)} directories to distribute")
        print(f"Date range: {date_dirs[0].name} to {date_dirs[-1].name}")
        
        if dry_run:
            print("\n*** DRY RUN - No files will be copied ***\n")
        
        # Distribute round-robin across batches
        for idx, date_dir in enumerate(date_dirs):
            batch_num = (idx % self.num_batches) + 1
            dest = self.batch_dirs[batch_num - 1] / date_dir.name
            
            action = "Would copy" if dry_run else "Copying"
            print(f"{action} {date_dir.name} -> batch{batch_num}/input/")
            
            if not dry_run:
                if dest.exists():
                    print(f"  Warning: {dest} already exists, skipping")
                    continue
                shutil.copytree(date_dir, dest)
        
        print("\nDistribution complete!" if not dry_run else "\nDry run complete!")
        for i in range(1, self.num_batches + 1):
            count = len(list(self.batch_dirs[i-1].iterdir()))
            print(f"Batch{i}: {count} directories")


def main():
    parser = argparse.ArgumentParser(
        description="Distribute date-named directories across batch folders"
    )
    parser.add_argument(
        "source_dir",
        help="Source directory containing date-named subdirectories"
    )
    parser.add_argument(
        "--start-date",
        default="20000101",
        help="Minimum date in YYYYMMDD format (default: 20000101)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Number of batch folders (default: 4)"
    )
    parser.add_argument(
        "--batch-base",
        default="/data/win_share",
        help="Base path for batch folders (default: /data/win_share)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying"
    )
    
    args = parser.parse_args()
    
    try:
        stager = FileStager(
            args.source_dir,
            num_batches=args.num_batches,
            batch_base=args.batch_base
        )
        stager.distribute(start_date=args.start_date, dry_run=args.dry_run)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
