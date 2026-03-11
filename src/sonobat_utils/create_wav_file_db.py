#!/usr/bin/env python
# **********************************************************
#
# @Author: Andreas Paepcke
# @Date:   2026-03-09 14:41:30
# @File:   /Users/paepcke/VSCodeWorkspaces/bats/src/sonobat_utils/organize_quintus_wav_files.py
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-10 15:06:31
#
# **********************************************************

"""
Highly specialized utility for creating an SQLite
database of bat chirp .wav file holdings. The db contains files 
that are marked in the filename as having been identified by SonoBat 
as being a chirp sequence of one bat species. 

It also separates chirps by location within Jasper Ridge. The
DB has two tables:

Table                              Columns                  Purpose
Locations                id (PK), folder_path, site_name    Stores unique directory paths only once.
Samples                  id (PK),                           The main registry of every .wav file.
                         filename, 
                         species_code,
                         recording_date,
                         location_id (FK), 
                         split                              

The species_code is the 4-letter code: Tabi, etc. The split
is for later use in ML training.                         
"""
import sqlite3
import os
import argparse
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Optional, Generator

from logging_service import LoggingService

# ------------------------ Class DBManager --------------------

class DBManager:
    """Handles SQLite schema creation and optimized batch insertions."""

    def __init__(self, db_path: str):
        self.log = LoggingService()
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._setup_schema()
        # Cache for folder_path -> location_id to minimize DB lookups
        self.location_cache: Dict[str, int] = {}

    def _setup_schema(self) -> None:
        """Initializes the normalized database schema."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS Locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_path TEXT UNIQUE,
                    site_name TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS Samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    species_code TEXT,
                    recording_date TEXT,
                    location_id INTEGER,
                    split TEXT DEFAULT 'unassigned',
                    FOREIGN KEY (location_id) REFERENCES Locations (id)
                )
            """)
            # Indexing for fast stratification and stats
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_species ON Samples(species_code)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_split ON Samples(split)")

    def get_or_create_location(self, folder_path: str, site_name: str) -> int:
        """Returns the ID for a location, inserting it if not present."""
        if folder_path in self.location_cache:
            return self.location_cache[folder_path]

        cursor = self.conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO Locations (folder_path, site_name) VALUES (?, ?)", (folder_path, site_name))
        self.conn.commit()
        
        cursor.execute("SELECT id FROM Locations WHERE folder_path = ?", (folder_path,))
        loc_id = cursor.fetchone()[0]
        self.location_cache[folder_path] = loc_id
        return loc_id

    def insert_samples_batch(self, batch: List[Tuple[str, str, str, int]]) -> None:
        """Bulk inserts a list of samples within a single transaction."""
        query = """
            INSERT INTO Samples (filename, species_code, recording_date, location_id)
            VALUES (?, ?, ?, ?)
        """
        try:
            with self.conn:
                self.conn.executemany(query, batch)
        except sqlite3.Error as e:
            self.log.err(f"Error during batch insert: {e}")

    def close(self):
        self.conn.close()

# ------------------------ Class FileCrawler --------------------

class FileCrawler:
    """Traverses the file system and parses metadata from paths."""

    def __init__(self, root_dir: str, batch_size: int = 10000, verbosity: int = -1):
        self.root_dir = str(Path(root_dir).resolve())
        self.verbosity = verbosity
        self.batch_size = batch_size
        self.log = LoggingService()

    #------------------------------------
    # stream_wildlife_files
    #-------------------

    def stream_wildlife_files(self, 
                              batch_size: int = 10000
                              ) -> Iterator[tuple[str, list[tuple[str, str, str, str]]]]:
        '''

        
        Note: this method must be fast, as we will be processing over a million
              file names at a time. Therefore the use of primitive datatypes.

        Runs through the file system, starting at root_dir. Finds .wav files with
        a particular format:

            barn/SMB_BARN_BATS/2022/20220327/barn1_D20220327T014116m521-Epfu.wav
            jasperride/lake2/grouped_audio/20131222_to_20131225/20131225/lake2_PST_D20131225T172506m144-Tabr.wav
            jasperridge/barn1/grouped_audio/20220617_to_20220620/20220620/barn1_D20220620T231016m607-Laci.wav
        
        It skips files that are either not .wav files, or that do not have
        the bat species spec at the end of the filename stem.
        
        Generates for each next() call:
             (directory-name, [
                                (wav-filename, recording-location, isotime-str, species),
                                (wav-filename, recording-location, isotime-str, species),
                                       ... <batch_size tuples>
                              ]
             )

        Each returned list will be up to batch_size entries.

        Example:
         [
           (barn1_D20220327T014116m521-Epfu.wav, 'barn1', '20220327T014116m521', 'Epfu'),
           (lake2_PST_D20131225T172506m144-Tabr.wav, 'lake2', '20131225T172506m144', 'Tabr')
                ...
         ]

        :param root_dir: start of file system crawl
        :param batch_size: how many result tuples to return on each next(), defaults to 10000
        :raises ValueError: if a .wav file does not contain location, date, and species info
        :yield: -> list[tuple[str, str, str]]
        '''

        # Move the search logic into a local variable for faster access
        filename: str
        current_batch = []
        for root, _dirs, files in os.walk(self.root_dir):
            for filename in files:
                if not filename.endswith('.wav'):
                    continue
                
                try:
                    # 1. Strip .wav
                    stem = filename[:-4]

                    # 2. Check suffix: dash + 4 letters
                    if stem[-5] != '-' or not stem[-4:].isalpha():
                        # Wav file has not been species-identified by SonoBat or other means:
                        continue
                    else:
                        # Grab the species:
                        species = stem[-4:]

                    # 3. Find prefix (before first underscore): recording location
                    first_underscore = stem.find('_')
                    if first_underscore == -1:
                        raise ValueError(f"No underscore: cannot find recording location at start of filename: {filename}")
                    recording_location = stem[:first_underscore]

                    # 4. Extract ISO (Search for the first digit after the prefix)
                    # This ignores '_D', '_PST_D', or just '_'
                    idx = first_underscore + 1
                    while idx < len(stem) and not stem[idx].isdigit():
                        idx += 1
                    
                    last_dash = stem.rfind('-')
                    iso_ts = stem[idx:last_dash]
                    
                    if not iso_ts:
                        raise ValueError(f"Cannot find timestamp in {filename}")

                    current_batch.append((filename, recording_location, iso_ts, species))
                    if len(current_batch) >= batch_size:
                        yield root, current_batch
                        current_batch = []


                except ValueError:
                    # Handle or log the error
                    continue

        # Final, partially filled batch
        if current_batch:
            yield root, current_batch

# ------------------------ Class DataConcentrator --------------------

class DataConcentrator:
    """Orchestrates the crawling and DB insertion process."""

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 root_dir: str, 
                 db_path: str,
                 batch_size: int = 10000,
                 verbosity: int = -1
                 ):
        self.verbosity = verbosity
        self.batch_size = batch_size
        self.log = LoggingService()
        self.crawler = FileCrawler(root_dir, verbosity)
        self.db = DBManager(db_path)

    #------------------------------------
    # run
    #-------------------
    
    def run(self):
        """Executes the full scan and population of the database."""
        if self.verbosity > -1:
            self.log.info(f"Scanning {self.crawler.root_dir}...")
        total_count = 0

        old_count_multiple = 0
        batch: list[tuple[str,str,str]]
        for batch_dir, batch in self.crawler.stream_wildlife_files(batch_size=self.batch_size):
            db_payload = []
            for (fname, site_name, date,  species) in batch:
                loc_id = self.db.get_or_create_location(batch_dir, site_name)
                db_payload.append((fname, species, date, loc_id))
            
            self.db.insert_samples_batch(db_payload)
            total_count += len(db_payload)

            # Should be report?            
            if self.verbosity > -1:
                new_count_multiple = total_count // self.verbosity
                if new_count_multiple > old_count_multiple:
                    self.log.info(f"Indexed {total_count} files...")
                    old_count_multiple = new_count_multiple

        if self.verbosity > -1:
            self.log.info(f"Finished. Total files indexed: {total_count}")

        self.db.close()

# ------------------- Main Section ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index bat chirps into SQLite.")
    parser.add_argument("root", help="Root directory containing .wav files")
    parser.add_argument("--db", default="bat_data.db", help="Output SQLite database name")
    parser.add_argument('-b', '--batch_size', 
                        type=int,
                        default=10000, 
                        help="number of files to collect for insertion into db in one transaction")
    parser.add_argument('-v', '--verbose', 
                        nargs='?',
                        type=int,
                        default=-1,  # Value if flag is absent
                        const=50000, # Value if flag present w/o a number
                        help=("Print progress every n .wav files. \n" 
                              "    If flag present, but no value: every 50000; \n" 
                              "    If flag present and value <n>; then every <n>; \n"
                              "    if flag absent: no progress report")
    )
    args = parser.parse_args()

    concentrator = DataConcentrator(args.root, args.db, batch_size=args.batch_size, verbosity=args.verbose)
    concentrator.run()
