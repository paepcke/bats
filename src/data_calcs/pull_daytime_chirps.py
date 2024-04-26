'''
Created on Apr 24, 2024

@author: paepcke
'''

from data_calcs.daytime_file_selection import DaytimeFileSelector
from logging_service.logging_service import LoggingService
from pathlib import Path
import argparse
import os
import sys
from sympy.liealgebras.type_e import TypeE

class DaytimeChirpPuller:
    '''
    Very specialized script to go through a list of 
    .csv SonoBat classification output files. For each
    .csv file, creates a new .csv file with only the
    daytime chirps.
    
    Resulting .csv files are stored in a given directory. 
    '''


    def __init__(self, 
                 dest_dir, 
                 csv_files,
                 target_format='feather',
                 file_name_column='Filename', 
                 postfix='_daytime'):
        '''
        If dest_dir does not exist, creates it.
        Then runs each of the csv_files through a
        DaytimeFileSelector. Creates .csv files with
        daytime chirps only in dest_dir. The string
        given in the postfix arg is appended to the name
        of each source file to create the destination csv
        file name.         
                
        :param dest_dir: destination directory of the daytime-only files
        :type dest_dir: str
        :param csv_files: list of absolute paths to csv files
        :type csv_files: (str)
        :param target_format: whether destination is to be
            .csv or .feather
        :type target_format: {'feather' | 'csv'}
        :param file_name_column: name of column that contains the 
            original's filename
        :type file_name_column: str
        :param postfix: string to append to orginial file names to
            create the corresponding daytime only copy
        :type postfix: str
        '''

        if target_format not in ('feather', 'csv'):
            raise TypeError(f"Target format must be 'feather' or 'csv', not {target_format}")

        self.target_format = target_format

        if not os.path.exists(dest_dir) or not os.path.isdir(dest_dir):
            # Allow group write permissions (umask is
            # subtracted from permission mode):
            os.umask(0o002)
            try:
                os.mkdir(dest_dir, mode='Oo775')
            except Exception as e:
                print(f"Could not created directory {dest_dir}: {e}")
                sys.exit(1)

        self.postfix = postfix
        self.file_name_col = file_name_column    
        self.log = LoggingService()
        
        # How many files to process before printing
        # number of files done:
        self.report_every = 50
        self.files_done = 0
        
        self.selector = DaytimeFileSelector()
        
        self._run(dest_dir, csv_files)

    #------------------------------------
    # run
    #-------------------

    def _run(self, dest_dir, csv_files):
        
        for fname in csv_files:
            src_path  = Path(fname)
            dst_suffix = '.csv' if self.target_format == 'csv' else '.feather'
            name_only = src_path.stem + self.postfix + dst_suffix
            dst_path  = os.path.join(dest_dir, name_only)
            self.selector.daytime_recordings(fname, dst_path, self.file_name_col)
            self.files_done += 1
            if self.files_done > self.report_every:
                self.log.info(f"Processed {self.files_done} csv files")
    
# ------------------------ Main ------------
if __name__ == '__main__':
    
    desc = ('Given a set of paths to SonoBat .csv lassification files,\n'
            'go through each .csv file, and find rows that represent\n'
            'daytime bat chirps. Copy those chirps to a given directory.\n'
            'The daytime-only files are named like their source files, with\n'
            'the string "daytime" appended to the stem.'
            )
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description=desc
                                     )

    parser.add_argument('infiles',
                        type=str,
                        nargs='+',
                        help='any number of paths to .csv files')

    parser.add_argument('dest_dir',
                        help='directory where daytime files are to be deposited; must exist'
                        )

    args = parser.parse_args()
    
    dest_dir = args.dest_dir
        