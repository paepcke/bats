#!/usr/bin/env python3
'''
Created on May 28, 2021

@author: Nick Gardner

Note: the #! line above allows this file to be
      run just by naming it, rather than having
      to explicitly specify python on the command
      line:
       
          greek_name_classifier.py
      instead of
          python greek_name_classifier.py'

      But if used, must be first line in file.

'''
import argparse
import os
import sys


class GreekNameClassifier(object):
    '''
    Classifies ancient Greek names across
    a set of professions.
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, name_file, filter=None, unittesting=False):
        '''
        Sanity checks, initialize data structures
        
        :param name_file: name of file containing names
        :type name_file: str
        :param filter: list of statuses to ignore
        :type filter: {None | [str]}
        :param unittesting: if True, returns without any action
        :type unittesting: bool
        '''
        
        if unittesting:
            # Allow unit tests to individually call methods
            # on this instance:
            return

        if not os.path.exists(name_file):
            raise FileNotFoundError(f"Name file {name_file} not found")
        
        self.filter = filter
        self.syll_dict = self.syllabify(name_file)
    
    #------------------------------------
    # syllabify 
    #-------------------

    def syllabify(self, name_file):
        '''
        Given a file with a name on each line,
        return a dict mapping each name to a list
        of syllables:
        
               {nm1 : [syll0, syll1, ...],
                nm2 : [syll0, syll1, ...],
                    ...
               }
               
        :param name_file: absolute path to name file
        :type name_file: str
        :return dict mapping names to syllable lists
        :rtype {str : [str]}
        '''
        
        syll_dict = {}
        with open(name_file, 'r') as nm_fd:
            for name in nm_fd:
                # Name will have a trailing newline; remove that:
                name = name.strip()
                syll_dict[name] = self.extract_syllables(name)
            
        return syll_dict
    
    #------------------------------------
    # extract_syllables 
    #-------------------
    
    def extract_syllables(self, name):
        '''
        Given a name, return a list of 
        its constitutent syllables
        
        :param name: name to syllabify
        :type name: str
        :return: list of syllables
        :rtype: [str]
        :raise ValueError if name cannot be syllabified
        '''
        if name == 'labile':
            return ['la', 'bile']
        elif name == 'facile':
            return ['fa', 'cile']
        else:
            raise ValueError(f"Unknown name '{name}'")

# ------------------------ Main ------------
if __name__ == '__main__':
    
    '''
    Use the argparse package to create an 
    automatic Help facility, and command line
    argument parsing.
    
    Example add_argument types:
       o Type constraint  : type=int
       o Default setting  : default="Foobar"
       o Choice of args   : choices=['foo', 'bar', 'fum']
       o Multiple entries : nargs='+'

    The nargs allows multiple args on the command line.
    Example: 'as many as user wants:

        parser.add_argument('-s', --status',
                            nargs='*'
                            help='a filter: list of status output to include'
                            )

      To use on command line:
      
          greek_name_classifier.py --status tinker tailor soldier -- my_namefile.txt
          
      Note that the '--' is needed to let the parser know 
           the end of the list.
          
    '''
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Predicts societal status from ancient Greek names"
                                     )


    parser.add_argument('-f', '--filter',
                    nargs='*', 
                    help='list of status outputs to ignore',
                    default=None
                    )

    parser.add_argument('namefile',
                        help='fully qualified name of file with names to classify'
                        )

    args = parser.parse_args()

    GreekNameClassifier(args.namefile, filters=args.filter)
        