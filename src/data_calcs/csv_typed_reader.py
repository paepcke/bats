'''
Created on Apr 25, 2024

@author: paepcke
'''
import csv
from numba.core.types.misc import NoneType

class CSVTypedReader:
    '''
    A csv file reader for which clients may provide
    conversion functions
    '''

    #------------------------------------
    # Constructor
    #-------------------


    def __init__(self, fd, fieldnames=None, type_schema={}, **kwargs):
        '''
        Constructor
        '''
        if type(type_schema) not in (dict, NoneType):
            err_msg = f"Type schema must be None, or a dict of col name:function, not {type_schema}"
            raise TypeError(err_msg)

        self._fieldnames = fieldnames
        # Initialize fieldnames from first line of file,
        # if given fieldnames are None:
        self.fieldnames
        
        for key in type_schema.keys():
            if key not in self.fieldnames:
                err_msg = f"Keys in type_schema must be column names, not {key}"
                raise ValueError(err_msg) 

        # For speed, create a dict <column-number : <type-convert-func>
        self.convertion_dict = {}
        for i, col_name in enumerate(self._fieldnames.keys()):
            self.convert_dict[i] = self._fieldnames[col_name]

        self.reader = csv.reader(fd, kwargs)

    #------------------------------------
    # fieldnames (getter)
    #-------------------
        
    @property
    def fieldnames(self):
        if self._fieldnames is None:
            try:
                self._fieldnames = next(self.reader)
            except StopIteration:
                pass
        return self._fieldnames

    #------------------------------------
    # fieldnames (setter)
    #-------------------

    @fieldnames.setter
    def fieldnames(self, value):
        self._fieldnames = value
        
    #------------------------------------
    # __next__
    #-------------------
    
    def __next__(self):
        
        # Get next array of strings
        row = next(self.reader)
        if self.conversion__dict is not None:
            for key, convert_func in self.conversion_dict:
                element = row[key]
                row[key] = convert_func[element]
        return row

        
# ------------------------ Main ------------
if __name__ == '__main__':
    CSVTypedReader('/Users/paepcke/tmp/trash2.csv')        