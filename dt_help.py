#!/usr/bin/python3

import csv
import os
import pandas as pd
import time
import yaml

from datetime import timedelta
from functools import wraps

class Helper():
    def __init__(self, input_directory, input_prm_file):
        self.input_directory = input_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, input parameter file  = {}'.format(self.input_directory, self.input_prm_file))

    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
            
    @staticmethod
    def timing(f):
        """Decorator for timing functions
        Usage:
        @timing
        def function(a):
        pass
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' % (f.__name__,  end - start))
            return(result)
        return wrapper

    @staticmethod
    def get_delim(filename):
        with open(filename, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
        return(dialect.delimiter)
        
    @staticmethod
    def get_daterange(date1, date2):
        date1 = pd.to_datetime(date1, format='%Y-%m-%d').date()
        date2 = pd.to_datetime(date2, format='%Y-%m-%d').date()
        for n in range(int((date2 - date1).days)+1):
            yield(date1 + timedelta(n))

    @staticmethod
    def get_spec_date(date1, date2):
        weekdays = [1,2,3,4,5,7]
        dates_all = []
        for dt in Helper.get_daterange(date1, date2):
            if dt.isoweekday() in weekdays:
                dates_all.append(dt.strftime("%Y-%m-%d"))
        return(dates_all)
