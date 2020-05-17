#!/usr/bin/python3

import datetime
import os
import pandas as pd
import yaml

from dt_help import Helper
from yahoofinancials import YahooFinancials

class DataProcessor():
    def __init__(self, input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))

    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
    
    @Helper.timing
    def read_tickers(self):
        self.ext_tick = self.conf.get('tickers_file').split('.')[1]
        self.base_tick = self.conf.get('tickers_file').split('.')[0]
        filename = os.path.join(self.input_directory,self.conf.get('tickers_file'))
        
        try:
            if('csv' in self.ext_tick):
                delim = Helper.get_delim(filename)
                data = pd.read_csv(filename)
                data.columns = [ ''.join(el for el in cl if el.isalnum()).lower() for cl in data.columns.values ]
                self.data_tickers = data
                self.tickers = self.data_tickers['ticker']
                self.description = self.data_tickers['description']
        except:
            raise ValueError("not supported format")
       
    @Helper.timing
    def process(self):
        start_date = self.conf.get('start_date')
        end_date = self.conf.get('end_date')
        end_date_future = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=self.conf.get('n_ahead'))
        end_date_future = str(end_date_future.date())

        date_range = Helper.get_spec_date(start_date, end_date)
        date_range_future = Helper.get_spec_date(start_date, end_date_future)
        diff_len = len(date_range_future) - len(date_range)
        
        values = pd.DataFrame({'Dates': date_range_future})
        values['Dates']= pd.to_datetime(values['Dates'])
        selected_tickers = self.conf.get('tickers_selected')
        
        for i in selected_tickers:
            raw_data = YahooFinancials(i)
            raw_data = raw_data.get_historical_price_data(start_date, end_date_future, "daily")
            df = pd.DataFrame(raw_data[i]['prices'])[['formatted_date','adjclose']]
            df.columns = ['Dates1',i]
            df['Dates1']= pd.to_datetime(df['Dates1'])
            values = values.merge(df,how='left',left_on='Dates',right_on='Dates1')
            values = values.drop(labels='Dates1',axis=1)

        values = values.fillna(method="ffill",axis=0)
        values = values.fillna(method="bfill",axis=0)
        cols = values.columns.drop('Dates')
        values[cols] = values[cols].apply(pd.to_numeric,errors='coerce').round(decimals=3)
        values.set_index('Dates',inplace=True)
        self.data = values[:-diff_len]
        self.data_future = values[-diff_len:]
        
    # @Helper.timing
    # def process(self):
    #     start_date = self.conf.get('start_date')
    #     end_date = self.conf.get('end_date')

    #     date_range = pd.bdate_range(start=start_date,end=end_date)
    #     values = pd.DataFrame({'Dates': date_range})
    #     values['Dates']= pd.to_datetime(values['Dates'])
    #     selected_tickers = self.conf.get('tickers_selected')
        
    #     for i in selected_tickers:
    #         raw_data = YahooFinancials(i)
    #         raw_data = raw_data.get_historical_price_data(start_date, end_date, "daily")
    #         df = pd.DataFrame(raw_data[i]['prices'])[['formatted_date','adjclose']]
    #         df.columns = ['Dates1',i]
    #         df['Dates1']= pd.to_datetime(df['Dates1'])
    #         values = values.merge(df,how='left',left_on='Dates',right_on='Dates1')
    #         values = values.drop(labels='Dates1',axis=1)

    #     values = values.fillna(method="ffill",axis=0)
    #     values = values.fillna(method="bfill",axis=0)
    #     cols = values.columns.drop('Dates')
    #     values[cols] = values[cols].apply(pd.to_numeric,errors='coerce').round(decimals=3)
    #     values.set_index('Dates',inplace=True)
    #     self.data = values
        
    def view_data(self):
        print(self.data.head())
        
    def drop_cols(self,col_names): 
        self.data.drop(col_names, axis=1, inplace=True)
        return(self)
               
    def write_to(self,name,flag):
        filename = os.path.join(self.output_directory,name)
        try:
            if('csv' in flag):
                self.data.to_csv(str(name)+'.csv')
            elif('xls' in flag):
                self.data.to_excel(str(name)+'xls')
        except:
            raise ValueError("not supported format")
               
    def save(self):
        pass
