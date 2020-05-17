#!/usr/bin/python3

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import pandas as pd 
import tensorflow as tf
import warnings
import yaml

from datetime import datetime, timedelta
from dt_model import DeepModelTS
from dt_help import Helper
from dt_read import DataProcessor
from pandas.plotting import register_matplotlib_converters

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()

if __name__ == '__main__':
    obj_helper = Helper('data_in','conf_help.yml')
    obj_helper.read_prm()
    
    fontsize = obj_helper.conf['font_size']
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['text.color'] = 'k'

    obj_0 = DataProcessor('data_in','data_out','conf_model.yml')
    obj_0.read_prm()   
    obj_0.read_tickers()
    obj_0.process()

    yvar = obj_0.conf.get('yvar')
    df = obj_0.data[[yvar]]
    
    # Initiating the class 
    deep_learner = DeepModelTS(
        data=df, 
        Y_var=yvar,
        lag=obj_0.conf.get('lag'),
        n_ahead=obj_0.conf.get('n_ahead'),
        LSTM_layer_depth=obj_0.conf.get('LSTM_layer_depth'),
        epochs=obj_0.conf.get('epochs'),
        batch_size=obj_0.conf.get('batch_size'),
        train_test_split=obj_0.conf.get('train_test_split')
    )
    
    # Fitting the model
    model, history = deep_learner.LSTModel()  
    
    # Diagnostics
    if(obj_helper.conf.get('plot_loss')):
        plt.figure(figsize=(32,20))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        
    # Making the prediction on the validation set
    yhat = deep_learner.predict()

    if len(yhat) > 0:
        # Constructing the forecast dataframe
        fc = df.tail(len(yhat)).copy()
        fc.reset_index(inplace=True)
        fc['forecast'] = yhat
        
        plt.figure(figsize=(32,20))
        for dtype in [yvar, 'forecast']:
            plt.plot(
                'Dates',
                dtype,
                data=fc,
                label=dtype,
                alpha=0.8
            )
        # xticks = plt.xticks()[0]
        # xticklabels = [(fc['Dates'].iloc[0] + x).strftime('%Y-%m-%d') for x in xticks.astype(int)]
        # xtickslabels = [ (fc['Dates'].iloc[0]+timedelta(days=x)).strftime("%Y-%m-%d") for x in xticks.astype(int) ]
        # plt.xticks(xticks, xticklabels)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.show()   

    # Forecasting n steps ahead
    n_ahead = deep_learner.n_ahead
    yhat = deep_learner.predict_n_ahead(n_ahead)
    yhat = [y[0][0] for y in yhat]

    # Constructing the forecast dataframe
    fc = df.tail(10).copy() 
    fc['type'] = 'original'

    fc.reset_index(inplace=True)
    last_date = max(fc['Dates'])
    hat_frame = pd.DataFrame({
        'Dates': [last_date + timedelta(days=x + 1) for x in range(n_ahead)], 
        yvar: yhat,
        'type': 'forecast'
    })

    fc = fc.append(hat_frame)
    fc.reset_index(inplace=True, drop=True)

    # Ploting the forecasts 
    f, ax = plt.subplots(figsize=(32,20))
    for col_type in ['original', 'forecast']:
        plt.plot(
            'Dates', 
            yvar, 
            data=fc[fc['type']==col_type],
            label=col_type
        )
    plt.xticks(rotation=45)
    days = mdates.DayLocator(interval = 1)
    h_fmt = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(h_fmt)
    obj_0.data_future.plot(ax=ax,color='r')
    plt.legend()
    plt.grid()
    plt.show()
    
