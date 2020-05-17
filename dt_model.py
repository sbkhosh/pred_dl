#!/usr/bin/python3

import pandas as pd
import numpy as np

from dt_help import Helper
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

class DeepModelTS():
    def __init__(
        self, 
        data: pd.DataFrame, 
        Y_var: str,
        lag: int,
        n_ahead: int,
        LSTM_layer_depth: int, 
        epochs=10, 
        batch_size=256,
        train_test_split=0,
        optimizer='adam'            
    ):

        self.data = data 
        self.Y_var = Y_var 
        self.lag = lag
        self.n_ahead = n_ahead    
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.optimizer = self.get_optimizer('')

    @staticmethod
    def get_optimizer(flag):
        if(flag=='rms'):
            return(optimizers.RMSprop(lr=params["lr"]))
        elif(flag=='sgd'):
            return(optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True))
        elif(flag=='adam_tn'):
            return(optimizers.Adam(lr=params["lr"]))
        elif(flag==''):
            return('adam')
            
    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        X, Y = [], []

        if len(ts) - lag <= 0:
            X.append(ts)
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag])
                X.append(ts[i:(i + lag)])

        X, Y = np.array(X), np.array(Y)
        print(X.shape)
        # Reshaping the X array to an LSTM input shape 
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return(X, Y)
        
    @Helper.timing
    def create_data_for_NN(
        self,
        use_last_n=None
        ):

        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]

        # The X matrix will hold the lags of Y 
        X, Y = self.create_X_Y(y, self.lag)

        # Creating training and test sets 
        X_train = X
        X_test = []

        Y_train = Y
        Y_test = []

        if self.train_test_split > 0:
            index = round(len(X) * self.train_test_split)
            X_train = X[:(len(X) - index)]
            X_test = X[-index:]     
            
            Y_train = Y[:(len(X) - index)]
            Y_test = Y[-index:]

        return(X_train, X_test, Y_train, Y_test)

    @Helper.timing
    def LSTModel(self):
        # Getting the data 
        X_train, X_test, Y_train, Y_test = self.create_data_for_NN()

        # Defining the model
        model = Sequential()   
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.lag,1), return_sequences=False))
        # model.add(LSTM(self.LSTM_layer_depth//2, activation='relu', return_sequences=True))
        # model.add(LSTM(self.LSTM_layer_depth//4, activation='relu', return_sequences=False))
        # model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(optimizer=self.optimizer, loss='mse')

        # Early stopping and checkpoint
        estp = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        mchk = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        # Defining the model parameter dict 
        keras_dict = {
            'x': X_train,
            'y': Y_train,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'shuffle': False,
            'callbacks': [estp,mchk]
        }

        if self.train_test_split > 0:
            keras_dict.update({
                'validation_data': (X_test, Y_test)
            })

        # Fitting the model 
        history = model.fit(
            **keras_dict
        )

        # Saving the model to the class 
        self.model = model

        return(model, history)

    @Helper.timing
    def predict(self) -> list:
        yhat = []

        if(self.train_test_split > 0):
        
            # Getting the last n time series 
            _, X_test, _, _ = self.create_data_for_NN()        

            # Making the prediction list 
            yhat = [y[0] for y in self.model.predict(X_test)]

        return(yhat)

    @Helper.timing
    def predict_n_ahead(self, n_ahead: int):
        X, _, _, _ = self.create_data_for_NN(use_last_n=self.lag)        

        # Making the prediction list 
        yhat = []

        for _ in range(n_ahead):
            # Making the prediction
            fc = self.model.predict(X)
            yhat.append(fc)

            # Creating a new input matrix for forecasting
            X = np.append(X, fc)

            # Ommiting the first variable
            X = np.delete(X, 0)

            # Reshaping for the next iteration
            X = np.reshape(X, (1, len(X), 1))

        return(yhat)
