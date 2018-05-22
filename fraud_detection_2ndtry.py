import pandas as pd
import numpy as np
import time
# import os
# import webbrowser
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import dask.dataframe as dd

train = pd.read_csv('test_small.csv', engine='c', low_memory=False).fillna(0)
test = pd.read_csv('train_small.csv', engine='c', low_memory=False).fillna(0)
# test = pd.read_csv('test.csv', engine='c', low_memory=False).fillna(0)
# unix = pd.to_datetime(train['click_time'])
# train['click_time'] = unix.view('int64') / pd.Timedelta(1, unit='s')

def throwaway():
	global train
	# print(train.head())
	print(test.info())
# throwaway()

def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    return df

def feature_engineering():
	#How many unique ips are there?
	#Time difference between unique ips?
	#write NN functions for easy use like sentdex
	pass
train = dataPreProcessTime(train)
test = dataPreProcessTime(test)
def preprocess():
	global train
	global test
	global train_scaled
	
	# train['click_time'] = train['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
	# train['attributed_time'] = train['attributed_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
	# test['click_time'] = test['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
	scaler = MinMaxScaler(feature_range=(0,1))
	train_scaled = scaler.fit_transform(train)
	test_scaled = scaler.fit(test)

	scaled_train = pd.DataFrame(train_scaled, columns=train_scaled.columns.values)
	scaled_test = pd.DataFrame(test_scaled, columns=test_scaled.columns.values)
	scaled_train.to_csv('scaled_train.csv')
	scaled_test.to_csv('scaled_test.csv')

preprocess()
def nn():
	X = pd.read_csv('scaled_train.csv').values
	df = pd.read_csv('train_small.csv')
	Y = df[['is_attributed']].values


	model = Sequential()
	model.add(CuDNNLSTM(50, return_sequences=True, input_shape=(X.shape[1], 8)))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(50, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(50, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(1, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.compile(loss="mean_squared_error", optimizer="RMSprop")
	model.fit(X, Y, epochs=100, batch_size=5000, verbose=2)

	X_test = pd.read_csv('scaled_test.csv')
	Y_test = Y

	test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
	print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

	model.save("trained_model.h5")
	print("Model saved to disk.")



























