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

train = dd.read_csv('train.csv', engine='c', low_memory=False, parse_dates=['click_time','attributed_time']).fillna(0)
test = pd.read_csv('test.csv', engine='c').fillna(0)
# unix = pd.to_datetime(train['click_time'])
# train['click_time'] = unix.view('int64') / pd.Timedelta(1, unit='s')

def throwaway():
	global train
	# print(train.head())
	print(test.info())
throwaway()

def dataPreProcessTime(df):
    # df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    return df

def preprocess():
	global train
	global train_scaled
	global X_train
	global Y_train
	scaler = MinMaxScaler(feature_range=(0,1))
	train_scaled = scaler.fit_transform(train)

	X_train = []
	Y_train = []
	for i in range(100000, 184903889):
		X_train.append(train_scaled[i-100000:i, 0])
		Y_train.append(train_scaled[i, 0])
	X_train, Y_train = np.array(X_train), np.array(Y_train)

	X_train = np.reshape(X_train, X_train.shape[1], 8)



	model = Sequential()
	model.add(CuDNNLSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 8)))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(50, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(50, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(1, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.compile(loss="mean_squared_error", optimizer="RMSprop")
	model.fit(X_train, Y_train, epochs=100, batch_size=5000, verbose=2)



	test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
	print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

	model.save("trained_model.h5")
	print("Model saved to disk.")

def nn():
	X = pd.read_csv('scaled_train.csv').values
	df = pd.read_csv('train.csv')
	Y = df[['SalePrice']].values

	model = Sequential()
	model.add(Dense(50, input_dim=208, activation='relu', name='Input_Layer'))
	model.add(Dense(100, activation='relu', name='Hidden_Layer1'))
	model.add(Dense(50, activation='relu', name='Hidden_Layer2'))
	model.add(Dense(1, activation='linear', name='Output_Layer'))
	model.compile(loss="mean_squared_error", optimizer="adam")
	model.fit(X, Y, epochs=50, shuffle=True, verbose=2)

	X_test = pd.read_csv('scaled_test.csv')
	Y_test = Y[0:1459]

	test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
	print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

	model.save("trained_model.h5")
	print("Model saved to disk.")


# nn()

























