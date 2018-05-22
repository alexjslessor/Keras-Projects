import pandas as pd
import numpy as np
import os
import webbrowser
from keras.models import Sequential
from keras.layers import *
from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
os.chdir(r'C:\Users\Neo\Anaconda3\envs\tensorflow\house_prediction_data')
# train = pd.read_csv('train.csv').fillna(0)
# Y = train[['SalePrice']].values

# test = pd.read_csv('test.csv').fillna(0)

def unique_values():
  for column in train.columns, test.columns:
    try:
    	pd.set_option('display.max_rows', None)
    	print(train[column].nunique(), test[column].nunique(), flush=True)
    except KeyError:
    	print('error')
# unique_values()

def feature_labels():
    pd.set_option('display.max_columns', None)
    df = list(train_df.columns)
    print(df)

def open_data_as_html():
	html = train[0:1450].to_html()
	with open("train.html", "w") as f:
		f.write(html)
	full_filename = os.path.abspath("train.html")
	webbrowser.open("file://{}".format(full_filename))

def drop_columns():
	global train
	global test
	global train_df
	train_df = train.drop(['Id','MSZoning','Condition2','HouseStyle','RoofMatl','Exterior1st','Heating','Electrical','KitchenQual','Functional','GarageQual','PoolQC',\
		'Fence','MiscFeature','SaleType','SalePrice'], 1)
	test = test.drop(['Id','MSZoning','Condition2','HouseStyle','RoofMatl','Exterior1st','Heating','Electrical','KitchenQual','Functional','GarageQual','PoolQC',\
		'Fence','MiscFeature','SaleType'], 1)
# drop_columns()

def process_dummies():
	global test
	global train_df
	train_df = pd.get_dummies(train_df, columns=['Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',\
		'BldgType','RoofStyle','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',\
		'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','FireplaceQu',\
		'GarageType','GarageFinish','GarageCond','PavedDrive','SaleCondition'])
	# train.to_csv('train_dummies.csv')
	test = pd.get_dummies(test, columns=['Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',\
		'BldgType','RoofStyle','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',\
		'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','FireplaceQu',\
		'GarageType','GarageFinish','GarageCond','PavedDrive','SaleCondition'])
	# test.to_csv('test_dummies.csv')
['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
# process_dummies()

def scale_dummies():
	global test
	global train_df
	# train_dummies = pd.read_csv('train_dummies.csv')
	# test_dummies = pd.read_csv('test_dummies.csv')
	
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_training = scaler.fit_transform(train_df)
	scaled_testing = scaler.transform(test)
	print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

	scaled_train = pd.DataFrame(scaled_training, columns=scaled_training.columns.values)
	scaled_test = pd.DataFrame(scaled_testing, columns=scaled_testing.columns.values)
	print(scaled_train.info())
	print(scaled_test.info())
	scaled_train.to_csv('scaled_train.csv')
	scaled_test.to_csv('scaled_test.csv')
# scale_dummies()

def nn():
	X = pd.read_csv('scaled_train.csv').values
	df = pd.read_csv('train.csv')
	Y = df[['SalePrice']].values


	model = Sequential()
	model.add(Dense(1000, input_dim=208, activation='relu', name='Input_Layer'))
	model.add(Dense(2000, activation='relu', name='Hidden_Layer1'))
	model.add(Dense(1000, activation='relu', name='Hidden_Layer2'))
	model.add(Dense(1, activation='linear', name='Output_Layer'))
	model.compile(loss="mean_squared_error", optimizer="adam")
	model.fit(X, Y, epochs=200, shuffle=True, verbose=2)


	X_test = pd.read_csv('scaled_test.csv')
	Y_test = Y[0:1459]

	test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
	print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

	model.save("trained_model.h5")
	print("Model saved to disk.")


# nn()

























