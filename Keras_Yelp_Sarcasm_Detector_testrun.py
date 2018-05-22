import os, logging, threading, json
import pandas as pd
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, CuDNNLSTM
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
os.chdir(r'C:\Users\Dropbox\SentimentAnalysis\03_Reddit\RedditSarcasmTraininData\PrincetonData')
logging.basicConfig(filename='sentimentPreProcessPortfolioLog.log', level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def timefunc(func):
	from time import time
	def f(*args, **kwargs):
		start = time()
		rv = func(*args, **kwargs)
		finish = time()
		print('RUN TIME: ', finish - start)
		return rv
	return f

def clean_text(text):
	import string, re, nltk
	stopwords = nltk.corpus.stopwords.words('english')
	text = "".join([word for word in text if word not in string.digits])
	text = "".join([word for word in text if word not in string.punctuation])
	tokens = re.split('^|\W+', text)
	tokens = [word.lstrip() for word in tokens]
	text = [word for word in tokens if word not in stopwords]
	return text

@timefunc
def pre_process():
	df = pd.read_table('test-balanced_pol0.csv', encoding='latin2', header=None)
	df = df.drop([2, 3, 4, 5, 6, 7, 8, 9], axis=1)
	df.columns = ['sarcasm', 'body']
	df['body'] = df['body'].apply(lambda x: clean_text(x.lower()))

	train_x = [x[1] for x in df.values]
	train_y = np.asarray([x[0] for x in df.values]).astype('float64')

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(train_x)
	token_dictionary = tokenizer.word_index
	sequences = tokenizer.texts_to_sequences(train_x)
	max_sequence_length = [len(i) for i in sequences]
	print(np.max(max_sequence_length))
	data = pad_sequences(sequences)

	# scaler = MinMaxScaler(feature_range=(0, 1))
	# scaled_data = scaler.fit_transform(data)

	global X_train
	global y_train
	
	global X_test
	global y_test
	# X_train, X_test, y_train, y_test = train_test_split(scaled_data, train_y, test_size=0.3)
	X_train, X_test, y_train, y_test = train_test_split(data, train_y, test_size=0.3)

pre_process()
# thread_word2vec = threading.Thread(target=pre_process)
# thread_word2vec.start()

@timefunc
def LSTM_Keras():
	logging.basicConfig(filename='Reddit_LSTM_Keras.log', level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
	model = Sequential()
	model.add(Embedding(12032, 100, input_length=59))
#Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
	
	model.add(CuDNNLSTM(100, return_sequences=True))
#CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)

	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train, y_train, validation_split=0.4, epochs=3, verbose=2)

	binary_crossentropy = model.evaluate(X_test, y_test, verbose=0)
	print("The Binary Cross Entropy for the test data set is: {}".format(binary_crossentropy))
	logging.info(binary_crossentropy)
	model.save("Sarcasm_1_test.h5")
	print("Model saved to disk.")


# thread_word2vec = threading.Thread(target=LSTM_Keras)
# thread_word2vec.start()

@timefunc
def Dense_Keras():
	logging.basicConfig(filename='Dense_Reddit_Keras.log', level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
	model = Sequential()
	model.add(Dense(1000, input_dim=208, activation='relu', kernel_initializer='uniform', name='Input_Layer'))
	model.add(Dense(2000, activation='relu', kernal_initializer='uniform', name='Hidden_Layer1'))
	model.add(Dense(1000, activation='relu', kernal_initializer='uniform', name='Hidden_Layer2'))
	model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform', name='Output_Layer'))
	model.compile(loss="mean_squared_error", optimizer="adam")
	model.fit(X, Y, epochs=200, shuffle=True, verbose=2)

	binary_crossentropy = model.evaluate(X_test, y_test, verbose=0)
	print("The Binary Cross Entropy for the test data set is: {}".format(binary_crossentropy))
	logging.info(binary_crossentropy)
	model.save("Sarcasm_1_test.h5")
	print("Model saved to disk.")
	logging.log()	

# thread_word2vec = threading.Thread(target=pre_process1)
# thread_word2vec.start()

@timefunc
def CUDNN_LSTM_Keras():
#CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)	
	model = Sequential()
	model.add(Embedding(12032, 100, input_length=59))
	model.add(CuDNNLSTM(100, return_sequences=True, input_shape=(X.shape[1], 8)))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(100, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(100, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(CuDNNLSTM(1, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(Dense(1), activation='sigmoid')
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
	model.fit(X_train, y_train, validation_split=0.4, epochs=3, verbose=2)

	binary_crossentropy = model.evaluate(X_test, y_test, verbose=0)
	print("The Binary Cross Entropy for the test data set is: {}".format(binary_crossentropy))

	model.save("CUDNN_Sarcasm_1_test.h5")
	print("Model saved to disk.")

# thread_word2vec = threading.Thread(target=CUDNN_LSTM_Keras)
# thread_word2vec.start()













