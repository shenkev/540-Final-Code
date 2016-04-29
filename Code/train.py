from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.layers.recurrent import LSTM
from keras.utils.visualize_util import plot, to_graph
from keras.regularizers import l2, activity_l2
import copy

import utils as util
import model as mod

# This script trains a LSTM + NN combined model and calculates the validation error.
# Inputs:
#   1. training_file: path to the file for training the neural network
#   2. testing_file: path to the file for testing the neural network

# ------------------------------------- Main Loop --------------------------------------------
# Retrieve the data
lstm_length = 21
training_file = "./../data/sineAndJump.txt"
testing_file = "./../data/testFile.txt"
[X_train, y_train, X_test, y_test, NN_train, NN_test]=util.get_data(training_file, lstm_length, 1, testing_file)

# Get rid of Y and REF because lstm doesn't want to train on this
X_train = X_train[:, :, 0:1]
X_test = X_test[:, :, 0:1]

# define the input sizes for the LSTM
lstm_data_dim = X_train.shape[2]
nn_data_dim = NN_train.shape[1]
timesteps = lstm_length

# construct and compile the model
model = mod.design_model(lstm_data_dim, nn_data_dim, timesteps)
start_time = time.time()
print "Compiling Model ..."
model.compile(loss="mse", optimizer="rmsprop")
print("Compile Time : %s seconds --- \n" % (time.time() - start_time))

# train the model
my_batch_size = 512
my_epoch = 5
start_time = time.time()
model.fit([X_train, NN_train], y_train, batch_size=my_batch_size, nb_epoch=my_epoch)
print("Training Time : %s seconds --- \n" % (time.time() - start_time))

# Validating the model
U_hat = model.predict([X_test, NN_test], verbose=1)
U_hat = U_hat.reshape((len(U_hat)))
loss_and_metrics = model.evaluate([X_test, NN_test], y_test[:, 0])
print "test error is: ", loss_and_metrics

# plot the predicted versus the actual U values
toPlot = np.column_stack((U_hat, y_test[:, 0]))
plt.plot(toPlot)
plt.show()