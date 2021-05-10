import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import csv
import time
import sys

# Keras and Tensorflow
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM # Long Short-Term Memory (LSTM)
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D # import convnet
from keras.layers.convolutional import MaxPooling1D # import convnet
from keras.callbacks import ModelCheckpoint # save partially trained models after each epoch
from keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

from keras import backend as K # to verify we are using the GPU 

import FuncEtc as fn_etc
import FuncNormalizeInput as fn_normalize

print(sys.argv)
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
mdl_features_length = 1000  # how many samples (near-neighbours) do we have for our input data?
mdl_features_types = 1      # how many different features does our input data have?
                            # e.g. 1 = Distances only, 2 = x and y coords (or 2 = distances and angles)
mdl_labels = 1              # how many labels do we need to find, i.e. size of the final output?
                            # 1 = when doing binary classification, e.g. for two labels (not-clustered, clustered) 
                            # 3 = when doing multiple classifications, e.g. for three labels (non-clustered, clustered(round), clustered(fibre)), etc.
input_data_shape = (mdl_features_length, mdl_features_types)
# input_data_shape = (2000, 1)

def build_model():
    classifier_model = Sequential()
    classifier_model.add(Dense(32, activation='relu', input_shape=input_data_shape))
    classifier_model.add(Flatten())
    classifier_model.add(Dense(mdl_labels, activation='sigmoid'))
    return classifier_model


## CAML 'expanded' configuration (11 layers)

# def build_model():
#    classifier_model = Sequential()
#    classifier_model.add(Conv1D(filters=32, kernel_size=3, padding='valid', strides=1, activation='relu', input_shape=input_data_shape))
#    classifier_model.add(MaxPooling1D(pool_size=4))
#    classifier_model.add(Dropout(0.2))
#    classifier_model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
#    classifier_model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
#    classifier_model.add(Dropout(0.2))
#    classifier_model.add(MaxPooling1D(pool_size=4))
#    classifier_model.add(Flatten())
# #    classifier_model.add(Dense(100, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='RandomUniform'))
#    classifier_model.add(Dense(32, input_dim=60, kernel_initializer='normal', activation='relu'))
#    classifier_model.add(Dense(16, kernel_initializer='normal', activation='relu'))
#    classifier_model.add(Dense(mdl_labels, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='RandomUniform'))
#    return classifier_model



outputpath_model = 'model'
modelID = modelID_full = sys.argv[3]
model_summary_fname = os.path.join(outputpath_model, modelID_full)
model_fname = os.path.join(outputpath_model, modelID_full + '.h5')
auto_save_model = False        # Save a partially trained model after every epoch.
delete_autosaved_models = True # Delete partial models once all epochs complete.
log_to_csv = True              # Per-epoch performance can be saved to a CSV file.

callbacks_list = [] # we can pass several callbacks, our list starts out empty.

if  auto_save_model:
    model_partial_base='_Partial_Epoch-{epoch:02d}_ValAcc-{val_acc:.2f}_ValLoss-{val_loss:.2f}'
    model_partial_fname = os.path.join(outputpath_model, modelID + model_partial_base + '.h5')
    # can save for every epoch or can only save models which perform better than before
    ## This will save models only if they have better performance than previous epochs
    # checkpoint = ModelCheckpoint(model_partial_fname, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    # This will save the model after every epoch, regardless of performance improvement
    checkpoint = ModelCheckpoint(model_partial_fname, verbose=0)
    callbacks_list.append(checkpoint)

if log_to_csv:
    csv_logger = keras.callbacks.CSVLogger(os.path.join(outputpath_model, modelID + '_training_log.csv'))
    callbacks_list.append(csv_logger)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
callbacks_list.append(early_stopping)

mdl_batchsize = 32
mdl_epochs = 100
mdl_learning_rate = 0.01 # initial learning rate
mdl_epoch_factor = 1.0    # influence of epochs on learning rate decay. Must be greater than zero!
mdl_decay_rate = mdl_learning_rate / (mdl_epoch_factor * mdl_epochs) # zero = no change in learning rate; it's always 
opt_adam = keras.optimizers.Adam(lr=mdl_learning_rate, decay=mdl_decay_rate)
model = build_model()


model.compile(loss='binary_crossentropy', optimizer=opt_adam, metrics=['accuracy'])

with open(model_summary_fname + ' - Model Summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

pickle_file = open('train_data_2.pkl', 'rb')
names = pickle.load(pickle_file)
for i in names:
    locals()[i] = pickle.load(pickle_file)
x_train = x_train[:,int(sys.argv[2])]
x_val = x_val[:,int(sys.argv[2])]

x_train = np.memmap('/data0/yangqingyuan/cell_python/data_1000_norm/140205_DV3D_RIMA647_PSDCy3_2.MemMap', dtype='float64', mode='r')
x_train = x_train.reshape(x_train.shape[0]//2000,2,1000,1)[:,0]
y_train = (np.array(pd.read_table('data2/' + '140205_DV3D_RIMA647_PSDCy3_2' + '.txt', header=None))[:,4] != 0).reshape(x_train.shape[0],1)

training_history = model.fit(x_train, y_train, batch_size=mdl_batchsize, epochs=mdl_epochs,
                             validation_data=(x_val, y_val), callbacks=callbacks_list, shuffle=True)

model.save(model_fname, include_optimizer=False)  # saves model as HDF5 file.


# plt.plot(training_history.epoch, training_history.history['accuracy'], color='orange', alpha=0.5)
# plt.plot(training_history.epoch, training_history.history['val_accuracy'], color='cyan', alpha=0.5)
# plt.scatter(training_history.epoch, training_history.history['accuracy'], color='orange', s=1)
# plt.scatter(training_history.epoch, training_history.history['val_accuracy'], color='cyan', s=1)
# plt.ylim((0.5,1.0))



# g = os.walk(r"data3")
# total_cpus = multiprocessing.cpu_count() - 2
# for path, dir_list, file_list in g:
#     for file_name in file_list:
#         if file_name.split('.')[1] != 'pkl':
#             continue
#         data = pickle.load(open(os.path.join(path, file_name), 'rb'))
#         dists_mmapped = np.memmap('data4/' + file_name.split('.')[0] + '.MemMap', dtype='float64', shape=(data.shape[0], 2, 1000), mode='w+')
#         Parallel(n_jobs=total_cpus, verbose=3)(delayed(fn_distcalc_3D.dnns_v3)(data, data, dists_mmapped, i) for i in range(data.shape[0]))
#         del dists_mmapped