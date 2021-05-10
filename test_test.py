import os
import numpy as np
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

pickle_file = open('train_data_2.pkl', 'rb')
names = pickle.load(pickle_file)
for i in names:
    locals()[i] = pickle.load(pickle_file)
x_train = x_train[:,int(sys.argv[2])]
x_val = x_val[:,int(sys.argv[2])]

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

# model = load_model('/data0/yangqingyuan/cell_python/model/4_layers_test.h5') # [[46448, 2764], [14432, 1972]]
# model = load_model('/data0/yangqingyuan/cell_python/trained_model/3D/GAXJPR (3D, 1000 NN)/GAXJPR - Norm-Self Train(500.0k,0.5×Clus) Val(100.0k,0.5×Clus).h5') # [[22870, 26342], [7078, 9326]]
model = load_model('/data0/yangqingyuan/cell_python/model/test model.h5') # [[22870, 26342], [7078, 9326]]

y = model.predict(x_val, batch_size=64, verbose=1)
y = [float(np.round(x)) for x in y]

a = [[0,0],[0,0]]
for i in range(len(y)):
    a[int(y_val[i,0])][int(y[i])] += 1
print(a)

