from keras.models import load_model
import numpy as np
import pandas as pd
import FuncEtc as fn_etc
import FuncNormalizeInput as fn_normalize
import FuncClusterClub as fn_cluclu
import matplotlib.pyplot as plt
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

data = np.array(pd.read_table(
    'data2/140531_DV3D_RIMA647_PSDCy3_PTX_1.txt', header=None))
tmp = data[:, 1] != 0
tmp = np.logical_and(data[:, 3] == 647, data[:, 4] != 0)
plt.scatter(data[tmp, 0], data[tmp, 1], c=data[tmp, 4], marker='.')
plt.savefig('img/testaaa.png')

# model_fname = '/data0/yangqingyuan/cell_python/model/test model.h5'
model_fname = 'trained_model/3D/GAXJPR (3D, 1000 NN)/GAXJPR - Norm-Self Train(500.0k,0.5×Clus) Val(100.0k,0.5×Clus).h5'
model = load_model(model_fname)
model_config = model.get_config()
pred_threshold = 0.5
import_dists_f = '/data0/yangqingyuan/cell_python/data_1000/140531_DV3D_RIMA647_PSDCy3_PTX_1.MemMap'
Dists_all_New = np.memmap(import_dists_f, dtype='float64', mode='r')
print(Dists_all_New)
Dists_all_New = Dists_all_New.reshape(Dists_all_New.shape[0]//2000, 2, 1000)
X_novel_1 = Dists_all_New[:,0,:]
X_novel_2 = Dists_all_New[:,1,:]
X_novel_1 = fn_normalize.normalize_dists(X_novel_1, Dists_all_New.shape[0], 'Norm-Self')
X_novel_2 = fn_normalize.normalize_dists(X_novel_2, Dists_all_New.shape[0], 'Norm-Self')

# X_novel_1 = Dists_all_New[data[:, 4] != 0,0,:]
# X_novel_2 = Dists_all_New[data[:, 4] != 0,1,:]
# X_novel_1 = fn_normalize.normalize_dists(X_novel_1, Dists_all_New.shape[0], 'Norm-Self')
# X_novel_2 = fn_normalize.normalize_dists(X_novel_2, Dists_all_New.shape[0], 'Norm-Self')


novel_probabilities_1 = model.predict(X_novel_1, batch_size=64, verbose=1)
novel_probabilities_2 = model.predict(X_novel_2, batch_size=64, verbose=1)
print(novel_probabilities_2)
novel_predictions_1 = [float(np.round(x - (pred_threshold - 0.5))) for x in novel_probabilities_1]
novel_predictions_2 = [float(np.round(x - (pred_threshold - 0.5))) for x in novel_probabilities_2]


# print(np.array(novel_predictions_1)*np.array(novel_predictions_2))


tmp = (np.array(novel_predictions_1)*np.array(novel_predictions_2)) == 1
# tmp = np.array(novel_predictions_1) == 1
# tmp = np.logical_and(tmp, data[:, 4] != 0)
plt.close()
plt.scatter(data[tmp, 0], data[tmp, 1], c=data[tmp, 4], marker='.')
plt.savefig('img/test10.png')
# print(novel_predictions_2)