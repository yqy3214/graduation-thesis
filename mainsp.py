import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

## parameter
# dbscan 中的 \epsilon & minpts
k = 60  # 60
Eps = 1.2  # 1.2

## threshold
# sh_e = 1.2 # 长轴/短轴
sh_dc = 5  # 中心距离
sh_x = 2  # x轴跨度
sh_y = 2  # y轴跨度
sh_x_r = 0.4  # x轴跨度
sh_y_r = 0.4  # y轴跨度
sh_m = 150  # 包含点数

data = np.array(pd.read_table(
    'data/140205_DV3D_RIMA647_PSDCy3_2.txt', header=None))
# result = np.array(pd.read_table(
#     'data/140205_DV3D_RIMA647_PSDCy3_2/synapse.txt', header=None))

# print(type(data), data)
# print(type(data[:, 3] == 647))

x1 = data[data[:, 3] == 647, : 3]
x2 = data[data[:, 3] == 561, : 3]

class1 = DBSCAN(eps=Eps, min_samples=k).fit_predict(x1)
class2 = DBSCAN(eps=Eps, min_samples=k).fit_predict(x2)
cluster_size1 = max(class1)
cluster_size2 = max(class2)
class1 = class1 - 1
class2 = class2 - 1

# plt.scatter(x1[y1.ravel(), 0], x1[y1.ravel(), 1], c=y1, marker='.', s=1)
plt.figure()

center1 = np.zeros((cluster_size1, 2))
center2 = np.zeros((cluster_size2, 2))
range1 = np.zeros((cluster_size1, 2))
range2 = np.zeros((cluster_size2, 2))
quantity1 = np.zeros(cluster_size1)
quantity2 = np.zeros(cluster_size2)

for i in range(cluster_size1):
    x_clu = x1[class1 == i, : 2]
    Xmax = max(x_clu[:, 0])
    Xmin = min(x_clu[:, 0])
    Ymax = max(x_clu[:, 1])
    Ymin = min(x_clu[:, 1])
    range1[i] = [Xmax - Xmin, Ymax - Ymin]
    quantity1[i] = np.size(x_clu, 0)
    center1[i] = np.mean(x_clu)

for i in range(cluster_size2):
    x_clu = x2[class2 == i, : 2]
    Xmax = max(x_clu[:, 0])
    Xmin = min(x_clu[:, 0])
    Ymax = max(x_clu[:, 1])
    Ymin = min(x_clu[:, 1])
    range2[i] = [Xmax - Xmin, Ymax - Ymin]
    quantity2[i] = np.size(x_clu, 0)
    center2[i] = np.mean(x_clu)
    
    # plt.scatter(x1[y1 == i, 0], x1[y1 == i, 1], c='r', marker='.')
D = cdist(center1, center2)
print(type(D))
print(np.max(D, axis=1), np.where(D == np.array([1,2,3])))
# print(np.where(a == np.max(a, axis=0)))
# (array([2, 2, 2], dtype=int64), array([0, 1, 2], dtype=int64))


tmp = np.logical_and(data[:, 3] == 647, data[:, 4] != 0)
plt.scatter(data[tmp, 0], data[tmp, 1], c=data[tmp, 4], marker='.')

plt.show()
