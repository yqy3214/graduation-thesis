import os
import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
import FuncDistCalcs_3D as fn_distcalc_3D
import multiprocessing


## 得到最近邻
g = os.walk(r"data3")
total_cpus = multiprocessing.cpu_count() - 2
for path, dir_list, file_list in g:
    for file_name in file_list:
        if file_name.split('.')[1] != 'pkl':
            continue
        data = pickle.load(open(os.path.join(path, file_name), 'rb'))
        dists_mmapped = np.memmap('data4/' + file_name.split('.')[0] + '.MemMap', dtype='float64', shape=(data.shape[0], 2, 1000), mode='w+')
        Parallel(n_jobs=total_cpus, verbose=3)(delayed(fn_distcalc_3D.dnns_v3)(data, data, dists_mmapped, i) for i in range(data.shape[0]))
        del dists_mmapped


## 统计数据
# g = os.walk(r"data")
# point = 0
# tuchu = 0
# point_t = 0
# pic_num = 0
# for path, dir_list, file_list in g:
#     for file_name in file_list:
#         if file_name == 'localized_3d_calibrated.txt':
#             f = open(os.path.join(path, file_name))
#             point += len(f.readlines())
#             pic_num += 1
#         if file_name == 'synapse.txt':
#             f = open(os.path.join(path, file_name)).readlines()
#             point_t += len(f)
#             tuchu += float(f[-2].split()[0])
#             print(float(f[-2].split()[0]), path)
# print(point, tuchu, point_t, pic_num)
# print(point/pic_num)
# print(tuchu/pic_num)
# print(point_t/tuchu)
# print(point/point_t)


##　合并数据
# g = os.walk(r"data")
# data = 0
# result = 0
# for path, dir_list, file_list in g:
#     for file_name in file_list:
#         if file_name == 'localized_3d_calibrated.txt':
#             data = np.array(pd.read_table(
#                 os.path.join(path, file_name), header=None))
#             print(data.shape)
#             data = np.c_[data, np.zeros((data.shape)[0])]
#         if file_name == 'synapse.txt':
#             result = np.array(pd.read_table(
#                 os.path.join(path, file_name), header=None))
#     if len(file_list) != 0:
#         for i in result:
#             idx = (np.abs(data[:, :4] - i[1:]).sum(axis=1)).argmin()
#             data[idx][4] = i[0]
#             if np.abs(data[idx][:4] - i[1:]).sum() > 0.001:
#                 print(data[idx][:4] - i[1:])
#         print(path)
#         # input()
#         p = open(path + '.txt', 'w')
#         for i in data:
#             for j in i[:-1]:
#                 p.write('%.6e' % j + '\t')
#             p.write('%.6e' % i[-1] + '\n')



# ##　修改数据存储方式
# g = os.walk(r"data2")
# for path, dir_list, file_list in g:
#     for file_name in file_list:
#         data = np.array(pd.read_table(
#             os.path.join(path, file_name), header=None))
#         datatable_mmapped = np.memmap(
#             'data4/' + file_name.split('.')[0] + '.MemMap',
#             dtype='float64',
#             shape=data.shape,
#             mode='w+')
#         datatable_mmapped = data
#         del datatable_mmapped
#         # pickle_file = open('data3/' + file_name.split('.')[0] + '.pkl', 'wb')
#         # pickle.dump(data, pickle_file)
#         # pickle_file.close()


## 统一数据格式
# g = os.walk(r"training_data")
# for path, dir_list, file_list in g:
#     p_path = path.replace('training_data', 'data')
#     for i in dir_list:
#         if not os.path.exists(os.path.join("data/", i)):
#             os.mkdir(os.path.join("data/", i))
#     for file_name in file_list:
#         print(path)
#         f = open(os.path.join(path, file_name))
#         p = open(p_path + '/' + file_name, 'w')
#         if file_name == 'synapse.txt':
#             a = []
#             for i in f.readlines():
#                 i = [float(j) for j in i.split()]
#                 if i[1] == 0:
#                     a = np.array(i[2:5:2])
#                 else:
#                     i = np.array(i)
#                     if i[1] == 647 or i[1] == 561:
#                         i = i[[0, 2, 3, 4, 1]]
#                     elif i[3] == 647 or i[3] == 561:
#                         i = i[[0, 1, 2, 4, 3]]
#                     elif i[4] == 647 or i[4] == 561:
#                         i = i[[0, 1, 2, 3, 4]]
#                     i[1:3] += a
#                     for j in i[:-1]:
#                         p.write('%.6e' % (round(j / 0.0001) / 10000) + '\t')
#                     p.write('%.6e' % (round(i[-1] / 0.0001) / 10000) + '\n')
#         else:
#             for i in f.readlines():
#                 i = [float(j) for j in i.split()]
#                 if not i[0] < 2000:
#                     print(i[0], type(i[0]))
#                     continue
#                 i = [i[0], i[1], i[3] / 100, i[2]]
#                 for j in i[:-1]:
#                     p.write('%.6e' % (round(j / 0.0001) / 10000) + '\t')
#                 p.write('%.6e' % (round(i[-1] / 0.0001) / 10000) + '\n')


## 测试
# f = open('synapse.txt')
# p = open('test', 'w')
# a = []
# for i in f.readlines():
#     i = [float(j) for j in i.split()]
#     if i[1] == 0:
#         a = np.array(i[2:5:2])
#         print(a)
#     else:
#         i = np.array(i)
#         if i[1] == 647 or i[1] == 561:
#             # i = np.array(i[0], i[2:4] + a, i[4], i[1])
#             i = i[[0, 2, 3, 4, 1]]
#             i[1:3] += a
#             print(i)
#             for j in i[:-1]:
#                 p.write('%.4f' % j + '\t')
#             p.write('%.4f' % i[-1] + '\n')
#         if i[3] == 647 or i[3] == 561:
#             # i = np.array(i[0], i[2:4] + a, i[4], i[1])
#             i = i[[0, 1, 2, 4, 3]]
#             i[1:3] += a
#             print(i)
#             for j in i[:-1]:
#                 p.write('%.4f' % j + '\t')
#             p.write('%.4f' % i[-1] + '\n')
