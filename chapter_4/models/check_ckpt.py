import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import copy
import heapq
# import netron
# netron.start(R'../THUCNews/saved_dict/grucell_PGRU_second.ckpt')

state_dict = torch.load('../THUCNews/saved_dict/grucell_US_pretrain_1.2_0.045.ckpt')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    print(k)

weight2 = state_dict['feature_linear2.weight']
weight2_sum = torch.sum(weight2, dim=0)   # 对列操作
print("weight2_sum", weight2_sum)
weight2_mean = torch.div(weight2_sum, 66)
print("weight2_mean", weight2_mean)


def softmax(x):                # 小数点后8位
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


max_important = softmax(weight2_mean.tolist()).tolist()    # 权重list
min_important = softmax(weight2_mean.tolist()).tolist()    # 权重list
print("重要度(列)：", max_important)

max_number = heapq.nlargest(8, max_important)

max_index = []
for t in max_number:
    index = max_important.index(t)
    max_index.append(index)
    max_important[index] = 0

min_number = heapq.nsmallest(8, min_important)
min_index = []
for t in min_number:
    index = min_important.index(t)
    min_index.append(index)
    min_important[index] = 1

# print("重要度(列)：", important)
print("重要度(列)的数值：", max_number)
print("重要度(列)的索引：", max_index)

# print("不重要度(列)：", important)
print("不重要度(列)的数值：", min_number)
print("不重要度(列)的索引：", min_index)

# softmax = nn.Softmax(dim=0)   # 小数点后4位
# print("重要度(列)：", softmax(weight2_mean))

# print("重要度(列)：", softmax([-6.2461e-03, -6.1994e-03, -1.2966e-02,  9.5482e-03,  4.7079e-03,
#          9.7469e-03,  1.4841e-03,  5.7014e-04, -1.6479e-03,  4.0689e-04,
#         -7.0329e-03,  3.8337e-03, -6.4134e-05,  1.1760e-03, -2.3008e-02,
#         -1.7441e-03, -5.5255e-03,  5.5952e-03,  7.0308e-03, -8.7592e-03,
#          2.1221e-03,  3.2512e-03, -7.9422e-03, -2.5922e-03,  6.6981e-03,
#         -1.4708e-03,  9.3082e-04,  8.6011e-03,  8.6546e-05,  1.7486e-03,
#          1.3459e-02, -7.1171e-04,  4.0751e-03, -7.6904e-04,  5.9960e-03,
#          6.2155e-04, -3.8693e-04, -9.2068e-03,  1.3083e-02, -4.3125e-03,
#          7.4808e-03, -4.5047e-03, -8.8375e-03,  5.2267e-03, -9.8456e-03,
#          5.5090e-03,  7.1129e-03, -8.0174e-03, -6.1564e-03,  5.6701e-03,
#         -4.9212e-03,  7.1382e-03,  6.8466e-04,  4.7527e-03,  4.4900e-03,
#         -7.8621e-03,  7.4467e-04, -1.2358e-02,  8.0090e-03, -4.7294e-03,
#          7.1627e-03, -9.2520e-03,  1.0206e-02,  7.2729e-03, -1.6138e-03,
#          6.7547e-03]))



# weight1 = state_dict['feature_linear1.weight']
# weight1_sum = torch.sum(weight1, dim=1)   # 对行操作
# print("weight1_sum", weight1_sum)
# weight1_mean = torch.div(weight1_sum, 66)
# print("weight1_mean", weight1_mean)
#
# softmax = nn.Softmax(dim=0)
# print("重要度(行)：", softmax(weight1_mean))
