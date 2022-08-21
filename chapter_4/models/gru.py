import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init
# import torchsnooper

"""
Neural Networks model : GRU
"""


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'gru'
        self.train_path = dataset + '/data/US/US_dataframe_train.csv'                                # 训练集
        self.test_path = dataset + '/data/US/US_dataframe_test.csv'                                  # 测试集
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        # self.save_path_pre = dataset + '/saved_dict/gru.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 2058                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 30                                             # epoch数
        self.batch_size = 1                                             # mini-batch大小
        self.pad_size = 1                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4                                       # 学习率
        self.input_size = 1
        self.hidden_layer_size = 100
        self.output_size = 1
        self.n_layers = 1
        self.n_features = 5
        self.seq_len = 7
        # self.dropout = nn.Dropout(0.1)

        # self.seq_len = 1
        # self.hidden_tmp = 0


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # 版本1
        self.hidden_layer_size = config.hidden_layer_size
        self.gru = nn.GRU(config.input_size, config.hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(config.seq_len * config.hidden_layer_size, config.output_size)
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                     torch.zeros(1, 1, self.hidden_layer_size))
        # self.hidden_cell = self.hidden_cell.to(self.device)
        # self.hidden_cell = self.hidden_cell.to(torch.float32)
        # self.hidden = (hidden_state, cell_state)

    # @torchsnooper.snoop()
    def forward(self, x):
        xt = x[0]
        # print("xt",xt)
        h0 = x[1]
        feature_t = x[2]
        number = x[3]
        restrict = x[4]

        number = number.view(1, 1)
        restrict = restrict.view(1, 1)

        # 这是之前的
        # x = xt.view(7, 1)   # 1*7*1 batch*seq_len_dim
        x = torch.unsqueeze(xt, 2)   # 1*7*1 batch*seq_len_dim  torch.unsqueeze(input, dim, out=None)
        # print("xt", xt)
        # print("xt.shape", xt.shape)
        # print("x.shape", x.shape)

        lstm_out, self.hidden_cell = self.gru(x)   # hidden不用写  输出维度为 1*7*1
        # print("lstm_out", lstm_out.shape)
        # print("lstm_out.view(len(x), -1)", lstm_out.view(len(x), -1))
        # print("lstm_out.view(len(x), -1)", lstm_out.view(len(x), -1).shape)   # 1*700的，所以linear中要改为（seq_len*hidden,1）
        predictions = self.linear(lstm_out.view(len(x), -1))  # -1表示自适应的

        return predictions[-1]