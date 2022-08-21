import torch
import torch.nn as nn
import snoop

"""
Neural Networks model : GRUCell
"""


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = '1DCNN'
        self.train_path = dataset + '/data/US/US_dataframe_train.csv'                                # 训练集
        self.test_path = dataset + '/data/US/US_dataframe_test.csv'                                  # 测试集
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        # self.save_path_pre = dataset + '/saved_dict/gru.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 2580                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 30                                            # epoch数
        self.batch_size = 1                                             # mini-batch大小
        self.pad_size = 1                                              #
        self.learning_rate = 1e-4                                       # 学习率
        # self.learning_rate = 1e-4                                       # 学习率
        self.input_num = 1
        self.hidden_num = 100
        self.output_num = 1
        # self.num_layers = 3
        # self.hidden_tmp = 0
        # self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        # self.num_filters = 256                                          # 卷积核数量(channels数)


class Model(nn.Module):
    # @snoop
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(1, 64, kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(64 * 6, 50)
        # self.fc2 = nn.Linear(50, 1)
        self.fc1 = nn.Linear(64*6, 1)
        self.fc2 = nn.Linear(1, 1)

    # @snoop
    def forward(self, x):
        xt = x[0]
        x = torch.unsqueeze(xt, 2)   # 1*7*1 batch*seq_len_dim  torch.unsqueeze(input, dim, out=None)
        # x = x.permute(0,2,1)   # 1*1*7 batch*embedding*seq_len_dim
        x = x.reshape(1, 1, -1)
        # 该模型的网络结构为 一维卷积层 -> Relu层 -> Flatten（降维） -> 全连接层1 -> 全连接层2
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # y = self.fc(x)
        return x