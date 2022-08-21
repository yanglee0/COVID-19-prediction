import torch
import torch.nn as nn
import snoop

"""
Neural Networks model : GRUCell
"""


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'grucell'
        self.train_path = dataset + '/data/US_true_semi/US_dataframe_train.csv'                                # 训练集
        self.test_path = dataset + '/data/US_true_semi/US_dataframe_test.csv'                                  # 测试集
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        # self.save_path_pre = dataset + '/saved_dict/gru.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 2580                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 300                                            # epoch数
        self.batch_size = 1                                             # mini-batch大小
        self.pad_size = 1                                              #
        self.learning_rate = 1e-3                                       # 学习率
        # self.learning_rate = 5e-4                                       # 学习率
        self.input_num = 1
        self.hidden_num = 100
        self.output_num = 1
        # self.num_layers = 3
        # self.hidden_tmp = 0
        # self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        # self.num_filters = 256                                          # 卷积核数量(channels数)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 现在input按我的理解，应该是train_text+train_other_featuree。
        # 之前是每一天的总感染人数为input_num,但是现在我希望是train_other_feature和应该是train_text。
        self.grucell1 = nn.GRUCell(config.input_num, config.hidden_num)
        self.grucell2 = nn.GRUCell(config.input_num, config.hidden_num)
        self.grucell3 = nn.GRUCell(config.input_num, config.hidden_num)
        self.grucell4 = nn.GRUCell(config.input_num, config.hidden_num)
        self.grucell5 = nn.GRUCell(config.input_num, config.hidden_num)
        self.grucell6 = nn.GRUCell(config.input_num, config.hidden_num)
        self.grucell7 = nn.GRUCell(config.input_num, config.hidden_num)


        # train_other_feature分别过两个全连接层
        # self.feature_linear1 = nn.Linear(66, 66)
        # self.feature_linear2 = nn.Linear(66, 1)

        # 要分开实验
        # self.feature_linear1_1 = nn.Linear(66, 66)
        # self.feature_linear1_2 = nn.Linear(66, 1)
        #
        # self.feature_linear2_1 = nn.Linear(66, 66)
        # self.feature_linear2_2 = nn.Linear(66, 1)
        #
        # self.feature_linear3_1 = nn.Linear(66, 66)
        # self.feature_linear3_2 = nn.Linear(66, 1)
        #
        # self.feature_linear4_1 = nn.Linear(66, 66)
        # self.feature_linear4_2 = nn.Linear(66, 1)
        #
        # self.feature_linear5_1 = nn.Linear(66, 66)
        # self.feature_linear5_2 = nn.Linear(66, 1)
        #
        # self.feature_linear6_1 = nn.Linear(66, 66)
        # self.feature_linear6_2 = nn.Linear(66, 1)
        #
        # self.feature_linear7_1 = nn.Linear(66, 66)
        # self.feature_linear7_2 = nn.Linear(66, 1)

        self.out_linear = nn.Linear(config.hidden_num+2, config.output_num)
        # 这里参数维度要调，加入torchsnooper看维度的变化。
        self.embedding = nn.Linear(34, config.hidden_num)   # 对静态特征的处理
        # train_other_feature过embedding
        self.feature_linear1 = nn.Linear(462, 66)            # 对动态特征的处理
        self.feature_linear2 = nn.Linear(66, 66)  # 相关性分析用
        self.feature_linear3 = nn.Linear(66, config.hidden_num)

    # @snoop
    def forward(self, x):      # x是train_eval中的(trains, y)中的trains
        xt = x[0]
        # print("xt",xt)
        h0 = x[1]  # 1 * 34
        feature_t = x[2]  # 1 * 7 * 66
        number = x[3]
        restrict = x[4]

        number = number.view(1, 1)
        restrict = restrict.view(1, 1)

        # 这是之前的
        x = xt.view(7, 1)
        x1 = x[0].view(1, 1)
        x2 = x[1].view(1, 1)
        x3 = x[2].view(1, 1)
        x4 = x[3].view(1, 1)
        x5 = x[4].view(1, 1)
        x6 = x[5].view(1, 1)
        x7 = x[6].view(1, 1)
        # print("x",x)
        # print("x7",x7)

        # feature = feature_t.view(7, 66)
        # feature1 = feature[0].view(1, 66)
        # feature1 = self.feature_linear1(feature1)
        # feature1 = self.feature_linear2(feature1)
        #
        # feature2 = feature[1].view(1, 66)
        # feature2 = self.feature_linear1(feature2)
        # feature2 = self.feature_linear2(feature2)
        #
        # feature3 = feature[2].view(1, 66)
        # feature3 = self.feature_linear1(feature3)
        # feature3 = self.feature_linear2(feature3)
        #
        # feature4 = feature[3].view(1, 66)
        # feature4 = self.feature_linear1(feature4)
        # feature4 = self.feature_linear2(feature4)
        #
        # feature5 = feature[4].view(1, 66)
        # feature5 = self.feature_linear1(feature5)
        # feature5 = self.feature_linear2(feature5)
        #
        # feature6 = feature[5].view(1, 66)
        # feature6 = self.feature_linear1(feature6)
        # feature6 = self.feature_linear2(feature6)
        #
        # feature7 = feature[6].view(1, 66)
        # feature7 = self.feature_linear1(feature7)
        # feature7 = self.feature_linear2(feature7)

        # print("feature",feature)
        # print("feature7",feature7)

        # 现在维度 (1, 2)
        # new_x1 = torch.cat((x1, feature1), dim=1)
        # new_x2 = torch.cat((x2, feature2), dim=1)
        # new_x3 = torch.cat((x3, feature3), dim=1)
        # new_x4 = torch.cat((x4, feature4), dim=1)
        # new_x5 = torch.cat((x5, feature5), dim=1)
        # new_x6 = torch.cat((x6, feature6), dim=1)
        # new_x7 = torch.cat((x7, feature7), dim=1)
        # print("new_x7",new_x7)
        # number_layer = self.number_layer
        # print("x.shape[0]", x.shape[0])

        # feature = feature_t.view(7, 66)  # 7 * 66
        feature = feature_t.view(feature_t.size(0), -1)  # 1 * 462

        # feature1 = feature[0].view(1, 66)
        feature_dong = self.feature_linear1(feature)  # 1 * 66
        feature_dong = self.feature_linear2(feature_dong)  # 1 * 66
        feature_dong = self.feature_linear3(feature_dong)  # 1 * 100

        hid_jing = self.embedding(h0)  # 1 * 100
        hid = hid_jing + feature_dong  # 1 * 100
        # print("hid", hid.shape)
        if hid is None:
            hid = torch.randn(x.shape[0], self.hidden_size)
        # print('hid', hid)
        hid1 = self.grucell1(x1, hid)   # 需要传入隐藏层状态
        # hid1 = hid1+feature2

        hid2 = self.grucell2(x2, hid1)  # 需要传入隐藏层状态
        # hid2 = hid2+feature3

        hid3 = self.grucell3(x3, hid2)  # 需要传入隐藏层状态
        # hid3 = hid3+feature4

        hid4 = self.grucell4(x4, hid3)  # 需要传入隐藏层状态
        # hid4 = hid4+feature5

        hid5 = self.grucell5(x5, hid4)  # 需要传入隐藏层状态
        # hid5 = hid5+feature6

        hid6 = self.grucell6(x6, hid5)  # 需要传入隐藏层状态
        # hid6 = hid6+feature7

        hid7 = self.grucell7(x7, hid6)  # 需要传入隐藏层状态
        # hid7 += self.hidden_tmp
        # print("hid7",hid7.shape)
        # print("restrict",restrict.shape)
        # print("number",number.shape)
        hid_output = torch.cat((hid7, restrict, number), dim=1)    # 本模型中restrict感觉没有真正用上
        # print("hid7", hid7)
        # print("hid_output", hid_output)
        # self.hidden_tmp = hid7
        y = self.out_linear(hid_output)
        # y = y.unsqueeze(1)
        return y
        # return y, hid_output.detach()  # detach()和detach_()都可以使用





# 加入动态特征到hid中，效果不太好。
# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         # 现在input按我的理解，应该是train_text+train_other_featuree。
#         # 之前是每一天的总感染人数为input_num,但是现在我希望是train_other_feature和应该是train_text。
#         self.grucell1 = nn.GRUCell(config.input_num, config.hidden_num)
#         self.grucell2 = nn.GRUCell(config.input_num, config.hidden_num)
#         self.grucell3 = nn.GRUCell(config.input_num, config.hidden_num)
#         self.grucell4 = nn.GRUCell(config.input_num, config.hidden_num)
#         self.grucell5 = nn.GRUCell(config.input_num, config.hidden_num)
#         self.grucell6 = nn.GRUCell(config.input_num, config.hidden_num)
#         self.grucell7 = nn.GRUCell(config.input_num, config.hidden_num)
#
#
#         # train_other_feature分别过两个全连接层
#         # self.feature_linear1 = nn.Linear(66, 66)
#         # self.feature_linear2 = nn.Linear(66, 1)
#
#         # 要分开实验
#         # self.feature_linear1_1 = nn.Linear(66, 66)
#         # self.feature_linear1_2 = nn.Linear(66, 1)
#         #
#         # self.feature_linear2_1 = nn.Linear(66, 66)
#         # self.feature_linear2_2 = nn.Linear(66, 1)
#         #
#         # self.feature_linear3_1 = nn.Linear(66, 66)
#         # self.feature_linear3_2 = nn.Linear(66, 1)
#         #
#         # self.feature_linear4_1 = nn.Linear(66, 66)
#         # self.feature_linear4_2 = nn.Linear(66, 1)
#         #
#         # self.feature_linear5_1 = nn.Linear(66, 66)
#         # self.feature_linear5_2 = nn.Linear(66, 1)
#         #
#         # self.feature_linear6_1 = nn.Linear(66, 66)
#         # self.feature_linear6_2 = nn.Linear(66, 1)
#         #
#         # self.feature_linear7_1 = nn.Linear(66, 66)
#         # self.feature_linear7_2 = nn.Linear(66, 1)
#
#         self.out_linear = nn.Linear(config.hidden_num+2, config.output_num)
#         # 这里参数维度要调，加入torchsnooper看维度的变化。
#         self.embedding = nn.Linear(34, config.hidden_num)
#         # train_other_feature过embedding
#         self.feature_linear1 = nn.Linear(66, 66)
#         self.feature_linear2 = nn.Linear(66, config.hidden_num)
#
#
#     # @snoop
#     def forward(self, x):      # x是train_eval中的(trains, y)中的trains
#         xt = x[0]
#         # print("xt",xt)
#         h0 = x[1]
#         feature_t = x[2]
#         number = x[3]
#         restrict = x[4]
#
#         number = number.view(1, 1)
#         restrict = restrict.view(1, 1)
#
#         # 这是之前的
#         x = xt.view(7, 1)
#         x1 = x[0].view(1, 1)
#         x2 = x[1].view(1, 1)
#         x3 = x[2].view(1, 1)
#         x4 = x[3].view(1, 1)
#         x5 = x[4].view(1, 1)
#         x6 = x[5].view(1, 1)
#         x7 = x[6].view(1, 1)
#         # print("x",x)
#         # print("x7",x7)
#
#         feature = feature_t.view(7, 66)
#         feature1 = feature[0].view(1, 66)
#         feature1 = self.feature_linear1(feature1)
#         feature1 = self.feature_linear2(feature1)
#
#         feature2 = feature[1].view(1, 66)
#         feature2 = self.feature_linear1(feature2)
#         feature2 = self.feature_linear2(feature2)
#
#         feature3 = feature[2].view(1, 66)
#         feature3 = self.feature_linear1(feature3)
#         feature3 = self.feature_linear2(feature3)
#
#         feature4 = feature[3].view(1, 66)
#         feature4 = self.feature_linear1(feature4)
#         feature4 = self.feature_linear2(feature4)
#
#         feature5 = feature[4].view(1, 66)
#         feature5 = self.feature_linear1(feature5)
#         feature5 = self.feature_linear2(feature5)
#
#         feature6 = feature[5].view(1, 66)
#         feature6 = self.feature_linear1(feature6)
#         feature6 = self.feature_linear2(feature6)
#
#         feature7 = feature[6].view(1, 66)
#         feature7 = self.feature_linear1(feature7)
#         feature7 = self.feature_linear2(feature7)
#
#         # print("feature",feature)
#         # print("feature7",feature7)
#
#         # 现在维度 (1, 2)
#         # new_x1 = torch.cat((x1, feature1), dim=1)
#         # new_x2 = torch.cat((x2, feature2), dim=1)
#         # new_x3 = torch.cat((x3, feature3), dim=1)
#         # new_x4 = torch.cat((x4, feature4), dim=1)
#         # new_x5 = torch.cat((x5, feature5), dim=1)
#         # new_x6 = torch.cat((x6, feature6), dim=1)
#         # new_x7 = torch.cat((x7, feature7), dim=1)
#         # print("new_x7",new_x7)
#         # number_layer = self.number_layer
#         # print("x.shape[0]", x.shape[0])
#
#         hid = self.embedding(h0)
#         hid = hid+feature1
#
#         if hid is None:
#             hid = torch.randn(x.shape[0], self.hidden_size)
#         # print('hid', hid)
#         hid1 = self.grucell1(x1, hid)   # 需要传入隐藏层状态
#         hid1 = hid1+feature2
#
#         hid2 = self.grucell2(x2, hid1)  # 需要传入隐藏层状态
#         hid2 = hid2+feature3
#
#         hid3 = self.grucell3(x3, hid2)  # 需要传入隐藏层状态
#         hid3 = hid3+feature4
#
#         hid4 = self.grucell4(x4, hid3)  # 需要传入隐藏层状态
#         hid4 = hid4+feature5
#
#         hid5 = self.grucell5(x5, hid4)  # 需要传入隐藏层状态
#         hid5 = hid5+feature6
#
#         hid6 = self.grucell6(x6, hid5)  # 需要传入隐藏层状态
#         hid6 = hid6+feature7
#
#         hid7 = self.grucell7(x7, hid6)  # 需要传入隐藏层状态
#         # hid7 += self.hidden_tmp
#         # print("hid7",hid7.shape)
#         # print("restrict",restrict.shape)
#         # print("number",number.shape)
#         hid_output = torch.cat((hid7, restrict, number), dim=1)    # 本模型中restrict感觉没有真正用上
#         # print("hid7", hid7)
#         # print("hid_output", hid_output)
#         # self.hidden_tmp = hid7
#         y = self.out_linear(hid_output)
#         # y = y.unsqueeze(1)
#         return y
#         # return y, hid_output.detach()  # detach()和detach_()都可以使用
#
#
#
#
