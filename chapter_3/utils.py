# coding: UTF-8
import torch
from sklearn import preprocessing
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset
import pandas as pd
import ast

pd.set_option('mode.use_inf_as_na', True)


def get_data(config):
    """Read data, and build dataset for dataloaders.
    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv, dev.csv and test.csv
    """
    # --------可根据自己数据情况进行调整--------

    # 一个列里面是用，隔开，列之间用\t隔开。
    train_df = pd.read_csv(config.train_path, sep='\t', header=None)
    test_df = pd.read_csv(config.test_path, sep='\t', header=None)

    # print("train_df",train_df.isnull().any())
    #
    # # print(np.isfinite(train).all())
    # print("train_df",np.isinf(train_df).all())
    # print("test_df",test_df.isnull().any())
    # print("test_df",train_df.isfinite(test_df))

    # 对训练集中各种数据进行切分
    train_labels = [v for v in train_df[1]]  # dataframe中的第几列 相当于y
    train_h0 = []    # h0
    for v in range(len(train_df)):   # len() = 642天
        train_h0.append([train_df.iloc[v, i] for i in range(10, 44)])   # 对每一天去遍历所有的静态向量     # h0放在一起
        # train_h0.append([train_df.iloc[v, i] for i in range(9, 43)])   # 对每一天去遍历所有的静态向量     # h0放在一起

    train_other_feature = []    # 弄成[[],[],[],[]]

    for v in range(len(train_df)):                  # 动态  其他特征放在一起
        # train_other_feature.append([ast.literal_eval(train_df.iloc[v, i]) for i in range(0, 9) and range(43,len(train_df.columns))])
        train_other_feature.append([ast.literal_eval(train_df.iloc[v, i]) for i in range(3, 10) and range(44, len(train_df.columns))])

    # train_other_feature = train_other_feature.view(642, 7, 66)
    train_text = [ast.literal_eval(v) for v in train_df[2]]  # 时间序列的7天的x
    # train_text = ast.literal_eval(train_text)   # 时间序列的7天的x

    train_restrict = [ast.literal_eval(v)[-1] for v in train_df[88]]
    train_number = [v for v in train_df[0]]
    print("train_labels", np.array(train_labels).shape)
    # print("train_labels", train_labels)
    print("---------------------")
    print('train_h0', np.array(train_h0).shape)
    # print('train_h0', train_h0)
    print("---------------------")
    print('train_other_featrue', np.array(train_other_feature).shape)
    # print('train_other_featrue', train_other_feature)
    print("---------------------")
    print('train_text', np.array(train_text).shape)
    # print('train_text', train_text)
    print("---------------------")
    print('train_restrict', np.array(train_restrict).shape)
    # print('train_restrict', train_restrict)
    print("---------------------")
    print('train_number', np.array(train_number).shape)
    # print('train_number', train_number)

    # print("len(test_df.columns)",len(test_df.columns))   # 109
    # 对测试集中各种数据进行切分
    test_labels = [v for v in test_df[1]]
    # test_labels = [ast.literal_eval(v)[-1] for v in test_df[1]]
    test_h0 = []    # h0
    for v in range(len(test_df)):
        test_h0.append([test_df.iloc[v, i] for i in range(10, 44)])  # h0静态放在一起
    # for i in range(9, 43):
    #     test_h0.append([v for v in test_df[i]])    # h0放在一起
    test_other_feature = []
    for v in range(len(test_df)):      # 动态  其他特征放在一起
        # test_other_feature.append([ast.literal_eval(test_df.iloc[v, i]) for i in range(0, 9) and range(43, len(test_df.columns))])
        test_other_feature.append([ast.literal_eval(test_df.iloc[v, i]) for i in range(3, 10) and range(44, len(test_df.columns))])
    # for i in range(0, 9) and range(43, len(test_df.columns)):          # 动态
    #     test_other_feature.append([v for v in test_df[i]])  # 其他特征放在一起。
    test_text = [ast.literal_eval(v) for v in test_df[2]]
    test_restrict = [ast.literal_eval(v)[-1] for v in test_df[88]]
    test_number = [v for v in test_df[0]]

    print("test_labels", np.array(test_labels).shape)
    # print("train_labels", train_labels)
    print("---------------------")
    print('test_h0', np.array(test_h0).shape)
    # print('train_h0', train_h0)
    print("---------------------")
    print('test_other_feature', np.array(test_other_feature).shape)
    # print('test_other_feature', test_other_feature)
    print("---------------------")
    print('test_text', np.array(test_text).shape)
    # print('train_text', train_text)
    print("---------------------")
    print('test_restrict', np.array(test_restrict).shape)
    # print('train_restrict', train_restrict)
    print("---------------------")
    print('test_number', np.array(test_number).shape)

    train_dataset = loader_dataset(
        train_text, train_h0, train_other_feature, train_restrict, train_number, train_labels, config)
    test_dataset = loader_dataset(
        test_text, test_h0, test_other_feature, test_restrict, test_number, test_labels, config)

    return train_dataset, test_dataset


class loader_dataset(Dataset):   # 提供data loader提供接口，dataloader会调这个，更快。
    # Data loader
    def __init__(self, dataset_text, dataset_h0, dataset_other_feature, dataset_restrict, dataset_number, dataset_label, config):
        self.text = dataset_text
        self.h0 = dataset_h0
        self.feature = dataset_other_feature
        self.restrict = dataset_restrict
        self.number = dataset_number
        self.labels = dataset_label
        self.config = config

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]   # 7天的序列数据
        h0 = self.h0[idx]
        feature = self.feature[idx]  # 7天的特征数据
        feature = np.array(feature).T
        restrict = self.restrict[idx]
        number = self.number[idx]
        labels = self.labels[idx]    # 7天的y
        # print("text",torch.tensor(text))
        # print("h0",torch.tensor(h0))
        # print("feature",torch.tensor(feature))
        # print("number",torch.tensor(number))
        # print("restrict",torch.tensor(restrict))
        # print('labels',torch.tensor(labels))
        return (torch.tensor(text), torch.tensor(h0), torch.tensor(feature), torch.tensor(number), torch.tensor(restrict)), torch.tensor(labels)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
