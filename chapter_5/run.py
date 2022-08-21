# coding: UTF-8
import time
import torch
import numpy as np
import pandas as pd
from train_eval_531 import train,test
# from train_eval_532 import train,test
import torch.utils.data as Data       # 引入dataloader
from importlib import import_module
from utils import get_data, get_time_dif, loader_dataset
import models


# argparse使用的4个步骤    为了输入命令行的使用的
import argparse      # 导入该模块
parser = argparse.ArgumentParser(description='COVID-19')     # 然后创建一个解析对象
parser.add_argument('--model', type=str, required=True, help='choose a model: GRU')  # 然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
args = parser.parse_args()  # 最后调用parse_args()方法进行解析；解析成功之后即可使用。

# torch.set_printoptions(precision=6)

if __name__ == '__main__':
    dataset = 'THUCNews'      # 数据集
    model_name = args.model   # grucell
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)       # 每次得到的随机数是固定的
    torch.manual_seed(1)    # 每次得到的随机数是固定的
    torch.cuda.manual_seed_all(1)    # 每次得到的随机数是固定的，这三句话都是为了每次的结果是一致的
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    train_data, test_data = get_data(config)
    train_iter = Data.DataLoader(
        dataset=train_data, batch_size=config.batch_size, shuffle=True)
    test_iter = Data.DataLoader(
        dataset=test_data, batch_size=1, shuffle=False)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # 做base实验的时候改的，去掉了.to(config.device)
    # model = x.Model(config)
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load('THUCNews/saved_dict/grucell_531.ckpt'))   # 加载预训练模型
    # model.load_state_dict(torch.load('THUCNews/saved_dict/grucell_US_pretrain_1.3.ckpt'))
    # model.load_state_dict(torch.load('THUCNews/saved_dict/grucell_Texas_6.9.ckpt'))
    # train(config, model, train_iter, test_iter)
    test(config, model, test_iter)
    # test_all(config, model, train_iter, test_iter)
