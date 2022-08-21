# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
# import snoop
from sklearn import metrics, preprocessing
import time
from utils import get_time_dif

min_max_scaler = preprocessing.MinMaxScaler()
# hid = 0


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def rmsle(x, y):
    x = np.log1p(x)
    y = np.log1p(y)
    return np.sqrt(metrics.mean_absolute_error(x, y))


# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_pred - y_true) / y_true))

def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape


# def train(config, model, train_iter, dev_iter, test_iter):

def train(config, model, train_iter, test_iter):
    torch.autograd.set_detect_anomaly(True)  # 正向传播时：开启自动求导的异常侦测
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)    # 优化器
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)    # 自动调整学习率

    total_batch = 0  # 记录进行到多少batch

    dev_test_mape = float('inf')

    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    loss_function = nn.MSELoss()
    # loss_function = mape()

    # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
    # 其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
    # 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
    model.train()
    #fgm = FGM(model)
    # with torchsnooper.snoop():

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # print(enumerate(train_iter))
        for i, (trains, y) in enumerate(train_iter):
            text, h0, feature, number, restrict = trains
            text, h0, feature, number, restrict, y = text.to(config.device),\
            h0.to(config.device), feature.to(config.device), number.to(config.device),\
            restrict.to(config.device), y.to(config.device)
            text, h0, feature, number, restrict, y = text.to(torch.float32), \
                                                     h0.to(torch.float32), feature.to(torch.float32), number.to(torch.float32), \
                                                     restrict.to(torch.float32), y.to(torch.float32)
            trains = (text, h0, feature, number, restrict)
            # print("trains", trains)
            y = y.unsqueeze(1)  # 这句话为torch.size为[1，1]，需要改为torch.size为[1]
            model.zero_grad()
            outputs = model(trains)

            α = 0.1  # 调这个！！！
            β = 1-α

            loss1 = loss_function(outputs[0], y[0])  # 监督损失
            loss2 = loss_function(outputs[1], α*y[0] + β*y[1])  # mixup损失
            loss = α * loss1 + β * loss2
            loss.backward()  # 反向传播。前面有正向传播，在此进行反向传播。

            # 对抗训练
            # fgm.attack()  # 在embedding上添加对抗扰动
            #outputs = model(trains)
            #loss_adv = loss_function(outputs, y)
            #loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            #fgm.restore()  # 恢复embedding参数

            optimizer.step()
            # lr_schedule.step(loss)   # 自动调整学习率

            if total_batch % 258 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # true = y.data.cpu()
                # print(true.size())
                # predic = outputs.data.cpu()
                # print(outputs.data.size())
                # train_rmse = np.sqrt(metrics.mean_squared_error(true, predic))
                # train_mse = metrics.mean_squared_error(true, predic)
                # train_mape = mape(true.data.cpu().numpy(), predic.data.cpu().numpy())
                # dev_mse, dev_loss, dev_mape = evaluate(config, model, dev_iter)
                test_loss, y_test_all, predict_test_all, y_test_all_inverse, \
                predict_test_all_inverse, test_mape, test_rmse, test_rmse_inverse = \
                    evaluate(config, model, test_iter, test=True)
                # msg = 'Test Loss: {0:>5.2}, Test mape:{1:>6.2}, Test rmse:{2:>6.2}, Test rmse_inverse:{3:>6.2}'
                # print(msg.format(test_loss, test_mape, test_rmse, test_rmse_inverse))
                #
                # print('true_test', y_test_all)
                # print("predict_test", predict_test_all)
                # print("y_test_all_inverse", y_test_all_inverse)
                # print("predict_test_all_inverse", predict_test_all_inverse)

                # # 验证集合的
                # dev_mse, dev_mape = evaluate(config, model)
                if test_mape < dev_test_mape:
                    dev_test_mape = test_mape
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>1},   Test MAPE: {1:.5f},   improve {2},   Train_loss:{3:.5f}'
                # msg = 'Test Loss: {0:>5.2}, Test mape:{1:>6.2}, Test rmse:{2:>6.2}, Test rmse_inverse:{3:>6.2}'
                print('total_batch',total_batch)
                print('last_improve',total_batch)
                print('config.require_improvement',config.require_improvement)
                print('flag',flag)
                print(msg.format(total_batch, test_mape, improve, loss))
                lr_schedule.step(dev_test_mape)
                # print(msg.format(test_loss, test_mape, test_rmse, test_rmse_inverse))

                model.train()
                # torch.save(model.state_dict(), config.save_path)
            total_batch = 1 + total_batch
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过50batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                print("结束训练")
                print('total_batch', total_batch)
                print('last_improve', total_batch)
                print('config.require_improvement', config.require_improvement)
                print('flag', flag)
                break
        if flag:
            break
    # torch.save(model.state_dict(), config.save_path)  # 预训练打开，微调时候要去掉
    test(config, model, test_iter)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([])
    y_all = np.array([])
    loss_function = nn.MSELoss()
    # loss_function = mape()

    predict_all_inverse = np.array([], dtype=float)
    y_all_inverse = np.array([], dtype=float)


    with torch.no_grad():
        for _x, y in data_iter:
            text, h0, feature, number, restrict = _x
            text, h0, feature, number, restrict, y = text.to(config.device), \
                                                     h0.to(config.device), feature.to(config.device), number.to(
                config.device), \
                                                     restrict.to(config.device), y.to(config.device)

            text, h0, feature, number, restrict, y = text.to(torch.float32), \
                                                     h0.to(torch.float32), feature.to(torch.float32), number.to(
                torch.float32), \
                                                     restrict.to(torch.float32), y.to(torch.float32)
            trains = (text, h0, feature, number, restrict)
            # print("trains", trains)
            y = y.unsqueeze(1)  # 这句话为torch.size为[1，1]，需要改为torch.size为[1]
            model.zero_grad()
            outputs = model(trains)

            loss = loss_function(outputs, y)  # 监督损失

            loss_total += loss

            y = y.data.cpu().numpy()

            # y_inverse = int(y*414188+1)  # 414188 印度
            # y_inverse = int(y*10901+1)     # NewYork
            y_inverse = int(y*287198+1)  # US
            # y_inverse = int(y*71734+1)

            predic = outputs.data.cpu().numpy()
            # predic = outputs[0].data.cpu().numpy()

            # predict_inverse = int(predic*414188+1)
            # predict_inverse = int(predic*10901+1)  # France
            predict_inverse = int(predic*287198+1)  # US
            # predict_inverse = int(predic*71734+1)

            y_all = np.append(y_all, y)
            predict_all = np.append(predict_all, predic)
            y_all_inverse = np.append(y_all_inverse, y_inverse)
            predict_all_inverse = np.append(predict_all_inverse, predict_inverse)

    # MSE/RMSE
    # MSE = metrics.mean_squared_error(y_all, predict_all)
    # if test:
    #     test_rmse = np.sqrt(metrics.mean_squared_error(y_all, predict_all))
    #     return MSE, loss_total / len(data_iter), y_all, predict_all, y_all_inverse, predict_all_inverse, test_rmse
    # return MSE, loss_total / len(data_iter)
    # print("test_y_all", y_all)
    # print("test_predict_all", predict_all)
    REMLE = rmsle(y_all, predict_all)
    MAPE = mape(y_all, predict_all)
    RMSE = np.sqrt(metrics.mean_squared_error(y_all, predict_all))
    RMSE_inverse = np.sqrt(metrics.mean_squared_error(y_all_inverse, predict_all_inverse))
    # MSE = metrics.mean_squared_error(y_all, predict_all)
    if test:
        test_rmse = np.sqrt(metrics.mean_squared_error(y_all, predict_all))
        test_rmse_inverse = np.sqrt(metrics.mean_squared_error(y_all_inverse, predict_all_inverse))

        # test_rmsle = rmsle(y_all, predict_all)
        test_mape = mape(y_all, predict_all)
        return loss_total / len(data_iter), y_all, predict_all, y_all_inverse, predict_all_inverse, test_mape, test_rmse, test_rmse_inverse
    return RMSE, loss_total / len(data_iter), MAPE


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))   # 调用最好的一次下模型效果
    model.eval()
    start_time = time.time()
    test_loss, y_test_all, predict_test_all, y_test_all_inverse, \
    predict_test_all_inverse, test_mape,  test_rmse, test_rmse_inverse = \
        evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:.5f}, Test mape:{1:.5f}, Test rmse:{2:.5f}, Test rmse_inverse:{3:.5f}'
    print(msg.format(test_loss, test_mape, test_rmse, test_rmse_inverse))

    print('true_test', y_test_all)
    print("predict_test", predict_test_all)
    print("y_test_all_inverse", y_test_all_inverse)
    print("predict_test_all_inverse", predict_test_all_inverse)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


# 用模型预测值来画图的。
def test_all(config, model, train_iter, test_iter):
    model.eval()
    start_time = time.time()
    train_loss, y_train_all, predict_train_all, y_train_all_inverse, predict_train_all_inverse, train_mape,  train_rmse, train_rmse_inverse = evaluate(config, model, train_iter, test=True)
    test_loss, y_test_all, predict_test_all, y_test_all_inverse, predict_test_all_inverse, test_mape, test_rmse, test_rmse_inverse = evaluate(
        config, model, test_iter, test=True)
    msg = 'Test Loss: {0:.5f}, Test mape:{1:.5f}, Test rmse:{2:.5f}, Test rmse_inverse:{3:.5f}'
    print(msg.format(test_loss, test_mape, test_rmse, test_rmse_inverse))

    # print('true_train', y_train_all)
    # print("predict_trian", predict_train_all)
    print("y_train_all_inverse", y_train_all_inverse)
    print("predict_train_all_inverse", predict_train_all_inverse)

    # print('true_test', y_test_all)
    # print("predict_test", predict_test_all)
    print("y_test_all_inverse", y_test_all_inverse)
    print("predict_test_all_inverse", predict_test_all_inverse)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
