# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: tarin.py
import argparse
import os
import time
import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from datasets import RPIseqDataset
from autocode import AutoEncodeCnn
from cnn import DeepCnn

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

seed = 333

if torch.cuda.is_available():
    device = "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
else:
    device = "cpu"

torch.set_num_threads(16)
num_workers = 0

learning_rate = 0.0001

curr_path = os.getcwd()

if 'autodl-tmp' in curr_path:
    logs_path = '../../tf-logs'
else:
    logs_path = './logs'


class RPIModel:
    def __init__(self,file_path):
        self.file_path = file_path
        self.path = os.getcwd()
        self.auto_path = "./models/AE"
        self.cnn_path = './models/CNN'

    def train_auto(self,model,dataset,batch_size,n_epoch,sample,step=10):
        """
        训练自编码器
        :param model:
        :param dataset:
        :param batch_size:
        :param n_epoch:
        :param sample:
        :param step:
        :return:
        """
        if os.path.exists(self.auto_path) and len(os.listdir(self.auto_path))>0:
            flag = True
        else:
            flag = False

        loss_fn = torch.nn.MSELoss()
        # 优化器
        optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                                    eps=1e-8, verbose=True)

        if flag:
            checkoint = torch.load(f'{self.auto_path}/model.pth')
            model.load_state_dict(checkoint['model'])
            optimizer.load_state_dict(checkoint['optimizer'])
            start_epoch = checkoint['epoch']
        else:
            start_epoch = 0

        total_loss = 0.0
        previous_loss = 1.0
        max_count = 0
        log_name = '{}_{}'.format('AutoEncode', time.strftime('%m-%d_%H.%M', time.localtime()))
        log_dir = os.path.join(logs_path, log_name)
        writer = SummaryWriter(log_dir=log_dir)

        train_size = int(sample * 0.8)
        test_size = sample - train_size
        train_dataset, test_dateset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers)
        test_data = torch.utils.data.DataLoader(test_dateset, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers)

        for epoch in range(start_epoch,n_epoch):
            model.train()

            print("{}AutoEncode Epoch: [{}/{}] {}".format('=' * 40, epoch + 1, n_epoch, '=' * 40))
            running_loss = 0.0
            start = time.time()
            for i,(x,y) in enumerate(train_data):
                x = x.to(device)
                encode, decode = model(x)
                loss = loss_fn(decode, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (i + 1) % step == 0 or (i + 1) == len(train_data):
                    print("Sample:[{}] Step: [{}/{}], train_loss: {:.10f} | lr: {:.10f}".format(
                        sample, str(i + 1).zfill(3), len(train_data), running_loss / (i + 1),
                        self.get_cur_lr(optimizer)))
                    s = (epoch * len(train_data))+(i+1)
                    writer.add_scalar('Auto_Loss/Step', running_loss / (i + 1), s)
            epoch_loss = running_loss / len(train_data)
            total_loss += epoch_loss
            # 进行验证
            model.eval()
            sum_loss = 0.0
            with torch.no_grad():
                for i,(x,y) in enumerate(test_data):
                    x = x.to(device)
                    encode, decode = model(x)
                    _loss = loss_fn(decode, x)
                    sum_loss += _loss.item()
            test_loss = sum_loss / len(test_data)
            if scheduler:
                # 根据测试损失调整
                scheduler.step(test_loss)
                writer.add_scalar('Auto_Epoch/Lr', self.get_cur_lr(optimizer),epoch)

            if test_loss < previous_loss and epoch > 5:
                # 如果当前损失小于上一个就保存
                previous_loss = test_loss
                max_count = 0
                state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}
                torch.save(state_dict, f'{self.auto_path}/model.pth')
            else:
                max_count += 1

            end = time.time()
            print('{} 耗时:{:.4f}s {}'.format('-' * 45, end - start, '-' * 45))
            print("[Epochs:%d] AutoEncode-Train-Avg-Total-Loss: %.10f, AutoEncode-Test-Current-Loss: %.10f, AutoEncode-Previous-Current-Loss: %.10f, max_count: %d" % (epoch, total_loss / (epoch+1), test_loss,previous_loss,max_count))

            if max_count > 100:
                break
        print(f'*********************结束AutoEncode训练{max_count}**********************')
        writer.close()
        return True

    def train_k_fold(self,model,auto_model,train_dataset,test_dataset,batch_size,n_epoch,sample,num,step=40):
        """
        训练十折交叉验证结果
        :param model:
        :param auto_model:
        :param train_dataset:
        :param test_dataset:
        :param batch_size:
        :param n_epoch:
        :param sample:
        :param num:
        :param step:
        :return:
        """
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # 优化器
        optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters(),weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                                    eps=1e-9, verbose=True)

        total_loss, total, correct = .0, .0, .0

        log_name = '{}_{}'.format('DeepCnn', time.strftime('%m-%d_%H.%M', time.localtime()))
        log_dir = os.path.join(logs_path, log_name)
        writer = SummaryWriter(log_dir=log_dir)

        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers)

        for epoch in range(n_epoch):
            model.train()
            print("{}Train_k_fold Epoch: [{}/{}/{}] {}".format('=' * 36, num ,epoch + 1, n_epoch, '=' * 36))
            running_loss = 0.0

            start = time.time()  # 计算耗时
            for i,(x,y) in enumerate(train_data):
                x, y = x.to(device), y.to(device)
                encoder = self.get_encoder(auto_model,x)
                out = model(encoder)
                loss = self.criterion(loss_fn,out,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total += y.size(0)

                # 累计预测正确的样本数
                correct += (out.argmax(dim=1) == y).type(torch.float32).sum().item()
                # 准确率
                train_acc = (correct / total) * 100.0
                if (i + 1) % step == 0 or (i + 1) == len(train_data):
                    print("Sample:[{}] Step: [{}/{}], train_loss: {:.10f} | train_acc: {:.8f}% | lr: {:.10f}".format(
                        sample[0], str(i + 1).zfill(3), len(train_data), running_loss / (i + 1), train_acc,
                        self.get_cur_lr(optimizer)))
                    s = (i + 1) + (epoch * len(train_data))
                    writer.add_scalar('Train:Loss/Step', running_loss / (i + 1), s)
                    writer.add_scalar('Train:Acc/Step', train_acc, s)

            epoch_loss = running_loss / len(train_data)
            total_loss += epoch_loss
            end = time.time()
            print('{} 耗时:{:.4f}s {}'.format('-' * 45, end - start, '-' * 45))

            if scheduler:
                scheduler.step(epoch_loss)
                writer.add_scalar('Epoch/Lr', self.get_cur_lr(optimizer),epoch)

        # 进行测试集验证
        print("{} Test data {}".format('*' * 45, '*' * 45))
        test_loss, test_acc, auc, auprc, accuracy, precision, recall, f1, y_ture, y_pred = self.evaluate(model, auto_model, test_dataset, loss_fn)
        _accuracy,_recall,_precision,_f1,_auc,_auprc,TP,FP,TN,FN = self.metrics(y_ture, y_pred)
        print("{} Test end {}".format('*' * 45, '*' * 45))

        max_params = {}
        max_params['loss'] = deepcopy(test_loss)
        max_params['acc'] = deepcopy(test_acc)
        max_params['auc'] = deepcopy(auc)
        max_params['auprc'] = deepcopy(auprc)
        max_params['accuracy'] = deepcopy(accuracy)
        max_params['precision'] = deepcopy(precision)
        max_params['recall'] = deepcopy(recall)
        max_params['f1'] = deepcopy(f1)
        max_params['_accuracy'] = _accuracy
        max_params['_recall'] = _recall
        max_params['_precision'] = _precision
        max_params['_f1'] = _f1
        max_params['_auc'] = _auc
        max_params['_auprc'] = _auprc
        max_params['TP'] = TP
        max_params['FP'] = FP
        max_params['TN'] = TN
        max_params['FN'] = FN

        writer.close()
        print(f"*****************************DeepCnn训练结束:{num}*****************************************")
        return max_params

    def train(self, model, auto_model, train_dataset, valid_dataset, batch_size, n_epoch, sample,loss_fn, step=40):
        """
        训练模型
        :param model:
        :param auto_model:
        :param train_dataset:
        :param test_dataset:
        :param batch_size:
        :param n_epoch:
        :param sample:
        :param step:
        :return:
        """
        if os.path.exists(self.cnn_path) and len(os.listdir(self.cnn_path)) > 0:
            flag = True
        else:
            flag = False

        # 优化器
        optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                               eps=1e-11, verbose=True)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40,50,60,65,70,75,80,85,90,95],gamma=0.1,verbose=True)
        if flag:
            checkoint = torch.load(f'{self.cnn_path}/model.pth',map_location=device)
            model.load_state_dict(checkoint['model'])
            optimizer.load_state_dict(checkoint['optimizer'])
            scheduler.load_state_dict(checkoint['lr_schedule'])
            start_num = checkoint['epoch']
            previous_acc= checkoint['previous_acc']
        else:
            start_num = 0
            previous_acc = 0.0

        model.to(device)
        total_loss, total, correct = 0, 0, 0

        previous_loss = 1.0
        max_count = 0
        log_name = '{}_{}'.format('TrainDeepCnn', time.strftime('%m-%d_%H.%M', time.localtime()))
        log_dir = os.path.join(logs_path, log_name)
        writer = SummaryWriter(log_dir=log_dir)


        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers)

        for epoch in range(start_num,n_epoch):
            model.train()
            print("{}DeepCnn Epoch: [{}/{}] {}".format('=' * 39, epoch + 1, n_epoch, '=' * 39))
            running_loss = 0.0

            start = time.time()  # 计算耗时
            for i, (x, y) in enumerate(train_data):
                x, y = x.to(device), y.to(device)
                encoder = self.get_encoder(auto_model, x)
                out = model(encoder)
                loss = self.criterion(loss_fn, out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total += y.size(0)

                # 累计预测正确的样本数
                correct += (out.argmax(dim=1) == y).type(torch.float32).sum().item()
                # 准确率
                train_acc = (correct / total) * 100.0
                if (i + 1) % step == 0 or (i + 1) == len(train_data):
                    print("Sample:[{}] Step: [{}/{}], train_loss: {:.10f} | train_acc: {:.8f}% | lr: {:.11f}".format(
                        sample[0], str(i + 1).zfill(3), len(train_data), loss.item(), train_acc,
                        self.get_cur_lr(optimizer)))
                    s = (i + 1) + (epoch * len(train_data))
                    writer.add_scalar('Train:Loss/Step', running_loss / (i + 1), s)
                    writer.add_scalar('Train:Acc/Step', train_acc, s)

            epoch_loss = running_loss / len(train_data)
            total_loss += epoch_loss
            end = time.time()
            print('{} 耗时:{:.4f}s {}'.format('-' * 43, end - start, '-' * 43))



            print("{} Test data {}".format('*' * 45, '*' * 45))
            test_loss, test_acc, auc, auprc, accuracy, precision, recall, f1, y_ture, y_pred = self.evaluate(model,
                                                                                                          auto_model,
                                                                                                          valid_dataset,
                                                                                                          loss_fn)
            print("{} Test end {}".format('*' * 45, '*' * 45))

            if scheduler:
                scheduler.step(test_loss)
                writer.add_scalar('Epoch/Lr', self.get_cur_lr(optimizer), epoch)

            if test_acc > previous_acc and epoch > 5:
                # 如果当前准确率大于上一个保存
                previous_acc = test_acc
                previous_loss = test_loss
                max_count = 0
                path = f'{self.cnn_path}/model.pth'
                # if os.path.isfile(path):
                #     os.remove(path)
                state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),'lr_schedule': scheduler.state_dict(),"previous_acc":previous_acc, "epoch": epoch + 1}
                torch.save(state_dict, path)
            else:
                max_count += 1

            print("[Epochs:{}|max_count:{}] Sample: [{}] Test_loss: {:.10f} | Test_acc: {:.8f} | previous_test_acc: {:.8f} | Train_loss: {:.10f} | Previous_test_loss: {:.10f}".format(epoch,max_count,sample[-1], test_loss, test_acc,previous_acc,epoch_loss,previous_loss))

            # if max_count > 80:
            #     break

        writer.close()
        print(f"*****************************DeepCnn训练结束:{max_count}*****************************************")

    def criterion(self,loss_fn,out,y):
        # 根据损失函数，改变数据形状和类型:BCELoss+Sigmoid=BCEWithLogitsLoss
        if isinstance(loss_fn,torch.nn.BCEWithLogitsLoss):
            y_hot = torch.nn.functional.one_hot(y, 2)
            y_hot = torch.as_tensor(y_hot, dtype=torch.float32)
        else:
            y_hot = y
        loss = loss_fn(out,y_hot).to(device)
        return loss

    def evaluate(self,model,auto_model, dataset,loss_fn):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True,
                                                num_workers=num_workers)
        total, correct = 0, 0
        running_loss = 0.
        test_preds, test_trues = [], []
        model.eval()  # 测试模式
        with torch.no_grad():  # 不计算梯度
            for i, (x, y) in enumerate(data_loader):
                x, y = x.to(device), y.to(device)  # CPU or GPU运行
                encoder = self.get_encoder(auto_model,x)
                out = model(encoder.to(device))  # 计算输出
                loss = self.criterion(loss_fn,out, y)  # 计算损失
                running_loss += loss.item()
                total += y.size(0)  # 计算测试集总样本数
                correct += (out.argmax(dim=1) == y).type(torch.float).sum().item()  # 计算测试集预测准确的样本数

                test_outputs = out.argmax(dim=1)
                test_preds.extend(test_outputs.detach().cpu().numpy())
                test_trues.extend(y.detach().cpu().numpy())

        test_acc = (correct / total) * 100.0  # 测试集准确率

        test_loss = running_loss / len(data_loader)
        # 就算指标
        auc, auprc, accuracy, precision, recall, f1 = self.report(test_trues, test_preds, test_acc)

        # 训练模式 （因为这里是因为每经过一个Epoch就使用测试集一次，使用测试集后，进入下一个Epoch前将模型重新置于训练模式）
        return test_loss, test_acc, auc, auprc,accuracy, precision, recall, f1,test_trues, test_preds

    def predict(self,model,auto_model,data,batch_size=64):
        """
        分类预测
        :param model:
        :param auto_model:
        :param data:
        :param batch_size:
        :return:
        """
        train_data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        preds = []
        model.eval()
        with torch.no_grad():  # 不计算梯度
            for i, x in enumerate(train_data):
                encoder = self.get_encoder(auto_model, x)
                out = model(encoder.to(device))  # 计算输出
                test_outputs = out.argmax(dim=1)
                pred = test_outputs.detach().cpu().numpy()
                preds.extend(pred)
        return preds

    def get_encoder(self,net, x):
        net.eval()
        with torch.no_grad():
            inputs = x.to(device)
            encoder, decoder = net(inputs)
            return encoder

    def get_cur_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def report(self,test_trues, test_preds, accuracy):
        # 分类报告 计算各项指标
        precision = precision_score(test_trues, test_preds, average='macro')
        recall = recall_score(test_trues, test_preds, average='macro')
        f1 = f1_score(test_trues, test_preds, average='macro')
        auc = roc_auc_score(test_trues, test_preds)
        auprc = average_precision_score(test_trues, test_preds)
        print(classification_report(test_trues, test_preds))

        print("[sklearn_metrics] auc:{:.4f} auprc:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
            auc,auprc,accuracy,precision,recall,f1))
        return auc,auprc,accuracy,precision,recall,f1

    def metrics(self,y_true,y_pred):
        """
        根据公式计算指标
        :param y_true:
        :param y_pred:
        :return:
        """
        y_true, y_pred = np.array(y_true),np.array(y_pred)
        part = y_pred ^ y_true
        pcount = np.bincount(part)
        tp_list = list(y_pred & y_true)
        fp_list = list(y_pred & ~y_true)
        TP = tp_list.count(1)   # 真阳性
        FP = fp_list.count(1)   # 真阴性
        TN = pcount[0] - TP     # 假阳性
        FN = pcount[1] - FP     # 假阴性

        accuracy = (TP+TN) / (TP+TN+FP+FN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        f1 = (2*precision*recall) / (precision+recall)
        auc = self.auc_calculate(y_true,y_pred)
        auprc = average_precision_score(y_true, y_pred,average='weighted')
        return accuracy,recall,precision,f1,auc,auprc,TP,FP,TN,FN

    def auc_calculate(self,y_true, y_pred):
        f = list(zip(y_pred, y_true))
        rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
        posNum = 0
        negNum = 0
        for i in range(len(y_true)):
            if (y_true[i] == 1):
                posNum += 1
            else:
                negNum += 1
        auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
        return auc

    def k_fold(self,load_data, auto_model, dic, epoch, k):
        """
         十折交叉验证
        :param load_data:
        :param auto_model:
        :param dic:
        :param epoch:
        :param k:
        :return:
        """
        train_loss_sum, valid_loss_sum = 0, 0
        train_acc_sum, valid_acc_sum = 0, 0
        accuracy_sum, precision_sum, recall_sum, F1_sum, auc_sum, auprc_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=22)
        X, y = load_data.transform, load_data.target_transform
        dataset = skf.split(X, y)
        log_name = '{}_{}'.format('k_fold', time.strftime('%m-%d_%H.%M', time.localtime()))
        log_dir = os.path.join(logs_path, log_name)
        writer = SummaryWriter(log_dir=log_dir)

        for i, (train_index, test_index) in enumerate(dataset):
            k = i + 1
            # 获取k折交叉验证的训练和验证数据
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            sample = (len(y_train), len(y_test))
            # 开始训练deepcnn,每份数据进行训练
            channels = 384
            layer = [3, 4, 6, 3]  # 50
            cnn_model = DeepCnn(layer, channels=channels, se_b=dic['se']).to(device)
            print(cnn_model)
            max_params = self.train_k_fold(cnn_model, auto_model, train_dataset, test_dataset, dic['batch_size'], epoch,sample, k)
            print('*' * 25, '第', k, '折', '*' * 25)
            print(
                "sklearn:=[k_flod:{}] valid_loss: {:.10f} | valid_acc: {:.8f}| auc: {:.8f} | auprc: {:.8f} | accuracy: {:.10f} | precision: {:.10f} | recall: {:.10f} | f1: {:.10f}".format(
                    k, max_params['loss'], max_params['acc'], max_params['auc'], max_params['auprc'],
                    max_params['accuracy'], max_params['precision'], max_params['recall'], max_params['f1']))
            print(
                "metrics:=[k_flod:{}] valid_loss: {:.10f} | valid_acc: {:.8f}| auc: {:.8f} | auprc: {:.8f} | accuracy: {:.10f} | precision: {:.10f} | recall: {:.10f} | f1: {:.10f}".format(
                    k, max_params['loss'], max_params['acc'], max_params['_auc'], max_params['_auprc'],
                    max_params['_accuracy'], max_params['_precision'], max_params['_recall'], max_params['_f1']))


            valid_loss_sum += max_params['loss']
            valid_acc_sum += max_params['acc']
            accuracy_sum += max_params['accuracy']
            precision_sum += max_params['precision']
            recall_sum += max_params['recall']
            F1_sum += max_params['f1']
            auc_sum += max_params['auc']
            auprc_sum += max_params['auprc']
            writer.add_hparams({"k": k}, max_params)
            print(
                "当前平均：[k_flod:{}] valid_loss: {:.10f} | valid_acc: {:.8f}| auc: {:.8f} | auprc: {:.8f} | accuracy: {:.10f} | precision: {:.10f} | recall: {:.10f} | f1: {:.10f}".format(
                    k, valid_loss_sum / k, valid_acc_sum / k, auc_sum / k, auprc_sum / k,
                    accuracy_sum / k, precision_sum / k, recall_sum / k,
                    F1_sum / k))
        writer.close()
        print('#' * 10, '最终k折交叉验证结果', '#' * 10)
        print(
            "[k_flod:{}] valid_loss: {:.10f} | valid_acc: {:.8f}| auc: {:.8f} | auprc: {:.8f} | accuracy: {:.10f} | precision: {:.10f} | recall: {:.10f} | recall: {:.10f}".format(
                k, valid_loss_sum / k, valid_acc_sum / k, auc_sum / k, auprc_sum / k, accuracy_sum / k,
                precision_sum / k, recall_sum / k, F1_sum / k))

    def save_file(self,data,filename):
        """
        保存预测结果
        :param data:
        :param filename:
        :param columns:
        :return:
        """
        output = os.path.join(self.path,'output')
        if not os.path.exists(output):
            os.mkdir(output)
        save_file_path = os.path.join(output,filename)
        df = pd.DataFrame(data)
        df.columns = ['pred']
        df.to_csv(save_file_path)

    def run(self,n_epoch,cnn_epoch,predict,k_fold,k):
        """
        程序入口
        :param n_epoch:
        :param cnn_epoch:
        :param predict:
        :param k_fold:
        :return:
        """

        dic = {'AutoEncode': 'cnn', 'Auto_conv': True, 'Auto_dim': 1, 'Auto_batch_size': 256, 'Auto_layer': 3, 'Auto_in_actName': 'Tanh', 'batch_size': 64, 'se': 1}

        load_data = RPIseqDataset(self.file_path, dic.get('Auto_conv'), model_type=dic.get('AutoEncode'))
        n_features = load_data.features.shape[-1]
        sample = load_data.target_transform.size(0)

        # 自动编码模型
        status = './auto_ok.txt'
        auto_model = AutoEncodeCnn(max_pool=True,in_actName=dic['Auto_in_actName'],layer=dic['Auto_layer'],n_features=n_features,dim=dic['Auto_dim'],device=device).to(device)
        print(auto_model)
        # deepcnn模型
        channels = 384
        layer = [3, 4, 6, 3]  # Block结构
        model = DeepCnn(layer, channels=channels, se_b=dic['se']).to(device)

        if os.path.isfile(status):
            # 自编码器已经训练完成
            auto_model.eval()
            state_dict = torch.load('{}/model.pth'.format(self.auto_path))
            auto_model.load_state_dict(state_dict, strict=False)
        else:
            status = self.train_auto(auto_model,load_data,dic['Auto_batch_size'],n_epoch,sample)
            if status:
                os.system('touch {}'.format(status))
            # 训练完，要加载最好的模型结果来预测
            auto_model.eval()
            state_dict = torch.load('{}/model.pth'.format(self.auto_path))
            auto_model.load_state_dict(state_dict, strict=False)
        dt = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
        # # 开始训练deepcnn
        if k_fold == 1:
            # 进行十折交叉验证
            self.k_fold(load_data, auto_model, dic, cnn_epoch,k)
        elif predict:
            # 直接预测
            filename = f'{dt}_predict.csv'
            pred_dataset = TensorDataset(load_data.transform)
            model.eval()
            state_dict = torch.load('{}/model.pth'.format(self.cnn_path))
            model.load_state_dict(state_dict, strict=False)
            print("{}开始进行预测{}".format('*' * 30, '*' * 30))
            preds = self.predict(model, auto_model, pred_dataset)
            self.save_file(preds,filename)
            print("{}保存预测结果{}".format('*' * 30, '*' * 30))
        else:
            # 进行训练
            # 数据划分6：2：2
            loss_fn = torch.nn.BCEWithLogitsLoss()
            train_size = int(sample * 0.7)
            test_size = sample - train_size
            valid_size = test_size // 2
            test_size = test_size - valid_size
            dataset = TensorDataset(load_data.transform, load_data.target_transform)
            train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, valid_size])
            self.train(model, auto_model, train_dataset, valid_dataset, dic['batch_size'], cnn_epoch, (train_size,test_size),loss_fn)
            print("{}训练结束{}".format('*'*35,'*'*35))
            # 测试
            model.eval()
            state_dict = torch.load('{}/model.pth'.format(self.cnn_path))
            model.load_state_dict(state_dict, strict=False)

            self.evaluate(model,auto_model,test_dataset,loss_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型命令行参数')
    parser.add_argument('-n', '--auto_epoch', type=int, default=3000,help='训练迭代次数')
    parser.add_argument('-e', '--cnn_epoch', type=int, default=120, help='训练迭代次数')
    parser.add_argument('-k', '--k_fold', type=int, default=1, help='是否交叉验证')
    parser.add_argument('-s', '--ks', type=int, default=10, help='折')
    parser.add_argument('-p', '--predict', type=int, default=0, help='直接预测，已经保存好训练的模型')
    parser.add_argument('-f', '--path', type=str, default='./data/RPIseq_32388_ros.csv',help='样本文件路径')
    args = parser.parse_args()
    print(args)
    rpi = RPIModel(file_path=args.path)
    rpi.run(args.auto_epoch,args.cnn_epoch,args.predict,args.k_fold,args.ks)

