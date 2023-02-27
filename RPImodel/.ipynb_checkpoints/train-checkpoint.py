# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: tarin.py
import argparse
import os
import time
from copy import deepcopy
import torch
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

from datasets import RPIseqDataset
from autocode import AutoEncodeCnn
from cnn import DeepCnn

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    average_precision_score
from sklearn.model_selection import StratifiedKFold

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

torch.set_num_threads(16)
num_workers = 0

learning_rate = 0.000001

curr_path = os.getcwd()

if 'autodl-tmp' in curr_path:
    logs_path = '../../tf-logs'
else:
    logs_path = './logs'

def train_auto(model,dataset,batch_size,n_epoch,path,sample,step=10):

    if os.path.exists(path) and len(os.listdir(path))>0:
        flag = True
    else:
        flag = False

    loss_fn = torch.nn.MSELoss()
    # 优化器
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                                eps=1e-8, verbose=True)

    if flag:
        checkoint = torch.load(f'{path}/model.pth')
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
                    get_cur_lr(optimizer)))
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
            writer.add_scalar('Auto_Epoch/Lr', get_cur_lr(optimizer),epoch)

        if test_loss < previous_loss and epoch > 5:
            # 如果当前损失小于上一个就保存
            previous_loss = test_loss
            max_count = 0
            state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}
            torch.save(state_dict, f'{path}/model.pth')
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


def train_cnn(model,auto_model,train_dataset,test_dataset,batch_size,n_epoch,path,sample,num,step=40):

    if os.path.exists(path) and len(os.listdir(path)) > 0:
        flag = True
    else:
        flag = False
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 优化器
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters(),weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                                eps=1e-9, verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40,50,60,65,70,75,80,85,90,95],gamma=0.1,verbose=True)
    # if flag:
    #     checkoint = torch.load(f'{path}/model.pth')
    #     model.load_state_dict(checkoint['model'])
    #     optimizer.load_state_dict(checkoint['optimizer'])
    #     start_num = checkoint['epoch']
    # else:
    #     start_num = 0
    start_num = 0

    total_test_loss, total_test_acc,total_auc, total_auprc, total_accuracy, total_precision, total_recall, total_f1 = 0,0,0, 0, 0, 0, 0, 0
    total_loss, total, correct = 0, 0, 0
    previous_acc = 0.0
    previous_loss = 1.0
    train_ls, test_ls,kpi = [],[],[]
    train_acc = 0.0
    max_count = 0
    log_name = '{}_{}'.format('DeepCnn', time.strftime('%m-%d_%H.%M', time.localtime()))
    log_dir = os.path.join(logs_path, log_name)
    writer = SummaryWriter(log_dir=log_dir)

    max_params = {}

    # train_size = int(sample * 0.8)
    # test_size = sample - train_size
    # #
    # train_dataset, test_dateset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True,
                                            num_workers=num_workers)
    print(start_num,n_epoch)
    for epoch in range(start_num,n_epoch):
        model.train()
        print("{}DeepCnn Epoch: [{}/{}] {}".format('=' * 40, epoch + 1, n_epoch, '=' * 40))
        running_loss = 0.0

        start = time.time()  # 计算耗时
        for i,(x,y) in enumerate(train_data):

            x, y = x.to(device), y.to(device)
            encoder = predict(auto_model,x)
            out = model(encoder)
            # loss = loss_fn(out, y)
            loss = criterion(loss_fn,out,y)
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
                print("Sample:[{}] Step: [{}/{}], train_loss: {:.10f} | train_acc: {:.8f}% | lr: {:.16f}".format(
                    sample[0], str(i + 1).zfill(3), len(train_data), running_loss / (i + 1), train_acc,
                    get_cur_lr(optimizer)))
                s = (i + 1) + (epoch * len(train_data))
                writer.add_scalar('Train:Loss/Step', running_loss / (i + 1), s)
                writer.add_scalar('Train:Acc/Step', train_acc, s)

        epoch_loss = running_loss / len(train_data)
        total_loss += epoch_loss
        end = time.time()
        print('{} 耗时:{:.4f}s {}'.format('-' * 45, end - start, '-' * 45))

        train_ls.append((epoch_loss,train_acc))

        # 进行测试集验证
        print("{} Test data {}".format('*' * 45, '*' * 45))
        test_loss, test_acc, auc, auprc, accuracy, precision, recall, f1 = valid(model,auto_model, test_data,loss_fn)
        writer.add_scalar('Test:Epoch/Loss', test_loss, epoch)
        writer.add_scalar('Test:Epoch/Auc', auc, epoch)
        writer.add_scalar('Test:Epoch/Auprc', auprc, epoch)
        writer.add_scalar('Test:Epoch/Acc', test_acc, epoch)
        writer.add_scalar('Test:Epoch/Precision', precision, epoch)
        writer.add_scalar('Test:Epoch/Recall', recall, epoch)
        writer.add_scalar('Test:Epoch/F1', f1, epoch)

        total_test_loss += test_loss
        total_test_acc += test_acc
        total_auc += auc
        total_auprc += auprc
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        if scheduler:
            scheduler.step(epoch_loss)
            writer.add_scalar('Epoch/Lr', get_cur_lr(optimizer),epoch)

        if test_acc > previous_acc:
            # 如果当前准确率大于上一个保存
            previous_acc = test_acc
            max_count = 0
            # state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}
            # torch.save(state_dict, f'{path}/model.pth')
            parameter = {}
            parameter['epoch'] = epoch
            parameter['lr'] = get_cur_lr(optimizer)
            parameter['k_fold'] = num
            writer.add_hparams(dict(parameter),{"Auc":auc,"Auprc":auprc,"Acc": accuracy, "Loss": test_loss, 'Recall': recall, 'F1': f1, 'Precision': precision})
            c =epoch+1
            max_params['epoch'] = c
            max_params['loss'] = deepcopy(test_loss)
            max_params['acc'] = deepcopy(test_acc)
            max_params['auc'] = deepcopy(auc)
            max_params['auprc'] = deepcopy(auprc)
            max_params['accuracy'] = deepcopy(accuracy)
            max_params['precision'] = deepcopy(precision)
            max_params['recall'] = deepcopy(recall)
            max_params['f1'] = deepcopy(f1)
        else:
            max_count += 1

        if epoch_loss < previous_loss:
            previous_loss = epoch_loss

        print("[Epochs:{}|max_count:{}|{}] Sample: [{}] Test_loss: {:.10f} | Test_acc: {:.8f} | previous_acc: {:.8f} | Train_loss: {:.10f} | Previous_loss: {:.10f}".format(epoch,max_count,num,sample[-1], test_loss, test_acc,previous_acc,epoch_loss,previous_loss))

        print("{} Test end {}".format('*' * 45, '*' * 45))
        # test_ls.append((test_loss,auc))
        # kpi.append({"k_fold":num,"Auc":auc,"Auprc":auprc,"Acc": accuracy, "Loss": test_loss, 'Recall': recall, 'F1': f1, 'Precision': precision})
        if max_count > 300:
            break

    writer.close()
    print(f"*****************************DeepCnn训练结束:{max_count}*****************************************")
    return total_test_loss / n_epoch, total_test_acc / n_epoch, total_accuracy / n_epoch, total_precision / n_epoch, total_recall / n_epoch, total_f1 / n_epoch,total_auc/n_epoch,total_auprc/n_epoch,max_params


def criterion(loss_fn,out,y):
    # 根据损失函数，改变数据形状和类型:BCELoss+Sigmoid=BCEWithLogitsLoss
    if isinstance(loss_fn,torch.nn.BCEWithLogitsLoss):
        y_hot = torch.nn.functional.one_hot(y, 2)
        y_hot = torch.as_tensor(y_hot, dtype=torch.float32)
    else:
        y_hot = y
    loss = loss_fn(out,y_hot).to(device)
    return loss


def valid(model,auto_model, test_data,loss_fn):
    total, correct = 0, 0
    running_loss = 0.
    test_preds, test_trues = [], []
    model.eval()  # 测试模式

    with torch.no_grad():  # 不计算梯度
        for i, (x, y) in enumerate(test_data):
            x, y = x.to(device), y.to(device)  # CPU or GPU运行
            encoder = predict(auto_model,x)
            out = model(encoder.to(device))  # 计算输出
            loss = criterion(loss_fn,out, y)  # 计算损失
            running_loss += loss.item()
            total += y.size(0)  # 计算测试集总样本数
            correct += (out.argmax(dim=1) == y).type(torch.float).sum().item()  # 计算测试集预测准确的样本数

            test_outputs = out.argmax(dim=1)
            test_preds.extend(test_outputs.detach().cpu().numpy())
            test_trues.extend(y.detach().cpu().numpy())

    test_acc = (correct / total) * 100.0  # 测试集准确率

    test_loss = running_loss / len(test_data)
    # 就算指标
    auc, auprc, accuracy, precision, recall, f1 = report(test_trues, test_preds, test_acc)

    # 训练模式 （因为这里是因为每经过一个Epoch就使用测试集一次，使用测试集后，进入下一个Epoch前将模型重新置于训练模式）
    return test_loss, test_acc, auc, auprc,accuracy, precision, recall, f1


def predict(net, x):
    net.eval()
    with torch.no_grad():
        inputs = x.to(device)
        encoder, decoder = net(inputs)
        return encoder


def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def report(test_trues, test_preds, accuracy):
    # 分类报告 计算各项指标
    precision = precision_score(test_trues, test_preds, average='micro')
    recall = recall_score(test_trues, test_preds, average='micro')
    f1 = f1_score(test_trues, test_preds, average='micro')
    auc = roc_auc_score(test_trues, test_preds)
    auprc = average_precision_score(test_trues, test_preds)
    print(classification_report(test_trues, test_preds))

    print("[sklearn_metrics] auc:{:.4f} auprc:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
        auc,auprc,accuracy,precision,recall,f1))
    return auc,auprc,accuracy,precision,recall,f1

def run(n_epoch,cnn_epoch):
    path = 'data/RPIseq_32388_ros.csv'
    dic = {'AutoEncode': 'cnn', 'Auto_conv': True, 'Auto_dim': 1, 'Auto_batch_size': 256, 'Auto_layer': 3, 'Auto_n_epoch': 2, 'Auto_in_actName': 'Tanh', 'Auto_train_size': 1, 'Auto_loss_fn': 'MSELoss', 'DeepCnn': 'resnet', 'DeepCnn_layer': 50, 'epoch': 2, 'batch_size': 32, 'train_size': 0.8, 'se': 1, 'loss_fn': 'CrossEntropyLoss', 'bsize': 737}

    load_data = RPIseqDataset(path, dic.get('Auto_conv'), model_type=dic.get('AutoEncode'))
    n_features = load_data.features.shape[-1]
    sample = load_data.target_transform.size(0)

    # 自动编码模型
    auto_path = "./models/AE"
    auto_model = AutoEncodeCnn(max_pool=True,in_actName=dic['Auto_in_actName'],layer=dic['Auto_layer'],n_features=n_features,dim=dic['Auto_dim'],device=device).to(device)
    if os.path.isfile('./auto_ok.txt'):
        # 自编码器已经训练完成
        auto_model.eval()
        state_dict = torch.load('{}/model.pth'.format(auto_path))
        auto_model.load_state_dict(state_dict, strict=False)
    else:
        status = train_auto(auto_model,load_data,dic['Auto_batch_size'],n_epoch,auto_path,sample)
        if status:
            os.system('touch ./auto_ok.txt')
        # 训练完，要加载最好的模型结果来预测
        auto_model.eval()
        state_dict = torch.load('{}/model.pth'.format(auto_path))
        auto_model.load_state_dict(state_dict, strict=False)

    # # 开始训练deepcnn
    # print('***********************DeepCnn******************************')
    # cnn_path = './models/CNN'
    # channels = 384
    # layer = [3, 4, 6, 3]  # 50
    # cnn_model = DeepCnn(layer,channels=channels,se_b=dic['se']).to(device)
    # train_cnn(cnn_model,auto_model,load_data,dic['batch_size'],cnn_epoch,cnn_path,sample)

    # 进行十折交叉验证
    k_fold(load_data,auto_model,dic,cnn_epoch)
    # 检验模型泛化能力
    # assessment()

def assessment():
    dic = {'AutoEncode': 'cnn', 'Auto_conv': True, 'Auto_dim': 1, 'Auto_batch_size': 256, 'Auto_layer': 3,
           'Auto_n_epoch': 2, 'Auto_in_actName': 'Tanh', 'Auto_train_size': 1, 'Auto_loss_fn': 'MSELoss',
           'DeepCnn': 'resnet', 'DeepCnn_layer': 50, 'epoch': 2, 'batch_size': 64, 'train_size': 0.8, 'se': 4,
           'loss_fn': 'CrossEntropyLoss', 'bsize': 737}

    path1 = curr_path + "/test_data/RPIseq369_new.csv" # 565
    path2 = curr_path + "/test_data/RPIseq488_new.csv" # 488
    path3 = curr_path + "/test_data/RPIseq1807_new.csv" # 3111
    path4 = curr_path + "/test_data/RPIseq2241_new.csv" # 4012
    load_data_1 = RPIseqDataset(path1, True, model_type='cnn')
    load_data_2 = RPIseqDataset(path2, True, model_type='cnn')
    load_data_3 = RPIseqDataset(path3, True, model_type='cnn')
    load_data_4 = RPIseqDataset(path4, True, model_type='cnn')

    n_features = load_data_1.features.shape[-1]
    sample = load_data_1.target_transform.size(0)
    print(sample,n_features)

    auto_path = "./models/AE"
    auto_model = AutoEncodeCnn(max_pool=True, in_actName=dic['Auto_in_actName'], layer=dic['Auto_layer'],
                               n_features=n_features, dim=dic['Auto_dim'], device=device).to(device)
    auto_model.eval()
    state_dict = torch.load('{}/model.pth'.format(auto_path))
    auto_model.load_state_dict(state_dict, strict=False)

    cnn_path = './models/CNN'
    channels = 384
    layer = [3, 4, 6, 3]  # 50
    model = DeepCnn(layer,channels=channels,se_b=dic['se']).to(device)
    model.eval()
    state_dict = torch.load('{}/model.pth'.format(cnn_path))
    model.load_state_dict(state_dict, strict=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    for i,data in enumerate([load_data_1,load_data_2,load_data_3,load_data_4]):
        print(f"******************{i}**************")
        test_data = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True,
                                                num_workers=num_workers)
        test_loss, test_acc, auc, auprc, accuracy, precision, recall, f1 = valid(model=model,auto_model=auto_model,test_data=test_data,loss_fn=loss_fn)


def k_fold(load_data,auto_model,dic,epoch,k=10):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    accuracy_sum,precision_sum,recall_sum,F1_sum,auc_sum,auprc_sum = 0.0,0.0,0.0,0.0,0.0,0.0

    skf = StratifiedKFold(n_splits=k, shuffle=True,random_state=22)
    X, y = load_data.transform, load_data.target_transform
    dd = skf.split(X, y)
    log_name = '{}_{}'.format('k_fold', time.strftime('%m-%d_%H.%M', time.localtime()))
    log_dir = os.path.join(logs_path, log_name)
    writer = SummaryWriter(log_dir=log_dir)

    for i, (train_index, test_index) in enumerate(dd):
        k = i+1
        # 获取k折交叉验证的训练和验证数据
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        sample = (len(y_train),len(y_test))
        # 开始训练deepcnn,每份数据进行训练
        print('***********************DeepCnn******************************')
        cnn_path = './models/CNN'
        channels = 384
        layer = [3, 4, 6, 3]  # 50
        cnn_model = DeepCnn(layer, channels=channels, se_b=dic['se']).to(device)
        valid_loss,valid_acc,accuracy,precision,recall,F1,auc,auprc, max_params = train_cnn(cnn_model, auto_model, train_dataset,test_dataset, dic['batch_size'], epoch, cnn_path, sample,k)

        print('*' * 25, '第', k, '折', '*' * 25)
        print("[k_flod:{}] valid_loss: {:.10f} | valid_acc: {:.8f}| auc: {:.8f} | auprc: {:.8f} | accuracy: {:.10f} | precision: {:.10f} | recall: {:.10f} | recall: {:.10f}".format(k, max_params['loss'], max_params['acc'],max_params['auc'],max_params['auprc'], max_params['accuracy'], max_params['precision'], max_params['recall'],max_params['f1']))
        print("[k_flod:{}] valid_loss: {:.10f} | valid_acc: {:.8f}| auc: {:.8f} | auprc: {:.8f} | accuracy: {:.10f} | precision: {:.10f} | recall: {:.10f} | recall: {:.10f}".format(k, max_params['loss']/k, max_params['acc']/k,max_params['auc']/k,max_params['auprc']/k, max_params['accuracy']/k, max_params['precision']/k, max_params['recall']/k,max_params['f1']/k))

        valid_loss_sum += max_params['loss']
        valid_acc_sum += max_params['acc']
        accuracy_sum += max_params['accuracy']
        precision_sum += max_params['precision']
        recall_sum += max_params['recall']
        F1_sum += max_params['f1']
        auc_sum += max_params['auc']
        auprc_sum += max_params['auprc']
        max_params['k'] = k
        writer.add_hparams(max_params,{"Auc":auc,"Auprc":auprc,"Acc": accuracy, "Loss": valid_loss, 'Recall': recall, 'F1': F1, 'Precision': precision})
    writer.close()
    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    print("[k_flod:{}] valid_loss: {:.10f} | valid_acc: {:.8f}| auc: {:.8f} | auprc: {:.8f} | accuracy: {:.10f} | precision: {:.10f} | recall: {:.10f} | recall: {:.10f}".format(k, valid_loss_sum/ k, valid_acc_sum/k, auc_sum/k, auprc_sum/k, accuracy_sum/k, precision_sum/k, recall_sum/k, F1_sum/k))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型命令行参数')
    parser.add_argument('-n', '--auto_epoch', type=int, default=3000,help='训练迭代次数')
    parser.add_argument('-e', '--cnn_epoch', type=int, default=3000, help='训练迭代次数')
    args = parser.parse_args()
    run(args.auto_epoch,args.cnn_epoch)
    # assessment()
