#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
from matplotlib.font_manager import FontProperties
import os
from sklearn.metrics import confusion_matrix
import itertools
import time
import warnings
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch import optim
import sys
import pandas as pd
sys.path.append('D:/pycharm/pythonProject/ms_CNN/utils')
import models
import datasets
from save import Save_Tool
from torch_geometric.data import DataLoader
import freeze
import matplotlib.pyplot as plt
import numpy as np
class train_utils_ms(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets_1 = {}
        self.datasets_2 = {}
        self.datasets_3 = {}
        self.label = {}

        self.datasets_1['train'], self.datasets_2['train'],self.datasets_3['train'],self.datasets_1['val'], self.datasets_2['val'],self.datasets_3['val'],\
        self.label['train'],self.label['val'] = Dataset(args.sample_length,args.data_dir, args.Input_type, args.task).data_preprare()
        self.N = self.datasets_1['train'].size(0)
        # Define the model
        self.model = getattr(models, args.model_name)(out_channel=Dataset.num_classes) #卷积使用这段代码

        if args.layer_num_last != 0:
            freeze.set_freeze_by_id(self.model, args.layer_num_last)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        args = self.args
        batch_size=args.batch_size
        num_epochs = args.max_epoch#迭代次数
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def plot_confusion_matrix(cm, labels_name):
            # cm = cm.astype('float') / cm.sum(axis=1)# 归一化
            # cm=np.round(cm,2)
            plt.figure()
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.imshow(cm, cmap=plt.cm.coolwarm, interpolation='nearest')  # 在特定的窗口上显示图像
            # plt.title(title)    # 图像标题
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j], fontsize=12, verticalalignment="center", horizontalalignment="center")
            plt.colorbar()
            num_local = np.array(range(len(labels_name)))
            plt.xticks(num_local, labels_name)  # 将标签印在x轴坐标上
            plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
            plt.ylabel('True label', fontsize=16)
            plt.xlabel('Predicted label', fontsize=16)
            plt.tick_params(axis='y', which='major', direction='in', labelsize=16)
            plt.tick_params(axis='x', which='major', direction='in', labelsize=16)
            plt.show()

        def compute_accuracy(model,feature_1,feature_2,feature_3,labels):
            correct_pred, num_examples = 0, 0
            l = 0
            N = feature_1.size(0)
            total_batch = int(np.ceil(N / batch_size))
            indices = np.arange(N)
            np.random.shuffle(indices)
            for i in range(total_batch):
                rand_index = indices[batch_size * i:batch_size * (i + 1)]
                features_1 = feature_1[rand_index, :]
                features_2 = feature_2[rand_index, :]
                features_3 = feature_3[rand_index, :]
                targets = labels[rand_index]
                features_1 = features_1.to(device)
                features_2 = features_2.to(device)
                features_3 = features_3.to(device)

                targets = targets.to(device)
                logits, probas = model(features_1,features_2,features_3)
                cost = loss(logits, targets)
                _, predicted_labels = torch.max(probas, 1)

                num_examples += targets.size(0)
                l += cost.item()
                correct_pred += (predicted_labels == targets).sum()

            return l / num_examples, correct_pred.float() / num_examples * 100, logits

        loss=torch.nn.CrossEntropyLoss()#交叉熵损失
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        for epoch in range(num_epochs):
            epoch_start = time.time()
            model = self.model.train()  # 启用dropout
            total_batch = int(np.ceil(self.N/ batch_size))
            indices = np.arange(self.N)
            np.random.shuffle(indices)
            avg_loss = 0
            for i in range(total_batch):
                rand_index = indices[batch_size * i:batch_size * (i + 1)]
                features_1 = self.datasets_1['train'][rand_index, :]
                features_2 = self.datasets_2['train'][rand_index, :]
                features_3 = self.datasets_3['train'][rand_index, :]
                targets = self.label['train'][rand_index]
                features_1 = features_1.to(device)
                features_2 = features_2.to(device)
                features_3 = features_3.to(device)
                targets = targets.to(device)
                logits, probas = model(features_1,features_2,features_3)
                cost = loss(logits, targets)
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

            model = model.eval()  # 关闭dropout
            with torch.set_grad_enabled(False):  # save memory during inference
                trl, trac, _ = compute_accuracy(model, self.datasets_1['train'],self.datasets_2['train'],self.datasets_3['train'],self.label['train'])
                val, vaac, _ = compute_accuracy(model, self.datasets_1['val'],self.datasets_2['val'],self.datasets_3['val'], self.label['val'])
                print('Epoch: %03d/%03d training accuracy: %.2f%% testing accuracy: %.2f%% Single diagnosis time:%.4f' % (
                    epoch + 1, num_epochs,
                    trac,
                    vaac,
                    time.time()-epoch_start))

            train_loss.append(trl)
            valid_loss.append(val)

            train_acc.append(trac)
            valid_acc.append(vaac)
        torch.save(model, 'trained_model/MSCNN_3.pth')  # 保存整个网络参数
        # In[]
        # loss曲线
        font_properties = FontProperties(size=24)
        fig, ax = plt.subplots(figsize=(8, 6))
        # plt.figure()
        ax.plot(np.array(train_loss), label='Training set')
        ax.plot(np.array(valid_loss), label='Testing set')
        ax.tick_params(axis='x', which='major', direction='in', labelsize=24)  # x 轴刻度标签字体大小
        ax.tick_params(axis='y', which='major', direction='in', labelsize=24)  # y 轴刻度标签字体大小
        # plt.title('损失曲线')
        plt.legend(prop=font_properties)
        plt.show()
        # accuracy 曲线
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(np.array(train_acc), label='Training set')
        ax1.plot(np.array(valid_acc), label='Testing set')
        ax1.tick_params(axis='x', which='major', direction='in', labelsize=24)  # x 轴刻度标签字体大小
        ax1.tick_params(axis='y', which='major', direction='in', labelsize=24)  # y 轴刻度标签字体大小
        # plt.title('准确率曲线')
        plt.legend(prop=font_properties)
        plt.show()

        labels = self.label['val'] .to(self.device)
        model = torch.load('trained_model/MSCNN_3.pth')
        inputs_1 = self.datasets_1['val'].to(self.device)
        inputs_2 = self.datasets_2['val'].to(self.device)
        inputs_3 = self.datasets_3['val'].to(self.device)
        model.to(self.device)
        model.eval()
        logits, probas = model(inputs_1, inputs_2, inputs_3)
        pred = logits.argmax(dim=1)
        labels = labels.detach().numpy()
        pre = pred.detach().numpy()
        C = confusion_matrix(labels, pre)

        # 混淆矩阵
        plot_confusion_matrix(C, ['0', '1', '2', '3', '4', '5', '6', '7'])

        ####tsne可视化
        colors = ['black', 'blue', 'purple', 'yellow', 'magenta', 'red', 'lime', 'cyan', 'orange', 'gray']
        tsne = TSNE(n_components=2)
        X2 = tsne.fit_transform(logits.detach().numpy())

        X2 = (X2 - X2.min(axis=0)) / (X2.max(axis=0) - X2.min(axis=0))
        Y = labels
        plt.figure()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.direction'] = 'in'
        plt.plot(X2[Y == 0, 0], X2[Y == 0, 1], '*', markersize=5, label='0', c=colors[0])
        plt.plot(X2[Y == 1, 0], X2[Y == 1, 1], '*', markersize=5, label='1', c=colors[1])
        plt.plot(X2[Y == 2, 0], X2[Y == 2, 1], '*', markersize=5, label='2', c=colors[2])
        plt.plot(X2[Y == 3, 0], X2[Y == 3, 1], '*', markersize=5, label='3', c=colors[3])
        plt.plot(X2[Y == 4, 0], X2[Y == 4, 1], '*', markersize=5, label='4', c=colors[4])
        plt.plot(X2[Y == 5, 0], X2[Y == 5, 1], '*', markersize=5, label='5', c=colors[5])
        plt.plot(X2[Y == 6, 0], X2[Y == 6, 1], '*', markersize=5, label='6', c=colors[6])
        plt.plot(X2[Y == 7, 0], X2[Y == 7, 1], '*', markersize=5, label='7', c=colors[7])
        plt.legend()
        plt.xlabel('First component', fontsize=16)
        plt.ylabel('Second component', fontsize=16)
        plt.tick_params(axis='y', which='major', direction='in', labelsize=16)
        plt.tick_params(axis='x', which='major', direction='in', labelsize=16)
        # plt.title('feature_data')
        plt.show()
