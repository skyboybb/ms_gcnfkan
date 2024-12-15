#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import sys
from sklearn.manifold import TSNE
import itertools
import numpy as np
sys.path.append('D:/pycharm/pythonProject/ms_CNN/utils')
import models
import datasets
from save import Save_Tool
from torch_geometric.data import DataLoader
import freeze
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class train_utils(object):
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
        self.datasets = {}

        self.datasets['train'], self.datasets['val'] = Dataset(args.sample_length,args.data_dir, args.Input_type, args.task).data_preprare()

        self.dataloaders = {x: DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        # Define the model
        InputType = args.Input_type
        if InputType == "TD":
            feature = args.sample_length
        else:
            print("The InputType is wrong!!")

        if args.task == 'Node':
            self.model = getattr(models, args.model_name)(feature=feature,out_channel=Dataset.num_classes)
            print('The task is wrong!')

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
        # self.criterion = nn.MSELoss()


    def train(self):
        """
        Training process
        :return:
        """


        args = self.args

        epoch_acc_a=[]
        epoch_loss_a = []
        epoch_acc_a_1 = []
        epoch_loss_a_1 = []
        epoch_acc_b=[]
        epoch_loss_b = []

        epoch_acc_b_1 = []
        epoch_loss_b_1 = []

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        save_list = Save_Tool(max_num=args.max_model_num)

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                sample_num = 0
                for data in self.dataloaders[phase]:
                    inputs = data.to(self.device)
                    labels = inputs.y
                    if args.task == 'Node':
                        bacth_num = inputs.num_nodes
                        sample_num += len(labels)
                    elif args.task == 'Graph':
                        bacth_num = inputs.num_graphs
                        sample_num += len(labels)
                    else:
                        print("There is no such task!!")
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):

                        # forward
                        if  args.task == 'Node':
                            logits = self.model(inputs)
                        elif args.task == 'Graph':
                            logits = self.model(inputs,args.pooltype)
                        else:
                            print("There is no such task!!")

                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * bacth_num
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += bacth_num

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time
                                logging.info('Epoch: {}, Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_loss, batch_acc, sample_per_sec, batch_time
                                ))

                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Print the train and val information via each epoch

                epoch_loss = epoch_loss / sample_num
                epoch_acc = epoch_acc / sample_num


                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time()-epoch_start
                ))

                # save the model
                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc or epoch > args.max_epoch-2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))
                if phase == 'train':
                    epoch_acc_a.append(epoch_acc)
                    epoch_loss_a.append(epoch_loss)

                else:
                    epoch_acc_b.append(epoch_acc)
                    epoch_loss_b.append(epoch_loss)
        torch.save(self.model, 'trained_model/GCNCWRU.pth')  # 保存整个网络参数

        plt.figure()
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.direction'] = 'in'
        plt.plot(epoch_acc_a, c='r', label='train')
        plt.plot(epoch_acc_b, c='g', label='valid')
        plt.ylabel('cross entropy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        plt.figure()
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.direction'] = 'in'
        plt.plot(epoch_loss_a, c='r', label='train')
        plt.plot(epoch_loss_b, c='g', label='valid')
        plt.ylabel('cross entropy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

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

        phase='val'
        for data_1 in self.dataloaders[phase]:
            inputs_1 = data_1.to(self.device)
            labels = inputs_1.y
            labels = labels.to(self.device)
        model = torch.load('trained_model/GCNCWRU.pth')
        model.to(self.device)
        model.eval()
        logits = model(inputs_1)
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