#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
import numpy as np
import torch
from utils.logger import setlogger
import logging
from utils.train_WPDCNN_utils import train_utils_wpd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.manifold import TSNE
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # basic parameters
    parser.add_argument('--model_name', type=str, default='WPD_CNN_3', help='the name of the model')
    parser.add_argument('--sample_length', type=int, default=2000, help='batchsize of the training process')
    parser.add_argument('--data_name', type=str, default='PDBONwpd_sets', help='the name of the data')#XJTUGearboxKnn
    #parser.add_argument('--data_name', type=str, default='MotorPath_a', help='the name of the data')  # XJTUGearboxKnn

    parser.add_argument('--Input_type', choices=['WPD_1','WPD_2','WPD_3'],type=str, default='WPD_3', help='the input type decides the length of input')

    parser.add_argument('--data_dir', type=str, default= "./data/PDBON", help='the directory of the data')#./data/XJTUGearbox/XJTUGearboxKnn.pkl
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # Define the tasks
    parser.add_argument('--task', choices=['Node','wpd'], type=str,
                        default='Node', help='Node classification or Graph classification')
    parser.add_argument('--pooltype', choices=['TopKPool', 'EdgePool', 'ASAPool', 'SAGPool'],type=str,
                        default='EdgePool', help='For the Graph classification task')

    # optimization information
    parser.add_argument('--layer_num_last', type=int, default=0, help='the number of last layers which unfreeze')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')
    parser.add_argument('--max_model_num', type=int, default=1, help='the number of most recent models to save')
    parser.add_argument('--max_epoch', type=int, default=150, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    args = parser.parse_args()
    return args

def plot_confusion_matrix(cm, labels_name):
    #cm = cm.astype('float') / cm.sum(axis=1)# 归一化
    #cm=np.round(cm,2)
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.imshow(cm,cmap=plt.cm.coolwarm,interpolation='nearest')    # 在特定的窗口上显示图像
    #plt.title(title)    # 图像标题
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],fontsize=12,verticalalignment="center",horizontalalignment="center")
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label',fontsize=16)
    plt.xlabel('Predicted label',fontsize=16)
    plt.tick_params(axis='y',which='major',direction='in',labelsize=16)
    plt.tick_params(axis='x', which='major', direction='in', labelsize=16)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    if args.task == 'wpd':
        sub_dir = args.task + '_' + args.model_name + '_' +args.data_name + '_' + args.Input_type +'_'+datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    else:
        sub_dir = args.task + '_' +  args.model_name + '_' + args.pooltype + '_' + args.data_name + '_' + args.Input_type + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils_wpd(args, save_dir)
    trainer.setup()
    trainer.train()








