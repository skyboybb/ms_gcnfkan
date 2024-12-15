import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan import FastKAN as KAN

class CNN(torch.nn.Module):

    def __init__(self, out_channel):
        super(CNN, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(128),
        )
        self.feature_layer = nn.Sequential(
            nn.Linear(128, 64),  # 全连接 120
            nn.ReLU(),  # relu激活
            nn.Dropout(0.2),
            #nn.Linear(64, out_channel),  # 全连接 84
        )
        self.feature_layer_2 = KAN([64, 32, out_channel])

    def forward(self, data_1):
        x_1 = data_1
        x_1 = torch.unsqueeze(x_1,dim=1)
        x_1 = self.net1(x_1)
        shape_1=x_1.shape[1]
        shape_2 = x_1.shape[2]
        x_1 = x_1.view(-1, shape_1*shape_2)

        fc = self.feature_layer(x_1)
        fc = self.feature_layer_2(fc)
        probas = F.softmax(fc, dim=1)# softmax分类器

        return fc,probas