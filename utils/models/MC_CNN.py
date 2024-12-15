import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan import FastKAN as KAN

class MC_CNN(torch.nn.Module):

    def __init__(self, out_channel):
        super(MC_CNN, self).__init__()

        self.net0 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=100,stride=1,padding=0))
        self.net0_1 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=200, stride=1, padding=0))
        self.net0_2 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=300, stride=1, padding=0))

        self.net1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=8,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 8, kernel_size=32,stride=4,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 8, kernel_size=16, stride=2, padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(8),
        )
        self.feature_layer = nn.Sequential(nn.Linear(112, out_channel))  # 全连接 120
        #self.feature_layer_2 = KAN([112, 64, out_channel])

    def forward(self, data_1):
        x_1 = torch.unsqueeze(data_1,dim=1)
        x_2 = torch.unsqueeze(data_1,dim=1)
        x_3 = torch.unsqueeze(data_1, dim=1)
        x_1 = self.net0(x_1)
        x_2 =self.net0_1(x_2)
        x_3 = self.net0_2(x_3)

        x_4 = torch.cat([x_1,x_2,x_3],dim=2)



        x_4 = self.net1(x_4)
        shape_1=x_4.shape[1]
        shape_2 = x_4.shape[2]
        x_4 = x_4.view(-1, shape_1*shape_2)

        fc = self.feature_layer(x_4)
        #fc = self.feature_layer_2(x_4)
        probas = F.softmax(fc, dim=1)# softmax分类器

        return fc,probas