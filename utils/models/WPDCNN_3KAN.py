import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan import FastKAN as KAN

class WPD_CNN_3KAN(torch.nn.Module):

    def __init__(self,out_channel):
        super(WPD_CNN_3KAN, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(32),
        )

        self.net2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(32),
        )

        self.net3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(32),
        )

        self.net4 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(32),
        )

        self.net5 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(32),
        )

        self.net6 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(32),
        )
        self.net7 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(32),
        )

        self.net8 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100,stride=2,padding=0),  # 卷积，433-5+1=429 ->1*429*6
            nn.MaxPool1d(kernel_size=2),  # 池化，429/3=143 ->  1*143*6
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=100,stride=2,padding=0),  # 卷积，143-3+1=141 ->1*141*16
            nn.MaxPool1d(kernel_size=2),  # 池化，141/3=47 ->1*47*16
            nn.ReLU(inplace=True),  # relu激活
            nn.BatchNorm1d(32),
        )

        self.feature_layer_1 = nn.Sequential(
            nn.Linear(8448, 1024),  # 全连接 120
            nn.ReLU(),  # relu激活
            nn.Dropout(0.2),
            #nn.Linear(1024, out_channel),  # 全连接 84
        )
        self.feature_layer_2 = KAN([1024,64,out_channel])



    def forward(self, data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8):
        x_1 = data_1
        x_2 = data_2
        x_3 = data_3
        x_4 = data_4
        x_5 = data_5
        x_6 = data_6
        x_7 = data_7
        x_8 = data_8

        x_1 = torch.unsqueeze(x_1,dim=1)
        x_2 = torch.unsqueeze(x_2, dim=1)
        x_3 = torch.unsqueeze(x_3, dim=1)
        x_4 = torch.unsqueeze(x_4,dim=1)
        x_5 = torch.unsqueeze(x_5, dim=1)
        x_6 = torch.unsqueeze(x_6, dim=1)
        x_7 = torch.unsqueeze(x_7,dim=1)
        x_8 = torch.unsqueeze(x_8, dim=1)
        x_1 = self.net1(x_1)
        shape_1=x_1.shape[1]
        shape_2 = x_1.shape[2]
        x_1 = x_1.view(-1, shape_1*shape_2)
        x_2 = self.net2(x_2)
        x_2 = x_2.view(-1, shape_1*shape_2)
        x_3 = self.net3(x_3)
        x_3 = x_3.view(-1, shape_1*shape_2)
        x_4 = self.net4(x_4)
        x_4 = x_4.view(-1, shape_1*shape_2)
        x_5 = self.net5(x_5)
        x_5 = x_5.view(-1, shape_1*shape_2)
        x_6 = self.net6(x_6)
        x_6 = x_6.view(-1, shape_1*shape_2)
        x_7 = self.net7(x_7)
        x_7 = x_7.view(-1, shape_1*shape_2)
        x_8 = self.net8(x_8)
        x_8 = x_8.view(-1, shape_1*shape_2)

        x_9 = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], dim=1)
        fc_1 = self.feature_layer_1(x_9)
        fc = self.feature_layer_2(fc_1)
        #x=torch.unsqueeze(x,1)
        probas = F.softmax(fc, dim=1)# softmax分类器
        return fc,probas