import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCNN_3(torch.nn.Module):

    def __init__(self, out_channel):
        super(MSCNN_3, self).__init__()

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

        self.feature_layer = nn.Sequential(
            nn.Linear(4352, 1024),  # 全连接 120
            nn.ReLU(),  # relu激活
            nn.Dropout(0.2),
            nn.Linear(1024, out_channel),  # 全连接 84
        )

    def forward(self, data_1,data_2,data_3):
        x_1 = data_1
        x_2 = data_2
        x_3 = data_3

        x_1 = torch.unsqueeze(x_1,dim=1)
        x_2 = torch.unsqueeze(x_2, dim=1)
        x_3 = torch.unsqueeze(x_3, dim=1)

        x_1 = self.net1(x_1)
        shape_1=x_1.shape[1]
        shape_2 = x_1.shape[2]
        x_1 = x_1.view(-1, shape_1*shape_2)
        x_2 = self.net2(x_2)
        shape_3=x_2.shape[1]
        shape_4 = x_2.shape[2]
        x_2 = x_2.view(-1, shape_3*shape_4)
        x_3 = self.net3(x_3)
        shape_5 = x_3.shape[1]
        shape_6 = x_3.shape[2]
        x_3 = x_3.view(-1, shape_5*shape_6)



        x = torch.cat([x_1, x_2, x_3], dim=1)
        fc = self.feature_layer(x)
        probas = F.softmax(fc, dim=1)# softmax分类器

        return fc,probas