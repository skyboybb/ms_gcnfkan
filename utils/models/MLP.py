import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
import torch.nn.functional as F
from fastkan import FastKAN as KAN
class MLP(torch.nn.Module):
    def __init__(self, feture,out_channel):
        super(MLP, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(feture, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        #self.fc4 = nn.Sequential(nn.Linear(128, out_channel))
        self.feature_layer_4 = KAN([128, 32, out_channel])


    def forward(self, data):
        x= data

        x = self.fc1(x)

        #x = self.fc2(x)

        x = self.fc3(x)
        x = self.dropout(x)
        x = self.feature_layer_4(x)
        #x = self.fc4(x)
        probas = F.softmax(x , dim=1)  # softmax分类器

        return x,probas