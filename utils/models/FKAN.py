from fastkan import FastKAN as KAN

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
import torch.nn.functional as F

class FKAN(torch.nn.Module):
    def __init__(self, out_channel):
        super(FKAN, self).__init__()

        self.fc1 = nn.Sequential(
            KAN([512, 64, 256]),
            nn.BatchNorm1d(256))

        self.fc2 = nn.Sequential(
            KAN([256, 64, 128]),
            nn.BatchNorm1d(128))
        self.dropout = nn.Dropout(0.2)
        self.feature_layer_4 = KAN([128, 32, 64])
        self.feature_layer_5 = KAN([64, 32, out_channel])

        self.fc3 = nn.Sequential(nn.Linear(64, out_channel))


    def forward(self, data):
        x= data

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.feature_layer_4(x)
        x = self.feature_layer_5(x)
        #x = self.fc3(x)
        probas = F.softmax(x , dim=1)  # softmax分类器

        return x,probas