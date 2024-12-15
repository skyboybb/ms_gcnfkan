import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
from torch_geometric.utils import dropout_adj
from fastkan import FastKAN as KAN
class ChebyNet(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(ChebyNet, self).__init__()

        self.GConv1_1 = ChebConv(feature,512,K=1)
        self.bn1_1 = BatchNorm(512)
        self.GConv2_1 = ChebConv(512,512,K=1)
        self.bn2_1 = BatchNorm(512)

        self.feature_layer = nn.Sequential(
            nn.Linear(512, 128),  # 全连接 120
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),  # relu激活
            nn.Linear(128, out_channel),  # 全连接 84
        )
        self.feature_layer_2 = KAN([128, 32, out_channel])



    def forward(self, data_1):

        x_1, edge_index_1, edge_weight_1 = data_1.x, data_1.edge_index, data_1.edge_attr
        x_1 = self.GConv1_1(x_1, edge_index_1, edge_weight_1)
        x_1 = self.bn1_1(x_1)
        x_1 = F.relu(x_1)
        edge_index_1, edge_weight_1 = dropout_adj(edge_index_1, edge_weight_1)
        x_1 = self.GConv2_1(x_1, edge_index_1, edge_weight_1)
        x_1 = self.bn2_1(x_1)

        fc = self.feature_layer(x_1)
        #fc = self.feature_layer_2(fc)

        return fc