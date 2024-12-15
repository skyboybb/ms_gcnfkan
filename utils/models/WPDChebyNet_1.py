import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
from torch_geometric.utils import dropout_adj
class Transformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim,num_layers,num_heads):
        super(Transformer, self).__init__()

class ChebyNet_1(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(ChebyNet_1, self).__init__()

        self.GConv1_1 = ChebConv(feature,512,K=1)
        self.bn1_1 = BatchNorm(512)
        self.GConv2_1 = ChebConv(512,512,K=1)
        self.bn2_1 = BatchNorm(512)
        #self.dropout1_1 = nn.Dropout(0.5)
        #self.fc2_1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True))
        #self.dropout2_1 = nn.Dropout(0.5)

        self.GConv1_2 = ChebConv(feature,512,K=1)
        self.bn1_2 = BatchNorm(512)
        self.GConv2_2 = ChebConv(512,512,K=1)
        self.bn2_2 = BatchNorm(512)

        self.feature_layer = nn.Sequential(
            nn.Linear(1024, 512),  # 全连接 120
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),  # relu激活
            nn.Linear(512, out_channel),  # 全连接 84
        )


    def forward(self, data_1,data_2):
        x_1, edge_index_1, edge_weight_1 = data_1.x, data_1.edge_index, data_1.edge_attr
        x_1 = self.GConv1_1(x_1, edge_index_1, edge_weight_1)
        x_1 = self.bn1_1(x_1)
        x_1 = F.relu(x_1)
        edge_index_1, edge_weight_1 = dropout_adj(edge_index_1, edge_weight_1)
        x_1 = self.GConv2_1(x_1, edge_index_1, edge_weight_1)
        x_1 = self.bn2_1(x_1)
        x_1 = F.relu(x_1)
        x_1 = self.fc1_1(x_1)
        #x_1 = self.dropout1_1(x_1)
        #x_1 = self.fc2_1(x_1)
        #x_1 = self.dropout2_1(x_1)

        x_2, edge_index_2, edge_weight_2 = data_2.x, data_2.edge_index, data_2.edge_attr
        x_2 = self.GConv1_2(x_2, edge_index_2, edge_weight_2)
        x_2 = self.bn1_2(x_2)
        x_2 = F.relu(x_2)
        edge_index_2, edge_weight_2 = dropout_adj(edge_index_2, edge_weight_2)
        x_2 = self.GConv2_2(x_2, edge_index_2, edge_weight_2)
        x_2 = self.bn2_2(x_2)
        x_2 = F.relu(x_2)
        x_2 = self.fc1_2(x_2)

        x=torch.cat([x_1,x_2],dim=1)
        fc=self.feature_layer(x)

        return fc