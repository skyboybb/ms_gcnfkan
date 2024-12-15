import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
from torch_geometric.utils import dropout_adj







class ChebyNet_4(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(ChebyNet_4, self).__init__()

        self.GConv1_1 = ChebConv(feature,512,K=1)
        self.bn1_1 = BatchNorm(512)
        self.GConv2_1 = ChebConv(512,512,K=1)
        self.bn2_1 = BatchNorm(512)


        self.GConv1_2 = ChebConv(feature,512,K=1)
        self.bn1_2 = BatchNorm(512)
        self.GConv2_2 = ChebConv(512,512,K=1)
        self.bn2_2 = BatchNorm(512)

        self.GConv1_3 = ChebConv(feature,512,K=1)
        self.bn1_3 = BatchNorm(512)
        self.GConv2_3 = ChebConv(512,512,K=1)
        self.bn2_3 = BatchNorm(512)


        self.GConv1_4 = ChebConv(feature,512,K=1)
        self.bn1_4 = BatchNorm(512)
        self.GConv2_4 = ChebConv(512,512,K=1)
        self.bn2_4 = BatchNorm(512)


        self.feature_layer = nn.Sequential(
            nn.Linear(2048, 512),  # 全连接 120
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),  # relu激活
            nn.Linear(512, out_channel),  # 全连接 84
        )



    def forward(self,data_1,data_2,data_3,data_4):
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
        #x_2 = self.dropout1_2(x_2)
        #x_2 = self.fc2_2(x_2)
        #x_2 = self.dropout2_2(x_2)


        x_3, edge_index_3, edge_weight_3 = data_3.x, data_3.edge_index, data_3.edge_attr
        x_3 = self.GConv1_3(x_3, edge_index_3, edge_weight_3)
        x_3 = self.bn1_3(x_3)
        x_3 = F.relu(x_3)
        edge_index_3, edge_weight_3 = dropout_adj(edge_index_3, edge_weight_3)
        x_3 = self.GConv2_3(x_3, edge_index_3, edge_weight_3)
        x_3 = self.bn2_3(x_3)
        x_3 = F.relu(x_3)
        x_3 = self.fc1_3(x_3)
        #x_3 = self.dropout1_2(x_3)
        #x_3 = self.fc2_3(x_3)
        #x_3 = self.dropout2_3(x_3)

        x_4, edge_index_4, edge_weight_4 = data_4.x, data_4.edge_index, data_4.edge_attr
        x_4 = self.GConv1_4(x_4, edge_index_4, edge_weight_4)
        x_4 = self.bn1_4(x_4)
        x_4 = F.relu(x_4)
        edge_index_4, edge_weight_4 = dropout_adj(edge_index_4, edge_weight_4)
        x_4 = self.GConv2_4(x_4, edge_index_4, edge_weight_4)
        x_4 = self.bn2_4(x_4)
        x_4 = F.relu(x_4)
        x_4 = self.fc1_4(x_4)

        x=torch.cat([x_1,x_2,x_3,x_4],dim=1)
        fc=self.feature_layer(x)

        return fc

