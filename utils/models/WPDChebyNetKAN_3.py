import torch
import torch.nn as nn
from fastkan import FastKAN as KAN
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
from torch_geometric.utils import dropout_adj


class ChebyNetKAN_3(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(ChebyNetKAN_3, self).__init__()

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


        self.GConv1_5 = ChebConv(feature,512,K=1)
        self.bn1_5 = BatchNorm(512)
        self.GConv2_5 = ChebConv(512,512,K=1)
        self.bn2_5 = BatchNorm(512)


        self.GConv1_6 = ChebConv(feature,512,K=1)
        self.bn1_6 = BatchNorm(512)
        self.GConv2_6 = ChebConv(512,512,K=1)
        self.bn2_6 = BatchNorm(512)


        self.GConv1_7 = ChebConv(feature,512,K=1)
        self.bn1_7 = BatchNorm(512)
        self.GConv2_7 = ChebConv(512,512,K=1)
        self.bn2_7 = BatchNorm(512)
        #self.fc1_7 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(inplace=True))
        #self.dropout1_7 = nn.Dropout(0.5)


        self.GConv1_8 = ChebConv(feature,512,K=1)
        self.bn1_8 = BatchNorm(512)
        self.GConv2_8 = ChebConv(512,512,K=1)
        self.bn2_8 = BatchNorm(512)
        #self.fc1_8 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(inplace=True)
        self.feature_layer_1 = nn.Sequential(
            nn.Linear(4096, 1024),  # 全连接 84
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),  # relu激活
        )
        self.feature_layer_2 = KAN([1024,32,out_channel])





    def forward(self,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8):
        x_1, edge_index_1, edge_weight_1 = data_1.x, data_1.edge_index, data_1.edge_attr
        x_1 = self.GConv1_1(x_1, edge_index_1, edge_weight_1)
        x_1 = self.bn1_1(x_1)
        x_1 = F.relu(x_1)
        edge_index_1, edge_weight_1 = dropout_adj(edge_index_1, edge_weight_1)
        x_1 = self.GConv2_1(x_1, edge_index_1, edge_weight_1)
        x_1 = self.bn2_1(x_1)
        x_1 = F.relu(x_1)


        x_2, edge_index_2, edge_weight_2 = data_2.x, data_2.edge_index, data_2.edge_attr
        x_2 = self.GConv1_2(x_2, edge_index_2, edge_weight_2)
        x_2 = self.bn1_2(x_2)
        x_2 = F.relu(x_2)
        edge_index_2, edge_weight_2 = dropout_adj(edge_index_2, edge_weight_2)
        x_2 = self.GConv2_2(x_2, edge_index_2, edge_weight_2)
        x_2 = self.bn2_2(x_2)
        x_2 = F.relu(x_2)



        x_3, edge_index_3, edge_weight_3 = data_3.x, data_3.edge_index, data_3.edge_attr
        x_3 = self.GConv1_3(x_3, edge_index_3, edge_weight_3)
        x_3 = self.bn1_3(x_3)
        x_3 = F.relu(x_3)
        edge_index_3, edge_weight_3 = dropout_adj(edge_index_3, edge_weight_3)
        x_3 = self.GConv2_3(x_3, edge_index_3, edge_weight_3)
        x_3 = self.bn2_3(x_3)
        x_3 = F.relu(x_3)


        x_4, edge_index_4, edge_weight_4 = data_4.x, data_4.edge_index, data_4.edge_attr
        x_4 = self.GConv1_4(x_4, edge_index_4, edge_weight_4)
        x_4 = self.bn1_4(x_4)
        x_4 = F.relu(x_4)
        edge_index_4, edge_weight_4 = dropout_adj(edge_index_4, edge_weight_4)
        x_4 = self.GConv2_4(x_4, edge_index_4, edge_weight_4)
        x_4 = self.bn2_4(x_4)
        x_4 = F.relu(x_4)

        x_5, edge_index_5, edge_weight_5 = data_5.x, data_5.edge_index, data_5.edge_attr
        x_5 = self.GConv1_5(x_5, edge_index_5, edge_weight_5)
        x_5 = self.bn1_5(x_5)
        x_5 = F.relu(x_5)
        edge_index_5, edge_weight_5 = dropout_adj(edge_index_5, edge_weight_5)
        x_5 = self.GConv2_5(x_5, edge_index_5, edge_weight_5)
        x_5 = self.bn2_5(x_5)
        x_5 = F.relu(x_5)


        x_6, edge_index_6, edge_weight_6 = data_6.x, data_6.edge_index, data_6.edge_attr
        x_6 = self.GConv1_6(x_6, edge_index_6, edge_weight_6)
        x_6 = self.bn1_6(x_6)
        x_6 = F.relu(x_6)
        edge_index_6, edge_weight_6 = dropout_adj(edge_index_6, edge_weight_6)
        x_6 = self.GConv2_6(x_6, edge_index_6, edge_weight_6)
        x_6 = self.bn2_6(x_6)
        x_6 = F.relu(x_6)


        x_7, edge_index_7, edge_weight_7 = data_7.x, data_7.edge_index, data_7.edge_attr
        x_7 = self.GConv1_7(x_7, edge_index_7, edge_weight_7)
        x_7 = self.bn1_7(x_7)
        x_7 = F.relu(x_7)
        edge_index_7, edge_weight_7 = dropout_adj(edge_index_7, edge_weight_7)
        x_7 = self.GConv2_7(x_7, edge_index_7, edge_weight_7)
        x_7 = self.bn2_7(x_7)
        x_7 = F.relu(x_7)


        x_8, edge_index_8, edge_weight_8 = data_8.x, data_8.edge_index, data_8.edge_attr
        x_8 = self.GConv1_8(x_8, edge_index_8, edge_weight_8)
        x_8 = self.bn1_8(x_8)
        x_8 = F.relu(x_8)
        edge_index_8, edge_weight_8 = dropout_adj(edge_index_8, edge_weight_8)
        x_8 = self.GConv2_8(x_8, edge_index_8, edge_weight_8)
        x_8 = self.bn2_8(x_8)
        x_8 = F.relu(x_8)

        x=torch.cat([x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8],dim=1)
        #x=torch.unsqueeze(x,1)
        x=self.feature_layer_1(x)
        x = self.feature_layer_2(x)


        #fc=self.feature_layer(x)

        #logits=self.classifier(fc)
        return x

