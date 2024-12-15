import torch
import torch.nn as nn
from fastkan import FastKAN as KAN
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa
from torch_geometric.utils import dropout_adj


class MMMChebyNetKAN_method2(torch.nn.Module):
    def __init__(self,feature,out_channel):
        super(MMMChebyNetKAN_method2, self).__init__()

        self.GConv1_1 = ChebConv(feature,512,K=1)
        self.bn1_1 = BatchNorm(512)
        self.GConv2_1 = ChebConv(512,512,K=1)
        self.bn2_1 = BatchNorm(512)



        #self.fc1_8 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(inplace=True)
        self.feature_layer_1 = nn.Sequential(
            nn.Linear(512, 128),  # 全连接 84
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),  # relu激活
            #nn.Linear(128, out_channel),  # 全连接 84
        )
        self.feature_layer_2 = KAN([512, 32, 128])



        self.feature_layer_3 = KAN([256,16,out_channel])





    def forward(self,data_1):
        x_1, edge_index_1, edge_weight_1 = data_1.x, data_1.edge_index, data_1.edge_attr
        x_1 = self.GConv1_1(x_1, edge_index_1, edge_weight_1)
        x_1 = self.bn1_1(x_1)
        x_1 = F.relu(x_1)
        edge_index_1, edge_weight_1 = dropout_adj(edge_index_1, edge_weight_1)
        x_1 = self.GConv2_1(x_1, edge_index_1, edge_weight_1)
        x_1 = self.bn2_1(x_1)
        x_1 = F.relu(x_1)

        #x=torch.unsqueeze(x,1)
        x_2 = self.feature_layer_1(x_1)
        x_3 = self.feature_layer_2(x_1)
        x = torch.cat([x_2, x_3], dim=1)
        x = self.feature_layer_3(x)


        #logits=self.classifier(fc)
        return x

