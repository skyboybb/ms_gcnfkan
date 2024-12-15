import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self,out_channel):
        super(LSTMModel, self).__init__()
        self.hidden_size = 16  # LSTM 单元的隐藏状态维度
        self.num_layers = 2  # LSTM 层数

        # 定义 LSTM 层
        self.lstm = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True,dropout=0.2)

        # 定义全连接层
        self.fc = nn.Sequential( nn.Flatten(),
            nn.Linear(16*512,256),
            nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256,out_channel))

    def forward(self, x):
        # 初始化 LSTM 的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = torch.unsqueeze(x, dim=2)

        # 前向传播通过 LSTM 层
        out, _ = self.lstm(x, (h0, c0))
        shape_1=out.shape[1]
        shape_2 = out.shape[2]
        out = out.reshape(-1,shape_1*shape_2)


        out = self.fc(out)
        probas = F.softmax(out, dim=1)  # softmax分类器

        return out,probas






