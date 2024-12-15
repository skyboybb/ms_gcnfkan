import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial.distance import pdist

def cal_sim(data,s1,s2):
    edge_index = [[],[]]
    edge_feature = []
    if s1 != s2:
        v_1 = data[s1]
        v_2 = data[s2]
        combine = np.vstack([v_1, v_2])
        likely = 1- pdist(combine, 'cosine')
        #w = np.exp((-(likely[0]) ** 2) / 30)
        if likely.item() >= 0:
            w = 1
            edge_index[0].append(s1)
            edge_index[1].append(s2)
            edge_feature.append(w)
    return edge_index,edge_feature

def Path_attr(data):

    node_edge = [[], []]

    for i in range(len(data) - 1):
        node_edge[0].append(i)
        node_edge[1].append(i + 1)

    distance = []
    for j in range(len(data) - 1):
        v_1 = data[j]
        v_2 = data[j + 1]
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])

    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))  #Gussion kernel高斯核

    return node_edge, w


def Gen_graph(graphType, data, label,task):
    data_list = []
    if graphType == 'PathGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label
            else:
                print("There is no such task!!")
            node_edge, w = Path_attr(graph_feature)
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    else:
        print("This GraphType is not included!")
    return data_list
