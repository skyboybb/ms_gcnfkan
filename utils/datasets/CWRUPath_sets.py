import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from PathGraph import PathGraph
from AuxFunction import wpd_plt
from AuxFunction import add_noise
from tqdm import tqdm
import numpy as np
import random
import pickle
# -------------------------------------------------------------
signal_size = 1024

datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
# For 12k Drive End Bearing Fault Data
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm
dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm
# For 12k Fan End Bearing Fault Data
dataname5 = ["278.mat", "282.mat", "294.mat", "274.mat", "286.mat", "310.mat", "270.mat", "290.mat",
             "315.mat"]  # 1797rpm
dataname6 = ["279.mat", "283.mat", "295.mat", "275.mat", "287.mat", "309.mat", "271.mat", "291.mat",
             "316.mat"]  # 1772rpm
dataname7 = ["280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat",
             "317.mat"]  # 1750rpm
dataname8 = ["281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat",
             "318.mat"]  # 1730rpm
# For 48k Drive End Bearing Fault Data
dataname9 = ["109.mat", "122.mat", "135.mat","189.mat", "201.mat", "213.mat", "226.mat","238.mat"]  # 1797rpm
dataname10 = ["110.mat", "123.mat", "136.mat", "175.mat", "190.mat", "202.mat", "214.mat", "251.mat",
              "263.mat"]  # 1772rpm
dataname11 = ["111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "252.mat",
              "264.mat"]  # 1750rpm
dataname12 = ["112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "253.mat",
              "265.mat"]  # 1730rpm
# label
label = [1, 2, 3, 4, 5, 6, 7, 8]  # The failure data is labeled 1-9
axis = ["_DE_time", "_FE_time", "_BA_time"]


# generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data_root1 = os.path.join(root, datasetname[3])
    data_root2 = os.path.join(root, datasetname[0])

    path1 = os.path.join(data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8 = data_load(sample_length, path1,
                                                                               axisname=normalname[0], label=0,
                                                                               InputType=InputType,
                                                                               task=task)  # nThe label for normal data is 0
    for i in tqdm(range(len(dataname9))):
        path2 = os.path.join(data_root2, dataname9[i])
        data1, data2, data3, data4, data5, data6, data7, data8 = data_load(sample_length, path2, dataname9[i],
                                                                           label=label[i], InputType=InputType,
                                                                           task=task)
        data_1 += data1
        data_2 += data2
        data_3 += data3
        data_4 += data4
        data_5 += data5
        data_6 += data6
        data_7 += data7
        data_8 += data8

    return data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8


def data_load(signal_size, filename, axisname, label, InputType, task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1, )

    data = []
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []
    data_6 = []
    data_7 = []
    data_8 = []
    graphset_1 = 0
    graphset_2 = 0
    graphset_3 = 0
    graphset_4 = 0
    graphset_5 = 0
    graphset_6 = 0
    graphset_7 = 0
    graphset_8 = 0
    start, end = 0, signal_size
    stride = 512
    while end <= fl[:signal_size * 1000].shape[0]:
        if InputType == "TD":
            x = fl[start:end]
            data.append(x)
        elif InputType == "WPD_1":
            x = fl[start:end]
            x = wpd_plt(x, 1)
            data_1.append(x[0])
            data_2.append(x[1])
        elif InputType == "WPD_2":
            x = fl[start:end]
            x = wpd_plt(x, 2)
            data_1.append(x[0])
            data_2.append(x[1])
            data_3.append(x[2])
            data_4.append(x[3])
        elif InputType == "WPD_3":
            x = fl[start:end]
            x = wpd_plt(x, 3)
            data_1.append(x[0])
            data_2.append(x[1])
            data_3.append(x[2])
            data_4.append(x[3])
            data_5.append(x[4])
            data_6.append(x[5])
            data_7.append(x[6])
            data_8.append(x[7])
        else:
            print("The InputType is wrong!!")
        start += stride
        end += stride

    if InputType == "TD":
        graphset_1 = PathGraph(8, data, label, task)
    if InputType == "WPD_1":
        graphset_1 = PathGraph(8, data_1, label, task)
        graphset_2 = PathGraph(8, data_2, label, task)
    if InputType == "WPD_2":
        graphset_1 = PathGraph(8, data_1, label, task)
        graphset_2 = PathGraph(8, data_2, label, task)
        graphset_3 = PathGraph(8, data_3, label, task)
        graphset_4 = PathGraph(8, data_4, label, task)
    if InputType == "WPD_3":
        graphset_1 = PathGraph(8, data_1, label, task)
        graphset_2 = PathGraph(8, data_2, label, task)
        graphset_3 = PathGraph(8, data_3, label, task)
        graphset_4 = PathGraph(8, data_4, label, task)
        graphset_5 = PathGraph(8, data_5, label, task)
        graphset_6 = PathGraph(8, data_6, label, task)
        graphset_7 = PathGraph(8, data_7, label, task)
        graphset_8 = PathGraph(8, data_8, label, task)

    return graphset_1, graphset_2, graphset_3, graphset_4, graphset_5, graphset_6, graphset_7, graphset_8

def train_test_split(list_data_1,list_data_2,list_data_3,list_data_4,list_data_5,list_data_6,list_data_7,list_data_8,test_size=0.3, random_state=None):
    if random_state:
        random.seed(random_state)

    data_copy_1 = list_data_1[:]  # 创建数据的副本，以免修改原始数据
    data_copy_2 = list_data_2[:]
    data_copy_3 = list_data_3[:]
    data_copy_4 = list_data_4[:]

    data_copy_5 = list_data_5[:]
    data_copy_6 = list_data_6[:]
    data_copy_7 = list_data_7[:]
    data_copy_8 = list_data_8[:]

    N=len(data_copy_1)
    indices = np.arange(N)
    np.random.shuffle(indices)
    indices=list(indices)
    #random.shuffle(data_copy_1)  # 随机打乱数据
    data_copy_1 = sorted(data_copy_1, key=lambda x: indices.index(data_copy_1.index(x)))
    data_copy_2 = sorted(data_copy_2, key=lambda x: indices.index(data_copy_2.index(x)))
    data_copy_3 = sorted(data_copy_3, key=lambda x: indices.index(data_copy_3.index(x)))
    data_copy_4 = sorted(data_copy_4, key=lambda x: indices.index(data_copy_4.index(x)))
    data_copy_5 = sorted(data_copy_5, key=lambda x: indices.index(data_copy_5.index(x)))
    data_copy_6 = sorted(data_copy_6, key=lambda x: indices.index(data_copy_6.index(x)))
    data_copy_7 = sorted(data_copy_7, key=lambda x: indices.index(data_copy_7.index(x)))
    data_copy_8 = sorted(data_copy_8, key=lambda x: indices.index(data_copy_8.index(x)))


    split_index = int(len(data_copy_1) * (1 - test_size))

    train_data_1 = data_copy_1[:split_index]
    train_data_2 = data_copy_2[:split_index]
    train_data_3 = data_copy_3[:split_index]
    train_data_4 = data_copy_4[:split_index]
    train_data_5 = data_copy_5[:split_index]
    train_data_6 = data_copy_6[:split_index]
    train_data_7 = data_copy_7[:split_index]
    train_data_8 = data_copy_8[:split_index]
    test_data_1 = data_copy_1[split_index:]
    test_data_2 = data_copy_2[split_index:]
    test_data_3 = data_copy_3[split_index:]
    test_data_4 = data_copy_4[split_index:]
    test_data_5 = data_copy_5[split_index:]
    test_data_6 = data_copy_6[split_index:]
    test_data_7 = data_copy_7[split_index:]
    test_data_8 = data_copy_8[split_index:]

    return train_data_1,train_data_2,train_data_3,train_data_4,train_data_5,train_data_6,train_data_7,\
        train_data_8,test_data_1,test_data_2,test_data_3,test_data_4,test_data_5,test_data_6,test_data_7,\
        test_data_8


class CWRUPath_sets(object):
    num_classes = 9

    def __init__(self, sample_length, data_dir,InputType,task):
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.InputType = InputType
        self.task = task


    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            list_data_1, list_data_2, list_data_3, list_data_4, list_data_5, list_data_6, list_data_7, list_data_8 = get_files(
                self.sample_length, self.data_dir, self.InputType, self.task, test)
            # with open(os.path.join(self.data_dir, "MotorKnn.pkl"), 'wb') as fo:
            # pickle.dump(list_data_1, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:
            train_data_1, train_data_2, train_data_3, train_data_4, train_data_5, train_data_6, train_data_7, \
                train_data_8, test_data_1, test_data_2, test_data_3, test_data_4, test_data_5, test_data_6, test_data_7,\
                test_data_8= train_test_split(list_data_1,list_data_2,list_data_3,
            list_data_4,list_data_5,list_data_6,list_data_7,list_data_8,test_size=0.3, random_state=40)
            return train_data_1, train_data_2, train_data_3, train_data_4, train_data_5, train_data_6, train_data_7, \
                train_data_8, test_data_1, test_data_2, test_data_3, test_data_4, test_data_5, test_data_6, test_data_7,\
                test_data_8

