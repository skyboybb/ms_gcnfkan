import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from PathGraph import PathGraph
from AuxFunction import msFFT_plt
from tqdm import tqdm
import pickle
import random
import numpy as np
# -------------------------------------------------------------
signal_size = 2000

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
    data_1, data_2, data_3 = data_load(sample_length, path1,axisname=normalname[0], label=0,InputType=InputType,task=task)  # nThe label for normal data is 0
    for i in tqdm(range(len(dataname9))):
        path2 = os.path.join(data_root2, dataname9[i])
        data1, data2, data3 = data_load(sample_length, path2, dataname9[i],label=label[i], InputType=InputType,task=task)
        data_1 += data1
        data_2 += data2
        data_3 += data3
        #data_4 += data4


    return data_1, data_2, data_3


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

    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    start, end = 0, signal_size
    while end <= fl[:signal_size*1000].shape[0]:
        if InputType == "MMMMSFFT":
            x = fl[start:end]
            x_1,x_2,x_3 = msFFT_plt(x)
            data_1.append(x_1)
            data_2.append(x_2)
            data_3.append(x_3)

        start += signal_size
        end += signal_size
    graphset_1 = PathGraph(8, data_1, label, task)
    graphset_2 = PathGraph(8, data_2, label, task)
    graphset_3 = PathGraph(8, data_3, label, task)


    return graphset_1,graphset_2,graphset_3

def train_test_split(list_data_1,list_data_2,list_data_3,test_size=0.3, random_state=None):
    if random_state:
        random.seed(random_state)

    data_copy_1 = list_data_1[:]  # 创建数据的副本，以免修改原始数据
    data_copy_2 = list_data_2[:]
    data_copy_3 = list_data_3[:]
    #data_copy_4 = list_data_4[:]



    N=len(data_copy_1)
    indices = np.arange(N)
    np.random.shuffle(indices)
    indices=list(indices)
    #random.shuffle(data_copy_1)  # 随机打乱数据
    data_copy_1 = sorted(data_copy_1, key=lambda x: indices.index(data_copy_1.index(x)))
    data_copy_2 = sorted(data_copy_2, key=lambda x: indices.index(data_copy_2.index(x)))
    data_copy_3 = sorted(data_copy_3, key=lambda x: indices.index(data_copy_3.index(x)))
    #data_copy_4 = sorted(data_copy_4, key=lambda x: indices.index(data_copy_4.index(x)))



    split_index = int(len(data_copy_1) * (1 - test_size))

    train_data_1 = data_copy_1[:split_index]
    train_data_2 = data_copy_2[:split_index]
    train_data_3 = data_copy_3[:split_index]
    #train_data_4 = data_copy_4[:split_index]

    test_data_1 = data_copy_1[split_index:]
    test_data_2 = data_copy_2[split_index:]
    test_data_3 = data_copy_3[split_index:]
    #test_data_4 = data_copy_4[split_index:]


    return train_data_1,train_data_2,train_data_3,test_data_1,test_data_2,test_data_3

class CWRUPathMSFFT_sets(object):
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
            list_data_1, list_data_2, list_data_3= get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
            # with open(os.path.join(self.data_dir, "MotorKnn.pkl"), 'wb') as fo:
            # pickle.dump(list_data_1, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:
            train_data_1, train_data_2, train_data_3,test_data_1, test_data_2, test_data_3= train_test_split(list_data_1,list_data_2,\
                                                        list_data_3,test_size=0.30, random_state=40)
            return train_data_1, train_data_2, train_data_3,test_data_1, test_data_2, test_data_3

