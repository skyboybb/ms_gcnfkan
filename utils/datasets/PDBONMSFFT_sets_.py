import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from AuxFunction import msFFT_plt
from tqdm import tqdm
import pickle
import random
import numpy as np
import torch
# -------------------------------------------------------------
signal_size = 2000
datasetname = ["Bearing fault","Normal Baseline Data"]
normalname = ["Normal.mat"]
dataname1 = ["OR_1_EDM.mat", "OR_1_ee.mat", "OR_2_ee.mat", "OR_1_dr.mat", "OR_2_dr.mat", "IR_1_EDM.mat","IR_1_ee","IR_2_ee"]  # 1500rpm

# label
label = [1, 2, 3, 4, 5, 6, 7,8]  # The failure data is labeled 1-9
axis = ["_data"]


# generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data_root1 = os.path.join(root, datasetname[1])
    data_root2 = os.path.join(root, datasetname[0])

    path1 = os.path.join(data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    data_1, data_2, data_3,label_0= data_load_1(sample_length, path1,axisname=normalname[0],InputType=InputType,
                                                                               task=task)  # nThe label for normal data is 0
    for i in tqdm(range(len(dataname1))):
        path2 = os.path.join(data_root2, dataname1[i])
        data1, data2, data3, label_all = data_load_2(sample_length, path2, dataname1[i],
                                                                           label=label[i], InputType=InputType,
                                                                           task=task)
        data_1 += data1
        data_2 += data2
        data_3 += data3
        label_0 += label_all

    return data_1, data_2, data_3, label_0

def data_load_1(signal_size, filename, axisname, InputType, task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    label=[]
    datanumber = axisname.split(".")
    realaxis = datanumber[0]
    fl = loadmat(filename)[realaxis]
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1,)
    data_1 = []
    data_2 = []
    data_3 = []
    start, end = 0, signal_size
    while end <= fl[:signal_size*1000].shape[0]:
        label.append(0)
        if InputType == "MS_1":
            x = fl[start:end]
            x = msFFT_plt(x,1)
            data_1.append(x[0])
        elif InputType == "MS_2":
            x = fl[start:end]
            x= msFFT_plt(x, 2)
            data_1.append(x[0])
            data_2.append(x[1])
        elif InputType == "MS_3":
            x = fl[start:end]
            x = msFFT_plt(x, 3)
            data_1.append(x[0])
            data_2.append(x[1])
            data_3.append(x[2])

        start += signal_size
        end += signal_size

    return data_1,data_2,data_3,label

def data_load_2(signal_size, filename, axisname, label, InputType, task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    realaxis = datanumber[0]
    fl = loadmat(filename)[realaxis]
    fl = (fl - fl.min()) / (fl.max() - fl.min())

    fl = fl.reshape(-1,)
    data_1 = []
    data_2 = []
    data_3 = []
    label_all=[]
    start, end = 0, signal_size
    while end <= fl[:signal_size*1000].shape[0]:
        label_all.append(label)
        if InputType == "MS_1":
            x = fl[start:end]
            x = msFFT_plt(x,1)
            data_1.append(x[0])
        elif InputType == "MS_2":
            x = fl[start:end]
            x= msFFT_plt(x, 2)
            data_1.append(x[0])
            data_2.append(x[1])
        elif InputType == "MS_3":
            x = fl[start:end]
            x_1,x_2,x_3 = msFFT_plt(x, 3)
            data_1.append(x_1)
            data_2.append(x_2)
            data_3.append(x_3)
        start += signal_size
        end += signal_size

    return data_1,data_2,data_3,label_all

def train_test_split(list_data_1,list_data_2,list_data_3,list_label,test_size=0.3, random_state=None):
    if random_state:
        random.seed(random_state)

    data_copy_1 = np.array(list_data_1[:])  # 创建数据的副本，以免修改原始数据
    data_copy_2 = np.array(list_data_2[:])
    data_copy_3 = np.array(list_data_3[:])

    data_label = np.array(list_label[:])

    N=len(data_copy_1)
    indices = np.arange(N)
    np.random.shuffle(indices)
    #indices=list(indices)
    #random.shuffle(data_copy_1)  # 随机打乱数据
    data_copy_1 = data_copy_1[indices]
    data_copy_2 = data_copy_2[indices]
    data_copy_3 = data_copy_3[indices]

    copy_data_label = data_label[indices]
    data_copy_1 = torch.tensor(data_copy_1).type(torch.FloatTensor)
    data_copy_2 = torch.tensor(data_copy_2).type(torch.FloatTensor)
    data_copy_3 = torch.tensor(data_copy_3).type(torch.FloatTensor)

    copy_data_label = torch.tensor(copy_data_label).type(torch.LongTensor)


    split_index = int(len(data_copy_1) * (1 - test_size))

    train_data_1 = data_copy_1[:split_index]
    train_data_2 = data_copy_2[:split_index]
    train_data_3 = data_copy_3[:split_index]

    train_data_label = copy_data_label[:split_index]
    test_data_1 = data_copy_1[split_index:]
    test_data_2 = data_copy_2[split_index:]
    test_data_3 = data_copy_3[split_index:]

    test_data_label = copy_data_label[split_index:]


    return train_data_1,train_data_2,train_data_3,test_data_1,test_data_2,test_data_3,train_data_label,test_data_label

class PDBONMSFFT_sets(object):
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
            list_data_1, list_data_2, list_data_3,list_label = get_files(
                self.sample_length, self.data_dir, self.InputType, self.task, test)
            # with open(os.path.join(self.data_dir, "MotorKnn.pkl"), 'wb') as fo:
            # pickle.dump(list_data_1, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:
            train_data_1, train_data_2, train_data_3,test_data_1, test_data_2, test_data_3,train_label,test_label= train_test_split(list_data_1,list_data_2,list_data_3,list_label,test_size=0.20, random_state=40)
            return train_data_1, train_data_2, train_data_3,test_data_1, test_data_2, test_data_3,train_label,test_label

