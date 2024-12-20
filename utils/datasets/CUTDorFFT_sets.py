import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from PathGraph import PathGraph
from AuxFunction import FFT_plt
from tqdm import tqdm
import numpy as np
import random
import pickle
import torch
# -------------------------------------------------------------

signal_size = 1024
datasetname = ["Geardata","Normal Baseline Data"]
normalname = ["NO_data.mat"]
# For 12k Drive End Bearing Fault Data
dataname1 = ["CT_1_data.mat", "CT_2_data.mat", "CT_3_data.mat", "CT_4_data.mat", "CT_5_data.mat", "MT_data.mat","RC_data.mat","ST_data.mat"]
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
    data_1, label_0 = data_load_1(sample_length, path1,axisname=normalname[0],InputType=InputType,task=task)  # nThe label for normal data is 0
    for i in tqdm(range(len(dataname1))):
        path2 = os.path.join(data_root2, dataname1[i])
        data1,label_all = data_load(sample_length, path2, dataname1[i],label=label[i], InputType=InputType,task=task)
        data_1 += data1
        label_0 += label_all


    return data_1,label_0
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
    fl = fl.reshape(-1, )
    data_1 = []
    start, end = 0, signal_size
    while end <= fl[:signal_size*1000].shape[0]:
        label.append(0)
        if InputType == "TD":
            x = fl[start:end]
            data_1.append(x)
        elif InputType == "FFT":
            x = fl[start:end]
            x=FFT_plt(x)
            data_1.append(x)
        start += signal_size
        end += signal_size

    return data_1,label

def data_load(signal_size, filename, axisname, label, InputType, task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    realaxis = datanumber[0]
    fl = loadmat(filename)[realaxis]
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1, )

    data_1 = []
    label_all = []
    start, end = 0, signal_size
    while end <= fl[:signal_size * 1000].shape[0]:
        label_all.append(label)
        if InputType == "TD":
            x = fl[start:end]
            data_1.append(x)
        elif InputType == "FFT":
            x = fl[start:end]
            x=FFT_plt(x)
            data_1.append(x)

        start += signal_size
        end += signal_size

    return data_1,label_all

def train_test_split(list_data_1,list_label,test_size=0.3, random_state=None):
    if random_state:
        random.seed(random_state)

    data_copy_1 = np.array(list_data_1[:])  # 创建数据的副本，以免修改原始数据




    data_label = np.array(list_label[:])

    N=len(data_copy_1)
    indices = np.arange(N)
    np.random.shuffle(indices)
    #indices=list(indices)
    #random.shuffle(data_copy_1)  # 随机打乱数据
    data_copy_1 = data_copy_1[indices]


    copy_data_label = data_label[indices]
    data_copy_1 = torch.tensor(data_copy_1).type(torch.FloatTensor)

    copy_data_label = torch.tensor(copy_data_label).type(torch.LongTensor)


    split_index = int(len(data_copy_1) * (1 - test_size))

    train_data_1 = data_copy_1[:split_index]


    train_data_label = copy_data_label[:split_index]
    test_data_1 = data_copy_1[split_index:]


    test_data_label = copy_data_label[split_index:]


    return train_data_1,test_data_1,train_data_label,test_data_label


class CUTDorFFT_sets(object):
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
            list_data_1, list_label = get_files(
                self.sample_length, self.data_dir, self.InputType, self.task, test)
            # with open(os.path.join(self.data_dir, "MotorKnn.pkl"), 'wb') as fo:
            # pickle.dump(list_data_1, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:
            train_data_1,  test_data_1, train_label,test_label= train_test_split(list_data_1,list_label,test_size=0.30, random_state=40)
            return train_data_1, test_data_1, train_label,test_label

