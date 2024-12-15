import numpy as np
import torch
import numpy as np
import scipy.signal as signal
import librosa
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import pywt
import pywt.data
def add_noise(x, snr):
    '''
    :param x: the raw siganl
    :param snr: the signal to noise ratio
    :return: noise signal
    '''
    d = np.random.randn(len(x))  # generate random noise
    P_signal = np.sum(abs(x) ** 2)
    P_d = np.sum(abs(d) ** 2)
    P_noise = P_signal / 10 ** (snr / 10)
    noise = np.sqrt(P_noise / P_d) * d
    noise_signal = x.reshape(-1) + noise
    return noise_signal

def wpd_plt(signal, n):
    wpd_data=[]
    if n==1:
        tree = pywt.WaveletPacket(data=signal, wavelet='dmey', mode='symmetric')
        node_paths = [node.path for node in tree.get_level(n, 'freq')[:8]]
        for node_path in node_paths:
            # 获取节点数据
            reconstructed_tree = pywt.WaveletPacket(data=None, wavelet='dmey', mode='symmetric')
            reconstructed_tree[node_path]=tree[node_path].data
            wpd_data.append(reconstructed_tree.reconstruct(update=True))
    if n==2:
        tree = pywt.WaveletPacket(data=signal, wavelet='dmey', mode='symmetric')
        node_paths = [node.path for node in tree.get_level(n, 'freq')[:8]]
        for node_path in node_paths:
            # 获取节点数据
            reconstructed_tree = pywt.WaveletPacket(data=None, wavelet='dmey', mode='symmetric')
            reconstructed_tree[node_path]=tree[node_path].data
            wpd_data.append(reconstructed_tree.reconstruct(update=True))
    if n==3:
        tree = pywt.WaveletPacket(data=signal, wavelet='dmey', mode='symmetric')
        node_paths = [node.path for node in tree.get_level(n, 'freq')[:8]]
        wpd_data=[]
        for node_path in node_paths:
            # 获取节点数据
            reconstructed_tree = pywt.WaveletPacket(data=None, wavelet='dmey', mode='symmetric')
            reconstructed_tree[node_path]=tree[node_path].data
            wpd_re_data=reconstructed_tree.reconstruct(update=True)
            a = wpd_re_data[0:1024]
            wpd_data.append(a)
    return wpd_data

def ms_plt(signal, n):
    ms_data=signal
    new_signal_1 = [(signal[i] + signal[i + 1])/2 for i in range(0, len(signal) - 1, 2)]
    new_signal_2 = [(signal[i] + signal[i + 1] + signal[i + 2])/3 for i in range(0, len(signal) - 2, 3)]

    return ms_data,new_signal_1,new_signal_2
def msFFT_plt(signal):
    window_size = 3#length
    numGroup = int(np.floor(len(signal)/window_size))
    max_values = np.zeros(numGroup)
    min_values = np.zeros(numGroup)
    mean_values =np.zeros(numGroup)

    if window_size >1 :
        for i in range(0, numGroup):
            startIdx = i * window_size
            endIdx = (i+1) * window_size
            max_values[i] = max(signal[startIdx:endIdx])
            min_values[i] = min(signal[startIdx:endIdx])
            mean_values[i] = np.mean(signal[startIdx:endIdx])
    else:
        mean_values = signal
        min_values = signal
        max_values = signal

    FFT_values = signal
    x = np.fft.fft(max_values)
    x = np.abs(x) / len(x)
    x_max = x[range(int(x.shape[0] / 2))]

    x_min = np.fft.fft(min_values)
    x_min = np.abs(x_min) / len(x_min)
    x_min = x_min[range(int(x_min.shape[0] / 2))]

    x_mean = np.fft.fft(mean_values)
    x_mean = np.abs(x_mean) / len(x_mean)
    x_mean = x_mean[range(int(x_mean.shape[0] / 2))]

    '''x_FFT = np.fft.fft(FFT_values)
    x_FFT = np.abs(x_FFT) / len(x_FFT)
    x_FFT = x_FFT[range(int(x_FFT.shape[0] / 2))]'''

    return x_max,x_min,x_mean

def FFT_plt(signal):

    x = np.fft.fft(signal)
    x = np.abs(x) / len(x)
    x  = x[range(int(x.shape[0] / 2))]

    '''x_FFT = np.fft.fft(FFT_values)
    x_FFT = np.abs(x_FFT) / len(x_FFT)
    x_FFT = x_FFT[range(int(x_FFT.shape[0] / 2))]'''

    return x



