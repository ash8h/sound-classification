
# coding: utf-8

# In[1]:

import glob
import os
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)  # メル周波数ケプストラム係数（Mel-Frequency Cepstrum Coefficients）
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)  # クロマベクトル(Chroma)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)  # メル周波数ケプストラムの平均値
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)  # スペクトル-コントラクト(Spectral Contrast)の平均
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)  #  和音に関する特徴量(tonal centroid features)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav', verbose=0):
    features, labels = np.empty((0,193)), np.empty(0)    # 193次元の特徴量配列
    for label, sub_dir in enumerate(sub_dirs):
        files = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))
        for fn in files:
            if (verbose):
                print('loading: ' + fn)
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])   # hstack: 配列を横方向に結合
            features = np.vstack([features,ext_features])
            paths = fn.split('/')
            labels = np.append(labels, paths[len(paths)-1].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):  # ラベルをスカラー形式からベクトル形式に変換
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# In[10]:

def load_urbansound8k(parent_dir, sub_dirs, verbose=0):
    file_ext = '*.wav'
    features, labels = parse_audio_files(parent_dir, sub_dirs, file_ext, verbose)
    labels = one_hot_encode(labels)
    return features, labels

def load_urbansound8k_train_and_test(parent_dir = 'UrbanSound8K/audio/',
                                     tr_sub_dirs = ['fold1', 'fold2'], 
                                     ts_sub_dirs = ['fold3'],
                                     verbose = 0):
    tr_features, tr_labels = load_urbansound8k(parent_dir, tr_sub_dirs, verbose)
    ts_features, ts_labels = load_urbansound8k(parent_dir, ts_sub_dirs, verbose)
    return tr_features, tr_labels, ts_features, ts_labels


# In[11]:

def save_to_npy_files(tr_features, tr_labels, ts_features, ts_labels, prefix=''):
    np.save(prefix + 'tr_features.npy', tr_features)
    np.save(prefix + 'tr_labels.npy', tr_labels)
    np.save(prefix + 'ts_features.npy', ts_features)
    np.save(prefix + 'ts_labels.npy', ts_labels)


# In[12]:

def load_from_npy_files(prefix = ''):
    tr_features = np.load(prefix + 'tr_features.npy')
    tr_labels = np.load(prefix + 'tr_labels.npy')
    ts_features = np.load(prefix + 'ts_features.npy')
    ts_labels = np.load(prefix + 'ts_labels.npy')
    return tr_features, tr_labels, ts_features, ts_labels


# In[9]:

if __name__ == '__main__':
    parent_dir = 'UrbanSound8K/audio/'
    tr_subdirs = ['fold1','fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7']
    ts_subdirs =['fold8', 'fold9', 'fold10']
    
    print('loading urbansound8k dataset.')    
    get_ipython().magic('time tr_features, tr_labels, ts_features, ts_labels = load_urbansound8k_train_and_test(parent_dir,  tr_subdirs, ts_subdirs, 1)')
    save_to_npy_files(tr_features, tr_labels, ts_features, ts_labels, prefix='all_')


# In[ ]:



