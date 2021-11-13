import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchaudio
from torch.utils.data import Dataset
from data_utils import extract_log_mel_spectrogram, extract_window
from datasets.data_utils import DataUtils

class IEMOCAPTrain(Dataset):
    def __init__(self,sample_rate=16000):        
        self.feat_root =  DataUtils.root_dir["IEMOCAP"]
        annotations_file=os.path.join(self.feat_root,"train_data.csv")
        self.uttr_labels= pd.read_csv(annotations_file)
        self.sample_rate = sample_rate
        self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3} 
        self.no_of_classes= len(self.labels_dict)

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['Path'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_normalised = tf.math.l2_normalize(wave_audio, epsilon=1e-9)
        wave_random1sec = extract_window(wave_normalised)
        uttr_melspec = extract_log_mel_spectrogram(wave_random1sec).numpy()
        label = row['Label']
        return uttr_melspec, self.labels_dict[label]


class IEMOCAPTest(Dataset):
    def __init__(self,sample_rate=16000):        
        self.feat_root =  DataUtils.root_dir["IEMOCAP"]
        annotations_file=os.path.join(self.feat_root,"test_data.csv")
        self.uttr_labels= pd.read_csv(annotations_file)
        self.sample_rate = sample_rate
        self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3} 
        self.no_of_classes= len(self.labels_dict)

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        raise NotImplementedError

