import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets.data_utils import extract_log_mel_spectrogram, extract_window, MelSpectrogramLibrosa, signal_to_frame, get_avg_duration
from datasets.data_utils import DataUtils
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
duration = 8
print(duration,'duration')
class Voxceleb1DatasetTrain(Dataset):
    def __init__(self,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/voxceleb/"
        self.uttr_labels= pd.read_csv(self.feat_root+"train_vox.csv")
        self.sample_rate = sample_rate
        self.no_of_classes= 1251
        self.to_mel_spec = MelSpectrogramLibrosa()

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['file_path'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        #wave_normalised = f.normalize(wave_audio,dim=-1,p=2)
        #wave_random1sec = extract_window(wave_normalised,data_size=duration)
        wave_audio = extract_window(wave_audio,data_size=duration)
        wave_random1sec=f.normalize(wave_audio,dim=-1,p=2)
        uttr_melspec = extract_log_mel_spectrogram(wave_random1sec, self.to_mel_spec)
        label = row['label']
        return uttr_melspec, label

class Voxceleb1DatasetTest(Dataset):
    def __init__(self,sample_rate=16000):        
        self.feat_root = "/nlsasfs/home/nltm-pilot/ashishs/voxceleb/"
        self.uttr_labels= pd.read_csv(self.feat_root+"test_vox.csv")
        self.sample_rate = sample_rate
        self.no_of_classes = 1251
        self.to_mel_spec = MelSpectrogramLibrosa()

    def __len__(self):
        return len(self.uttr_labels)
    
    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['file_path'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        #wave_normalised = f.normalize(wave_audio,dim=-1,p=2)
        #wave_random1sec = extract_window(wave_normalised,data_size=duration)
        wave_audio = extract_window(wave_audio,data_size=duration)
        wave_random1sec=f.normalize(wave_audio,dim=-1,p=2)

        uttr_melspec = extract_log_mel_spectrogram(wave_random1sec, self.to_mel_spec)
        label = row['label']
        return uttr_melspec, label