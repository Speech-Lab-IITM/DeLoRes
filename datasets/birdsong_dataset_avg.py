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
#random sample is taken from the whole audio frame
complete_data = pd.read_csv("/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/birdsong/combined_data.csv")
duration = get_avg_duration(complete_data,"/nlsasfs/home/nltm-pilot/ashishs/Bird_audio/")
print(duration,'duration')
train, test = train_test_split(complete_data, test_size=0.2, random_state=1, stratify=complete_data['Label'])

class BirdSongDatasetTrain(Dataset):
    def __init__(self,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/Bird_audio/"
        self.uttr_labels= train
        self.sample_rate = sample_rate
        self.no_of_classes = 2
        self.to_mel_spec = MelSpectrogramLibrosa()

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        wave_normalised = f.normalize(wave_audio,dim=-1,p=2)
        wave_random1sec = extract_window(wave_normalised,data_size=duration)
        uttr_melspec = extract_log_mel_spectrogram(wave_random1sec, self.to_mel_spec)
        label = row['Label']
        return uttr_melspec, label

class BirdSongDatasetTest(Dataset):
    def __init__(self,sample_rate=16000):        
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/Bird_audio/"
        self.uttr_labels= test
        self.sample_rate = sample_rate
        self.no_of_classes = 2
        self.to_mel_spec = MelSpectrogramLibrosa()

    def __len__(self):
        return len(self.uttr_labels)
    
    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        wave_normalised = f.normalize(wave_audio,dim=-1,p=2)
        wave_random1sec = extract_window(wave_normalised,data_size=duration)
        uttr_melspec = extract_log_mel_spectrogram(wave_random1sec, self.to_mel_spec)
        label = row['Label']
        return uttr_melspec, label
        