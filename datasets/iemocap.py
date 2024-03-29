import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets.data_utils import extract_log_mel_spectrogram, extract_window, MelSpectrogramLibrosa, signal_to_frame
from datasets.data_utils import DataUtils
import torch.nn.functional as f

class IEMOCAPTrain(Dataset):
    def __init__(self,sample_rate=16000):        
        self.feat_root =  "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/iemocap/iemocap/IEMOCAP/"
        annotations_file=os.path.join(self.feat_root,"train_data.csv")
        self.uttr_labels= pd.read_csv(annotations_file)
        self.sample_rate = sample_rate
        self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3} 
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        wave_normalised = f.normalize(wave_audio,dim=-1,p=2)
        wave_random1sec = extract_window(wave_normalised)
        uttr_melspec = extract_log_mel_spectrogram(wave_random1sec, self.to_mel_spec)
        label = row['Label']
        return uttr_melspec, self.labels_dict[label]


class IEMOCAPTest(Dataset):
    def __init__(self,sample_rate=16000):        
        self.feat_root =  "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/iemocap/iemocap/IEMOCAP/"
        annotations_file=os.path.join(self.feat_root,"test_data.csv")
        self.uttr_labels= pd.read_csv(annotations_file)
        self.sample_rate = sample_rate
        self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3} 
        self.no_of_classes= len(self.labels_dict)
        self._n_frames = 96 # * Taken from cola implementation equivalent to 960 milliseconds
        self.to_mel_spec = MelSpectrogramLibrosa()

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        wave_audio_chopped = signal_to_frame(wave_audio,frame_length=self._n_frames * 160,
                                    frame_step=self._n_frames * 160,pad_end=True)

        # wave_audio_chopped = tf.signal.frame(
        #                             wave_audio,frame_length=self._n_frames * 160,
        #                             frame_step=self._n_frames * 160,pad_end=True)

        #wave_audio_normalised = tf.math.l2_normalize(wave_audio_chopped, axis=-1, epsilon=1e-9)

        extracted_logmel =[]
        for i in np.arange(wave_audio_chopped.shape[0]):
            wave_audio_normalised = f.normalize(wave_audio_chopped[i],dim=-1,p=2)
            extracted_logmel.append(extract_log_mel_spectrogram(wave_audio_normalised, self.to_mel_spec))
        label = row['Label']
        return torch.stack(extracted_logmel).unsqueeze(dim=1), self.labels_dict[label]

