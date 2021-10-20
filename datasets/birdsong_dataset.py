import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets.data_utils import extract_log_mel_spectrogram, extract_window, MelSpectrogramLibrosa
from datasets.data_utils import DataUtils
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
#random sample is taken from the whole audio frame
complete_data = pd.read_csv("/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/birdsong/combined_data.csv")
train, test = train_test_split(complete_data,test_size=0.2)
class BirdSongDatasetTrain(Dataset):
    def __init__(self,sample_rate=16000):
        #self.feat_root =  "/speech/Databases/Birdsong/IEMOCAP/"
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/Bird_audio/"
        #annotations_file=os.path.join(self.feat_root,"train_data.csv")
        self.uttr_labels= train
        self.sample_rate = sample_rate
        #self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3}
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
        wave_random1sec = extract_window(wave_normalised)
        uttr_melspec = extract_log_mel_spectrogram(wave_random1sec, self.to_mel_spec)
        label = row['Label']
        return uttr_melspec, label

#audio is divided into chunks of 1sec and then tested
def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            #print(pad_axis.shape)
            #print(type(signal))
            signal = f.pad(signal, (0,pad_axis[0]), "constant", pad_value)
    frames=signal.unfold(axis, frame_length, frame_step)
    return frames
class BirdSongDatasetTest(Dataset):
    def __init__(self,sample_rate=16000):
        #self.feat_root =  "/speech/Databases/Birdsong/IEMOCAP/"
        self.feat_root = "/nlsasfs/home/nltm-pilot/ashishs/Bird_audio/"
        #annotations_file=os.path.join(self.feat_root,"test_data.csv")
        self.uttr_labels= test
        self.sample_rate = sample_rate
        #self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3}
        self.no_of_classes= 2
        self._n_frames = 98 # * Taken from cola implementation equivalent to 980 milliseconds
        self.to_mel_spec = MelSpectrogramLibrosa()

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        #wave_audio_chopped = tf.signal.frame(
        #                            wave_audio,frame_length=self._n_frames * 160,
        #                            frame_step=self._n_frames * 160,pad_end=True)
        wave_audio_chopped = frame(wave_audio,frame_length=self._n_frames * 160,
                                    frame_step=self._n_frames * 160,pad_end=True)
        #wave_audio_normalised = tf.math.l2_normalize(wave_audio_chopped, axis=-1, epsilon=1e-9)
        #wave_audio_normalised = f.normalize(wave_audio_chopped,dim=-1,p=2)
        extracted_logmel =[]
        for i in np.arange(wave_audio_chopped.shape[0]):
            #print('check-1')
            wave_audio_normalised = f.normalize(wave_audio_chopped[i],dim=-1,p=2)
            #print('check-2')
            extracted_logmel.append(extract_log_mel_spectrogram(wave_audio_normalised, self.to_mel_spec))
        label = row['Label']
        return torch.stack(extracted_logmel).unsqueeze(dim=1), label
        #wave_audio = torch.tensor(wave_audio)
        #wave_normalised = f.normalize(wave_audio,dim=-1,p=2)
        #wave_random1sec = extract_window(wave_normalised)
        #uttr_melspec = extract_log_mel_spectrogram(wave_random1sec, self.to_mel_spec)
        #label = row['Label']
        #return uttr_melspec, self.labels_dict[label]
