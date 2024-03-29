import torch
import numpy as np
import torch.utils.data as data
import tensorflow as tf
from torch.utils.data import Dataset
import librosa
import torch.nn.functional as f

from utils import extract_log_mel_spectrogram, extract_window, extract_log_mel_spectrogram_torch, extract_window_torch, MelSpectrogramLibrosa


tf.config.set_visible_devices([], 'GPU')
AUDIO_SR = 16000


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## padd
    batch_1 = [torch.Tensor(t) for t,_ in batch]
    batch_1 = torch.nn.utils.rnn.pad_sequence(batch_1,batch_first = True)
    #batch = batch.reshape()
    batch_1 = batch_1.unsqueeze(1)

    batch_2 = [torch.Tensor(t) for _,t in batch]
    batch_2 = torch.nn.utils.rnn.pad_sequence(batch_2,batch_first = True)
    #batch = batch.reshape()
    batch_2 = batch_2.unsqueeze(1)

    return batch_1, batch_2


class BARLOW(Dataset):

    def __init__(self, args, data_dir_list):
        self.audio_files_list = data_dir_list
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.length = args.length_wave

    def __getitem__(self, idx):
        audio_file = self.audio_files_list[idx]
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        wave = torch.tensor(wave)
        x = f.normalize(wave,dim=-1,p=2)

        waveform_a = extract_window_torch(self.length,x)
        log_mel_spec_a = extract_log_mel_spectrogram_torch(waveform_a, self.to_mel_spec)

        waveform_b = extract_window_torch(self.length,x)
        log_mel_spec_b = extract_log_mel_spectrogram_torch(waveform_b, self.to_mel_spec)

        return log_mel_spec_a , log_mel_spec_b

    def __len__(self):
        return len(self.audio_files_list)


# class BARLOW(Dataset):

#     def __init__(self, data_dir_list):
#         self.audio_files_list = data_dir_list

#     def __getitem__(self, idx):
#         audio_file = self.audio_files_list[idx]
#         wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
#         x = tf.math.l2_normalize(wave, epsilon=1e-9)

#         waveform_a = extract_window(x)
#         log_mel_spec_a = extract_log_mel_spectrogram(waveform_a)

#         waveform_b = extract_window(x)
#         log_mel_spec_b = extract_log_mel_spectrogram(waveform_b)

#         return log_mel_spec_a.numpy() , log_mel_spec_b.numpy()

#     def __len__(self):
#         return len(self.audio_files_list)