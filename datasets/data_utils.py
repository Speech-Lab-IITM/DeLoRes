import torch
import torchaudio
import librosa
import tensorflow as tf
import numpy as np

def extract_window(wav, seg_length=16000):
    """Extract random window of 1 second"""
    unit_length = int(0.96 * 16000)
    length_adj = unit_length - len(wav)
    if length_adj > 0:
        half_adj = length_adj // 2
        wav = F.pad(wav, (half_adj, length_adj - half_adj))

    # random crop unit length wave
    length_adj = unit_length - len(wav)
    start = random.randint(0, length_adj) if length_adj > 0 else 0
    wav = wav[start:start + unit_length]

    return wav


class MelSpectrogramLibrosa:
    """Mel spectrogram using librosa."""
    def __init__(self, fs=16000, n_fft=1024, shift=160, n_mels=64, fmin=60, fmax=7800):
        self.fs, self.n_fft, self.shift, self.n_mels, self.fmin, self.fmax = fs, n_fft, shift, n_mels, fmin, fmax
        self.mfb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __call__(self, audio):
        X = librosa.stft(np.array(audio), n_fft=self.n_fft, hop_length=self.shift)
        return torch.tensor(np.matmul(self.mfb, np.abs(X)**2 + np.finfo(float).eps))

def extract_log_mel_spectrogram(waveform, to_mel_spec):
    """Mel spectrogram using librosa.
    waveform: torch tenspr waveform
    to_mel_spec: object of MelSpectrogramLibrosa class"""

    log_mel_spectrograms = (to_mel_spec(waveform) + torch.finfo().eps).log()
    return log_mel_spectrograms


class DataUtils():

    root_dir ={
        "Birdsong" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/birdsong",
        "IEMOCAP" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/iemocap/iemocap/IEMOCAP",
        "Libri100" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/libri100/",
        "MusicalInstruments" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/magenta",
        "tut_urban" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/utu/TUT-urban-acoustic-scenes-2018-development",
        "voxceleb_v1" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/voxceleb/",
        "language_identification" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/lid"
    }

    @classmethod
    def map_labels(cls,label_array):
        uarray = np.unique(label_array)
        label_dict = dict()
        for i,label in enumerate(uarray):
            label_dict[label] = i
        return label_dict

    @classmethod
    def collate_fn_padd_2(cls,batch):
        '''
        Padds batch of variable length
        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## padd
        
        batch_x = [torch.Tensor(t) for t,y in batch]
        batch_y = [y for t,y in batch]
        batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first = True)
        batch_x = batch_x.unsqueeze(1)
        batch_y = torch.Tensor(batch_y).type(torch.LongTensor)

        return batch_x,batch_y

    @classmethod
    def collate_fn_padd(cls,batch):
        '''
        Padds batch of variable length
        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## padd
        
        batch_x = [torch.squeeze(torch.Tensor(t)) for t,y in batch]
        batch_y = [y for t,y in batch]
        batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first = True)
        batch_x = batch_x.unsqueeze(1)
        batch_y = torch.Tensor(batch_y).type(torch.LongTensor)
        # print("batch y",batch_y)
        return batch_x,batch_y