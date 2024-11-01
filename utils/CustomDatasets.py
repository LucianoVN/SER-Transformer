import torchaudio
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class CremaDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 num_samples=64000,
                 data_portion=None,
                 transform=None):
        
        # load .csv with dataset information
        labels_path = os.path.join(audio_dir , annotations_file)
        self.audio_labels = pd.read_csv(labels_path)
        
        # select partition of the dataset (train, validation, test)
        if data_portion:
            self.audio_labels = self.audio_labels[self.audio_labels['partition'] == data_portion]
        self.transform = transform
        self.audio_dir = audio_dir
        self.num_samples = num_samples
        
    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        # get audio path
        audio_path = os.path.join(self.audio_dir , self.audio_labels.iloc[idx]['path'])

        # load audio and label
        audio_signal, _ = torchaudio.load(audio_path)
        label = self.audio_labels.iloc[idx]['class']
        
        # modify the length of the signal
        audio_signal = self._cut_if_necessary(audio_signal)
        audio_signal = self._right_pad_if_necessary(audio_signal)
        
        # if transformation is given, apply to the audio
        if self.transform:
            audio_signal = self.transform(audio_signal)

        # return audio and label
        return audio_signal, label
    
    # cut audio if it exceeds the defined length
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples: 
            signal = signal[:, :self.num_samples] 
        return signal

    # zero padding if the audio is too short
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples: 
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal