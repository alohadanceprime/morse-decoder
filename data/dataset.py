import os
import torch
from torch.utils.data import Dataset
from .preprocessing import augment_spectrogram
import numpy as np


class SpectrogramDataset(Dataset):
    def __init__(self, annotations_df, root_dir, char_to_index, augment=False):
        self.annotations = annotations_df
        self.root_dir = root_dir
        self.char_to_index = char_to_index
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        spec_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        spec_path = spec_path[:-4] + "npy"
        spectrogram = torch.from_numpy(np.load(spec_path)).float().unsqueeze(0)
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)   # Нормализация

        if self.augment:
            spectrogram = augment_spectrogram(spectrogram)

        message = self.annotations.iloc[idx, 1]
        target = [self.char_to_index[char] for char in message]  # Переводим в список индексов
        return spectrogram, torch.tensor(target, dtype=torch.long), len(target)  # Длинна цели нужна для CTCLoss


class TestSpectrogramDataset(Dataset):
    def __init__(self, ids, root_dir):
        self.ids = ids
        self.root_dir = root_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        spec_path = os.path.join(self.root_dir, self.ids[idx])
        spec_path = spec_path[:-4] + "npy"
        spectrogram = torch.from_numpy(np.load(spec_path)).float().unsqueeze(0)
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
        return spectrogram, self.ids[idx]
