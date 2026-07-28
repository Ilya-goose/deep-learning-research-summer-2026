import os
import librosa
import torch
from torch.utils.data import Dataset

def read_protocol(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                key = parts[1]
                value = parts[4]
                data[key] = value
    return data

class ASVDataset(Dataset):
    def __init__(self, audio_dir, protocol):
        self.audio_dir = audio_dir
        self.protocol = protocol
        self.ids = list(protocol.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        label = self.protocol[id_]
        path = os.path.join(self.audio_dir, f"{id_}.flac")
        wave, sr = librosa.load(path, sr=None)
        wave = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)
        target = 1.0 if label == "bonafide" else 0.0
        return wave, target