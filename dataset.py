import os
import random
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

def read_protocol(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                data[parts[1]] = parts[4]
    return data

class ASVDataset(Dataset):
    def __init__(self, audio_dir, protocol, is_train=True, max_len=64000):
        self.audio_dir = audio_dir
        self.protocol = protocol
        self.ids = list(protocol.keys())
        self.is_train = is_train
        self.max_len = max_len
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=16000,
            n_lfcc=60,
            speckwargs={"n_fft": 512, "hop_length": 160}
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        label = self.protocol[id_]
        path = os.path.join(self.audio_dir, f"{id_}.flac")
        data, sr = sf.read(path)
        wave = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        if wave.shape[1] > self.max_len:
            if self.is_train:
                start = random.randint(0, wave.shape[1] - self.max_len)
            else:
                start = 0
            wave = wave[:, start:start + self.max_len]
        else:
            pad = self.max_len - wave.shape[1]
            wave = torch.nn.functional.pad(wave, (0, pad))

        features = self.lfcc(wave)
        target = 1.0 if label == "bonafide" else 0.0
        return features, target