import os
import torch
from torch.utils.data import DataLoader
from dataset import read_protocol, ASVDataset
from model import BaselineModel

if __name__ == "__main__":
    protocol_file = "ASVspoof2019.LA.cm.eval.trl.txt"
    audio_folder = "LA/ASVspoof2019_LA_eval/flac"
    out_dir = "students_solutions"
    out_path = os.path.join(out_dir, "ilalbogatyrev@edu.hse.ru.csv")

    os.makedirs(out_dir, exist_ok=True)

    labels = read_protocol(protocol_file)
    dataset = ASVDataset(audio_folder, labels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = BaselineModel()
    model.eval()

    with open(out_path, 'w') as f:
        with torch.no_grad():
            for idx, (wave, _) in enumerate(loader):
                file_id = dataset.ids[idx]
                score = model(wave).item()
                f.write(f"{file_id},{score}\n")