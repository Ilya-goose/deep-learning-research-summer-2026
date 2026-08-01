import os
import torch
from torch.utils.data import DataLoader
from dataset import read_protocol, ASVDataset
from model import LightCNN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protocol_file = "ASVspoof2019.LA.cm.eval.trl.txt"
    audio_folder = "LA/ASVspoof2019_LA_eval/flac"
    out_dir = "students_solutions"
    out_path = os.path.join(out_dir, "ilalbogatyrev@edu.hse.ru.csv")

    os.makedirs(out_dir, exist_ok=True)

    labels = read_protocol(protocol_file)
    dataset = ASVDataset(audio_folder, labels, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = LightCNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", weights_only=True, map_location=device))
    model.eval()

    with open(out_path, 'w') as f:
        with torch.no_grad():
            for idx, (wave, _) in enumerate(loader):
                wave = wave.to(device)
                file_id = dataset.ids[idx]
                score = torch.sigmoid(model(wave)).item()
                f.write(f"{file_id},{score}\n")