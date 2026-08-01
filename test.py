import os
import csv
import torch
from torch.utils.data import DataLoader
from dataset import read_protocol, ASVDataset
from model import LightCNN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = read_protocol("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    dataset = ASVDataset("LA/ASVspoof2019_LA_eval/flac", labels, is_train=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = LightCNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", weights_only=True, map_location=device))
    model.eval()

    os.makedirs("students_solutions", exist_ok=True)

    idx = 0
    with open("students_solutions/ilalbogatyrev.csv", "w", newline="") as f:
        writer = csv.writer(f)
        with torch.no_grad():
            for feats, _ in loader:
                feats = feats.to(device)
                out = model(feats)
                scores = torch.sigmoid(out).squeeze(1).cpu().numpy()
                for score in scores:
                    file_id = dataset.ids[idx]
                    writer.writerow([file_id, float(score)])
                    idx += 1