import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
from dataset import read_protocol, ASVDataset
from model import LightCNN
from calculate_eer import compute_eer

if __name__ == "__main__":
    wandb.init(project="asvspoof-lcnn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labels = read_protocol("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    eval_labels = read_protocol("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")

    train_set = ASVDataset("LA/ASVspoof2019_LA_train/flac", train_labels, is_train=True)
    eval_set = ASVDataset("LA/ASVspoof2019_LA_eval/flac", eval_labels, is_train=False)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_set, batch_size=32, shuffle=False, num_workers=0)

    model = LightCNN().to(device)
    weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    best = 100.0

    for epoch in range(15):
        model.train()
        loss_sum = 0.0
        for feats, targets in train_loader:
            feats = feats.to(device)
            targets = targets.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            out = model(feats)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        model.eval()
        target_scores = []
        nontarget_scores = []
        with torch.no_grad():
            for feats, targets in eval_loader:
                feats = feats.to(device)
                out = model(feats)
                scores = torch.sigmoid(out).squeeze(1).cpu().numpy()
                for i in range(len(scores)):
                    if targets[i].item() == 1.0:
                        target_scores.append(scores[i])
                    else:
                        nontarget_scores.append(scores[i])

        target_scores = np.array(target_scores)
        nontarget_scores = np.array(nontarget_scores)
        eer, _ = compute_eer(target_scores, nontarget_scores)
        eval_eer = eer * 100

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss_sum / len(train_loader),
            "eval_eer": eval_eer
        })

        if eval_eer < best:
            best = eval_eer
            torch.save(model.state_dict(), "best_model.pth")