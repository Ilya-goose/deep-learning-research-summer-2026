import torch
import torch.nn as nn

class MFM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        half = x.size(1) // 2
        return torch.max(x[:, :half], x[:, half:])

class LightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.mfm1 = MFM()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.mfm2 = MFM()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.mfm3 = MFM()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 32)
        self.mfm4 = MFM()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.pool2(self.mfm2(self.conv2(x)))
        x = self.pool3(self.mfm3(self.conv3(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.mfm4(self.fc1(x))
        x = self.fc2(x)
        return x