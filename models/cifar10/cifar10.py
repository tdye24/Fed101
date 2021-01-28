import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(5 * 5 * 64, 1024)
        self.drop_out1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.drop_out2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # 100x3x32x32
        x = F.relu(self.conv1(x), inplace=True)
        # 100x32x28x28
        x = F.max_pool2d(x, 2, 2)
        # 100x32x14x14
        x = F.relu(self.conv2(x), inplace=True)
        # 100x64x10x10
        x = F.max_pool2d(x, 2, 2)
        # 100x64x5x5
        x = x.view(-1, 5 * 5 * 64)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.drop_out1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.drop_out2(x)
        x = self.fc3(x)
        return x


