import torch
import torch.nn as nn


class FEMNIST(nn.Module):
    def __init__(self):
        super(FEMNIST, self).__init__()
        self.global_feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.local_feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.critic_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

        self.clf_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        global_feat = self.global_feature(x)
        local_feat = self.local_feature(x)
        global_feat_flat = global_feat.flatten(start_dim=1)
        local_feat_flat = local_feat.flatten(start_dim=1)
        feature = torch.cat((global_feat_flat, local_feat_flat), dim=1)
        output = self.clf_fc(feature)
        return output
