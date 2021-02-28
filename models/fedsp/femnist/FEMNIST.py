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
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # 改成2048
        self.fc = nn.Sequential(
            nn.Linear(64*7*7 + 16*7*7, 512),  # 乘2因为global_feat和local_feat拼在一起
            nn.ReLU(inplace=True),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        global_feat = self.global_feature(x)
        local_feat = self.local_feature(x)
        global_feat_flat = global_feat.flatten(start_dim=1)
        local_feat_flat = local_feat.flatten(start_dim=1)
        feature = torch.cat((global_feat_flat, local_feat_flat), dim=1)
        output = self.fc(feature)
        return output


if __name__ == '__main__':
    model = FEMNIST()
    _x = torch.rand((50, 1, 28, 28))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

