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

        self.fc = nn.Sequential(
            nn.Linear(64*7*7*2, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 62)
        )

    def forward(self, x):
        x = x.reshape((-1, 1, 28, 28))
        global_feat = self.global_feature(x)
        global_feat_flatten = global_feat.flatten(start_dim=1)
        local_feat = self.local_feature(x)
        local_feat_flatten = local_feat.flatten(start_dim=1)
        feature = torch.cat((global_feat_flatten, local_feat_flatten), dim=1)
        # feature_flatten = feature.flatten(start_dim=1)
        output = self.fc(feature)
        return output


if __name__ == '__main__':
    model = FEMNIST()
    _x = torch.rand((50, 1, 28, 28))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

