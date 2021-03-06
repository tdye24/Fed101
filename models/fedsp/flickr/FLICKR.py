import torch
import torch.nn as nn


class FLICKR(nn.Module):
    def __init__(self):
        super(FLICKR, self).__init__()
        self.global_feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.local_feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16*2*5*5, 60),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(60, 32),
            nn.Dropout(p=0.5),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        global_feat = self.global_feature(x)
        local_feat = self.local_feature(x)
        feature = torch.cat((global_feat, local_feat), dim=1)
        feature_flatten = feature.flatten(start_dim=1)
        output = self.fc(feature_flatten)
        return output


if __name__ == '__main__':
    model = FLICKR()
    _x = torch.rand((50, 3, 32, 32))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

