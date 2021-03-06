import torch
import torch.nn as nn


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.global_feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2)
        )

        self.local_feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(8, 16, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64*5*5 + 16*5*5, 1024),  # 乘2因为global_feat和local_feat拼在一起
            torch.nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        global_feat = self.global_feature(x)
        local_feat = self.local_feature(x)

        global_feat = global_feat.flatten(start_dim=1)
        local_feat = local_feat.flatten(start_dim=1)

        feature = torch.cat((global_feat, local_feat), dim=-1)
        output = self.fc(feature)
        return output


if __name__ == '__main__':
    model = CIFAR10()
    _x = torch.rand((50, 3, 32, 32))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

