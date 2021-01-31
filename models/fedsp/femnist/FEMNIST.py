import torch
import torch.nn as nn


class FEMNIST(nn.Module):
    def __init__(self):
        super(FEMNIST, self).__init__()
        self.global_feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2)
        )

        self.local_feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64*4*4*2, 2048),  # 乘2因为global_feat和local_feat拼在一起
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 62)
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
    model = FEMNIST()
    _x = torch.rand((50, 1, 28, 28))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

