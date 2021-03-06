import copy
import torch
import torch.nn as nn
import numpy as np


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.global_feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.local_feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.GATE_feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.GATE_fc = nn.Sequential(
            nn.Linear(64*5*5, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )

        # 改成2048
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 62),
        )

    def forward(self, x):
        global_feat = self.global_feature(x)
        global_feat_flatten = global_feat.flatten(start_dim=1).unsqueeze(1)
        local_feat = self.local_feature(x)
        local_feat_flatten = local_feat.flatten(start_dim=1).unsqueeze(1)
        feature = torch.cat((global_feat_flatten, local_feat_flatten), 1)
        gate_feat = self.GATE_feature(x)
        gate_feat_flatten = gate_feat.flatten(start_dim=1)
        gate_output = self.GATE_fc(gate_feat_flatten)

        avg_gate_output = torch.sum(gate_output, dim=0)/gate_output.shape[0]
        # print(avg_gate_output)

        gate_output = gate_output.unsqueeze(-1)
        feature = torch.sum(feature * gate_output, dim=1)
        output = self.fc(feature)
        return output, avg_gate_output


if __name__ == '__main__':
    model = CIFAR10()
    _x = torch.rand((50, 3, 32, 32))
    _output, avg_gate_output_ = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

