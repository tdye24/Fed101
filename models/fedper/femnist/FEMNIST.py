import torch
import torch.nn as nn


class FEMNIST(nn.Module):
    def __init__(self):
        super(FEMNIST, self).__init__()
        self.shared_base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.private_personalization = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 62)
        )

        # self.shared_fc = nn.Sequential(
        #     nn.Linear(512, 62)
        # )

    def forward(self, x):
        base_output = self.shared_base(x)
        base_output_flatten = base_output.flatten(start_dim=1)
        output = self.private_personalization(base_output_flatten)
        # output = self.shared_fc(per_output)
        return output


if __name__ == '__main__':
    model = FEMNIST()
    _x = torch.rand((50, 1, 28, 28))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

