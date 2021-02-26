
import torch.nn as nn
import torch.nn.functional as F


class ML100K(nn.Module):
    def __init__(self):
        super(ML100K, self).__init__()
        self.fc1 = nn.Linear(22, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = ML100K()
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))
