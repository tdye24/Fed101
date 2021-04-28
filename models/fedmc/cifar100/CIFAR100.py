import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR100(nn.Module):
    def __init__(self):
        super(CIFAR100, self).__init__()
        self.g_cov1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.g_cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.g_drop_out1 = nn.Dropout(p=0.25)
        self.g_cov3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.g_cov4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.g_drop_out2 = nn.Dropout(p=0.25)

        self.l_cov1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.l_cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.l_drop_out1 = nn.Dropout(p=0.25)
        self.l_cov3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.l_cov4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.l_drop_out2 = nn.Dropout(p=0.25)

        self.clf1 = nn.Linear(64 * 5 * 5 * 2, 512)
        self.clf_drop_out = nn.Dropout(p=0.5)
        self.clf2 = nn.Linear(512, 100)

        self.critic1 = nn.Linear(64 * 5 * 5, 512)
        self.critic_drop_out = nn.Dropout(p=0.5)
        self.critic2 = nn.Linear(512, 10)

    def forward(self, x):
        g_x = F.relu(self.g_cov1(x), inplace=True)
        g_x = F.relu(self.g_cov2(g_x), inplace=True)
        g_x = F.max_pool2d(g_x, 2, 2)
        g_x = self.g_drop_out1(g_x)
        g_x = F.relu(self.g_cov3(g_x), inplace=True)
        g_x = F.relu(self.g_cov4(g_x), inplace=True)
        g_x = F.max_pool2d(g_x, 2, 2)
        g_x = self.g_drop_out2(g_x)

        l_x = F.relu(self.l_cov1(x), inplace=True)
        l_x = F.relu(self.l_cov2(l_x), inplace=True)
        l_x = F.max_pool2d(l_x, 2, 2)
        l_x = self.l_drop_out1(l_x)
        l_x = F.relu(self.l_cov3(l_x), inplace=True)
        l_x = F.relu(self.l_cov4(l_x), inplace=True)
        l_x = F.max_pool2d(l_x, 2, 2)
        l_x = self.l_drop_out2(l_x)

        g_x = g_x.flatten(start_dim=1)
        l_x = l_x.flatten(start_dim=1)

        feature = torch.cat((g_x, l_x), dim=-1)
        clf1_out = self.clf1(feature)
        clf1_out = self.clf_drop_out(clf1_out)
        clf2_out = self.clf2(clf1_out)

        g_critic1_out = self.critic1(g_x)
        g_critic1_out = self.critic_drop_out(g_critic1_out)
        g_critic2_out = F.softmax(self.critic2(g_critic1_out), dim=1)

        l_critic1_out = self.critic1(l_x)
        l_critic1_out = self.critic_drop_out(l_critic1_out)
        l_critic2_out = F.softmax(self.critic2(l_critic1_out), dim=1)

        return g_critic2_out, l_critic2_out, clf2_out


if __name__ == '__main__':
    model = CIFAR100()
    x = torch.rand((50, 3, 32, 32))
    g_critic_out, l_critic_out, output = model(x)
    print(f'{x.shape}->g_critic_out{g_critic_out.shape}')
    print(f'{x.shape}->l_critic_out{l_critic_out.shape}')
    print(f'{x.shape}->{output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))

    print("Comm.")
    total = 0
    for key, param in model.named_parameters():
        if key.startswith('g') or key.startswith('critic'):
            total += param.numel()
    print("Comm. Parameters {}".format(total))