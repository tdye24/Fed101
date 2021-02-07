import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import copy
from data.femnist.FEMNIST_DATASET import get_femnist_dataloaders


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
        # 改成2048
        self.fc = nn.Sequential(
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
        output = self.fc(feature)
        return output


def client_train(dataloader, model, _device):
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)

    epoch_loss = []
    for epoch in range(5):
        batch_loss = []
        for step, (data, labels) in enumerate(dataloader):
            data, labels = data.to(_device), labels.to(_device)
            optimizer.zero_grad()
            output = model(data)
            _loss = criterion(output, labels)
            _loss.backward()
            optimizer.step()
            batch_loss.append(_loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def client_test(dataloader, model, _device):
    model.eval()
    test_loss = 0
    correct = 0
    sample_num = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, (data, labels) in enumerate(dataloader):
            data, labels = data.to(_device), labels.to(_device)
            output = model(data)
            test_loss += criterion(output, labels).item()
            output = output.data.max(1)[1]
            correct += output.eq(labels.data.view_as(output)).float().cpu().sum()
            sample_num += len(labels)
    accuracy = 100 * float(correct) / sample_num
    test_loss /= sample_num
    # print(accuracy, test_loss)
    return accuracy, test_loss


def fedavg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


if __name__ == '__main__':
    optimal = 0.0
    # 数据集
    users, trainloaders, testloaders = get_femnist_dataloaders(batch_size=10, train_transform=None, test_transform=None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # global model
    net_global = FEMNIST().to(device)
    # global params
    net_global.train()
    w_global = net_global.state_dict()

    # clients' model
    local_nets = {}
    for user in users:
        local_nets[user] = FEMNIST().to(device)
        local_nets[user].load_state_dict(w_global)

    # start training
    for round_th in range(1000):
        print("Round {}".format(round_th))

        # training
        selected_ids = np.random.choice(a=len(users), size=10)
        selected_users = [users[id_] for id_ in selected_ids]
        w_locals, loss_locals = [], []
        for user in selected_users:
            local_nets[user].train()
            w, loss = client_train(dataloader=trainloaders[user], model=local_nets[user], _device=device)
            w_locals.append(w)
            loss_locals.append(copy.deepcopy(loss))
        avg_loss = sum(loss_locals) / len(loss_locals)
        w_global = fedavg(w_locals)
        net_global.load_state_dict(w_global)

        for user in users:
            local_nets[user].global_feature.load_state_dict(net_global.global_feature.state_dict())

        # test
        acc = 0
        loss = 0
        for user in users:
            a, l = client_test(dataloader=testloaders[user], model=local_nets[user], _device=device)
            acc += a
            loss += l
        acc, loss = acc/len(users), loss/len(users)
        if acc > optimal:
            optimal = acc
            print("\033[1;31m" + "***Best Model***SAVE***" + "\033[0m")
        print(f"acc: {acc}, loss: {loss}")
