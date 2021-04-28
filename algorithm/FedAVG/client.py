import torch
import torch.optim as optim

from algorithm.BASE import BASE
from algorithm.CLIENT_BASE import CLIENT_BASE


class Client(CLIENT_BASE):
    def __init__(self, user_id, trainloader, testloader, model_name: str, lr=3e-4, epoch=1,
                 seed=123, lr_decay=0.99, decay_step=200):
        CLIENT_BASE.__init__(self,
                             algorithm='fedavg',
                             user_id=user_id,
                             trainloader=trainloader,
                             testloader=testloader,
                             seed=seed,
                             epoch=epoch,
                             model_name=model_name,
                             lr=lr,
                             lr_decay=lr_decay,
                             decay_step=decay_step)

    # rewrite
    # def train(self, round_th):
    #     pass

    # rewrite
    # def test(self, dataset='test'):
    #     pass
