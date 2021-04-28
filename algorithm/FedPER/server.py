import os
import copy
import torch
from torchvision import transforms
import numpy as np

from algorithm.BASE import BASE
from algorithm.SERVER_BASE import SERVER_BASE


class Server(SERVER_BASE):
    def __init__(self,
                 algorithm='fedper',
                 seed=123,
                 rounds=20,
                 epoch=1,
                 clients_per_round=1,
                 eval_interval=1,
                 dataset_name='femnist',
                 model_name='femnist',
                 lr=3e-4,
                 batch_size=1,
                 lr_decay=0.99,
                 decay_step=200,
                 note=''):
        SERVER_BASE.__init__(self,
                             algorithm=algorithm,
                             seed=seed,
                             rounds=rounds,
                             epoch=epoch,
                             clients_per_round=clients_per_round,
                             eval_interval=eval_interval,
                             dataset_name=dataset_name,
                             model_name=model_name,
                             lr=lr,
                             batch_size=batch_size,
                             lr_decay=lr_decay,
                             decay_step=decay_step,
                             note=note)

    # rewrite federate function
    # def federate(self):
    #     pass
