import torch

from models.fedavg.femnist.FEMNIST import FEMNIST as FedAVG_FEMNIST
from models.fedavg.mnist.MNIST import MNIST as FedAVG_MNIST

from models.fedsp.femnist.FEMNIST import FEMNIST as FedSP_FEMNIST

from models.fedper.femnist.FEMNIST import FEMNIST as FedPER_FEMNIST

from models.fedprox.femnist.FEMNIST import FEMNIST as FedPROX_FEMNIST


class BASE:
    def __init__(self,
                 algorithm='fedavg',
                 seed=123,
                 epoch=1,
                 model_name='femnist',
                 lr=3e-4,
                 batch_size=1,
                 lr_decay=0.99,
                 decay_step=200):

        torch.manual_seed(seed)  # recurrence experiment
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed
        self.epoch = epoch
        self.model = self.select_model(algorithm, model_name)
        self.lr = lr
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.decay_step = decay_step

    @staticmethod
    def select_model(algorithm, model_name):
        model = None
        if algorithm == 'fedavg':
            if model_name == 'femnist':
                model = FedAVG_FEMNIST()
            elif model_name == 'mnist':
                model = FedAVG_MNIST()
        elif algorithm == 'fedsp':
            if model_name == 'femnist':
                model = FedSP_FEMNIST()
        elif algorithm == 'fedper':
            if model_name == 'femnist':
                model = FedPER_FEMNIST()
        elif algorithm == 'fedprox':
            if model_name == 'femnist':
                model = FedPROX_FEMNIST()
        return model