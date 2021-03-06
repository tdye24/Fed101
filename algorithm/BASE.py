import torch

# fedavg
from models.fedavg.femnist.FEMNIST import FEMNIST as FedAVG_FEMNIST
from models.fedavg.cifar10.CIFAR10 import CIFAR10 as FedAVG_CIFAR10
from models.fedavg.mnist.MNIST import MNIST as FedAVG_MNIST
from models.fedavg.flickr.FLICKR import FLICKR as FedAVG_FLICKR
from models.fedavg.celeba.CELEBA import CELEBA as FedAVG_CELEBA

# fedsp
from models.fedsp.femnist.FEMNIST import FEMNIST as FedSP_FEMNIST
from models.fedsp.cifar10.CIFAR10 import CIFAR10 as FedSP_CIFAR10
from models.fedsp.flickr.FLICKR import FLICKR as FedSP_FLICKR
from models.fedsp.celeba.CELEBA import CELEBA as FedSP_CELEBA

# fedper
from models.fedper.femnist.FEMNIST import FEMNIST as FedPER_FEMNIST
from models.fedper.cifar10.CIFAR10 import CIFAR10 as FedPER_CIFAR10

# fedprox
from models.fedprox.femnist.FEMNIST import FEMNIST as FedPROX_FEMNIST
from models.fedprox.cifar10.CIFAR10 import CIFAR10 as FedPROX_CIFAR10

# fedmoe
from models.fedmoe.femnist.FEMNIST import FEMNIST as FedMoE_FEMNIST
from models.fedmoe.cifar10.CIFAR10 import CIFAR10 as FedMoE_CIFAR10


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
            elif model_name == 'cifar10':
                model = FedAVG_CIFAR10()
            elif model_name == 'mnist':
                model = FedAVG_MNIST()
            elif model_name == 'flickr':
                model = FedAVG_FLICKR()
            elif model_name == 'celeba':
                model = FedAVG_CELEBA()
        elif algorithm == 'fedsp':
            if model_name == 'femnist':
                model = FedSP_FEMNIST()
            elif model_name == 'cifar10':
                model = FedSP_CIFAR10()
            elif model_name == 'flickr':
                model = FedSP_FLICKR()
            elif model_name == 'celeba':
                model = FedSP_CELEBA()
        elif algorithm == 'fedper':
            if model_name == 'femnist':
                model = FedPER_FEMNIST()
            elif model_name == 'cifar10':
                model = FedPER_CIFAR10()
        elif algorithm == 'fedprox':
            if model_name == 'femnist':
                model = FedPROX_FEMNIST()
            elif model_name == 'cifar10':
                model = FedPROX_CIFAR10()
        elif algorithm == 'fedmoe':
            if model_name == 'femnist':
                model = FedMoE_FEMNIST()
            elif model_name == 'cifar10':
                model = FedMoE_CIFAR10()
        return model
