import torch
import random
import numpy as np
from torchvision.transforms import transforms

# models
# fedavg
from models.fedavg.mnist.MNIST import MNIST as FedAVG_MNIST
from models.fedavg.femnist.FEMNIST import FEMNIST as FedAVG_FEMNIST
from models.fedavg.cifar10.CIFAR10 import CIFAR10 as FedAVG_CIFAR10
from models.fedavg.cifar100.CIFAR100 import CIFAR100 as FedAVG_CIFAR100
from models.fedavg.flickr.FLICKR import FLICKR as FedAVG_FLICKR
from models.fedavg.celeba.CELEBA import CELEBA as FedAVG_CELEBA
from models.fedavg.har.HAR import HAR as FedAVG_HAR

# fedsp
from models.fedsp.mnist.MNIST import MNIST as FedSP_MNIST
from models.fedsp.femnist.FEMNIST import FEMNIST as FedSP_FEMNIST
from models.fedsp.cifar10.CIFAR10 import CIFAR10 as FedSP_CIFAR10
from models.fedsp.cifar100.CIFAR100 import CIFAR100 as FedSP_CIFAR100
from models.fedsp.flickr.FLICKR import FLICKR as FedSP_FLICKR
from models.fedsp.celeba.CELEBA import CELEBA as FedSP_CELEBA
from models.fedsp.har.HAR import HAR as FedSP_HAR

# fedper
from models.fedper.mnist.MNIST import MNIST as FedPER_MNIST
from models.fedper.femnist.FEMNIST import FEMNIST as FedPER_FEMNIST
from models.fedper.cifar10.CIFAR10 import CIFAR10 as FedPER_CIFAR10
from models.fedper.flickr.FLICKR import FLICKR as FedPER_FLICKR
from models.fedper.har.HAR import HAR as FedPER_HAR

# fedprox
from models.fedprox.mnist.MNIST import MNIST as FedPROX_MNIST
from models.fedprox.femnist.FEMNIST import FEMNIST as FedPROX_FEMNIST
from models.fedprox.cifar10.CIFAR10 import CIFAR10 as FedPROX_CIFAR10
from models.fedprox.cifar100.CIFAR100 import CIFAR100 as FedPROX_CIFAR100
from models.fedprox.flickr.FLICKR import FLICKR as FedPROX_FLICKR
from models.fedprox.har.HAR import HAR as FedPROX_HAR

# fedmoe
from models.fedmoe.femnist.FEMNIST import FEMNIST as FedMoE_FEMNIST
from models.fedmoe.cifar10.CIFAR10 import CIFAR10 as FedMoE_CIFAR10

# fedmc
from models.fedmc.mnist.MNIST import MNIST as FedMC_MNIST
from models.fedmc.cifar10.CIFAR10 import CIFAR10 as FedMC_CIFAR10
from models.fedmc.cifar100.CIFAR100 import CIFAR100 as FedMC_CIFAR100
from models.fedmc.femnist.FEMNIST import FEMNIST as FedMC_FEMNIST
from models.fedmc.celeba.CELEBA import CELEBA as FedMC_CELEBA
from models.fedmc.flickr.FLICKR import FLICKR as FedMC_FLICKR
from models.fedmc.har.HAR import HAR as FedMC_HAR

# fedlg
from models.fedlg.mnist.MNIST import MNIST as FedLG_MNIST
from models.fedlg.cifar10.CIFAR10 import CIFAR10 as FedLG_CIFAR10
from models.fedlg.cifar100.CIFAR100 import CIFAR100 as FedLG_CIFAR100
from models.fedlg.flickr.FLICKR import FLICKR as FedLG_FLICKR
from models.fedlg.har.HAR import HAR as FedLG_HAR


class BASE:
    def __init__(self,
                 algorithm='fedavg',
                 seed=123,
                 epoch=1,
                 model_name='femnist',
                 dataset_name='femnist',
                 lr=3e-4,
                 batch_size=1,
                 lr_decay=0.99,
                 decay_step=200):

        self.setup_seed(seed)

        self.algorithm = algorithm
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.epoch = epoch
        self.model = self.select_model()
        self.lr = lr
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.decay_step = decay_step

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def select_model(self):
        algorithm = self.algorithm
        model_name = self.model_name
        model = None
        if algorithm == 'fedavg':
            if model_name == 'mnist':
                model = FedAVG_MNIST()
            elif model_name == 'femnist':
                model = FedAVG_FEMNIST()
            elif model_name == 'cifar10':
                model = FedAVG_CIFAR10()
            elif model_name == 'cifar100':
                model = FedAVG_CIFAR100()
            elif model_name == 'mnist':
                model = FedAVG_MNIST()
            elif model_name == 'flickr':
                model = FedAVG_FLICKR()
            elif model_name == 'celeba':
                model = FedAVG_CELEBA()
            elif model_name == 'har':
                model = FedAVG_HAR()
        elif algorithm == 'fedsp':
            if model_name == 'mnist':
                model = FedSP_MNIST()
            if model_name == 'femnist':
                model = FedSP_FEMNIST()
            elif model_name == 'cifar10':
                model = FedSP_CIFAR10()
            elif model_name == 'cifar100':
                model = FedSP_CIFAR100()
            elif model_name == 'flickr':
                model = FedSP_FLICKR()
            elif model_name == 'celeba':
                model = FedSP_CELEBA()
            elif model_name == 'har':
                model = FedSP_HAR()
        elif algorithm == 'fedmc' or algorithm == 'fedmc-wo':
            if model_name == 'mnist':
                model = FedMC_MNIST()
            elif model_name == 'cifar10':
                model = FedMC_CIFAR10()
            elif model_name == 'cifar100':
                model = FedMC_CIFAR100()
            elif model_name == 'femnist':
                model = FedMC_FEMNIST()
            elif model_name == 'celeba':
                model = FedMC_CELEBA()
            elif model_name == 'flickr':
                model = FedMC_FLICKR()
            elif model_name == 'har':
                model = FedMC_HAR()
        elif algorithm == 'fedper':
            if model_name == 'femnist':
                model = FedPER_FEMNIST()
            elif model_name == 'mnist':
                model = FedPER_MNIST()
            elif model_name == 'cifar10':
                model = FedPER_CIFAR10()
            elif model_name == 'flickr':
                model = FedPER_FLICKR()
            elif model_name == 'har':
                model = FedPER_HAR()
        elif algorithm == 'fedprox':
            if model_name == 'mnist':
                model = FedPROX_MNIST()
            elif model_name == 'femnist':
                model = FedPROX_FEMNIST()
            elif model_name == 'cifar10':
                model = FedPROX_CIFAR10()
            elif model_name == 'cifar100':
                model = FedPROX_CIFAR100()
            elif model_name == 'flickr':
                model = FedPROX_FLICKR()
            elif model_name == 'har':
                model = FedPROX_HAR()
        elif algorithm == 'fedlg':
            if model_name == 'mnist':
                model = FedLG_MNIST()
            elif model_name == 'cifar10':
                model = FedLG_CIFAR10()
            elif model_name == 'cifar100':
                model = FedLG_CIFAR100()
            elif model_name == 'flickr':
                model = FedLG_FLICKR()
            elif model_name == 'har':
                model = FedLG_HAR()
        elif algorithm == 'fedmoe':
            if model_name == 'femnist':
                model = FedMoE_FEMNIST()
            elif model_name == 'cifar10':
                model = FedMoE_CIFAR10()
        return model
