from torch.nn import (
    Module,
    Sequential,
    Linear,
    LeakyReLU,
    BatchNorm1d,
    Tanh
)


class BaseGenerator(Module):
    def __init__(self, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.net = self.get_net().to(self.device)

    def get_net(self):
        return Sequential(
            Linear(self.embedding_dim, 4096),
            BatchNorm1d(4096),
            LeakyReLU(),
            #
            Linear(4096, 4096),
            BatchNorm1d(4096),
            LeakyReLU(),
            #
            Linear(4096, self.embedding_dim * 2),
            BatchNorm1d(self.embedding_dim * 2),
            Tanh()
        )

    def forward(self, grasp_object):
        return self.net(grasp_object)


class DeepThinGenerator(BaseGenerator):
    def __init__(self, embedding_dim, device):
        super().__init__(embedding_dim, device)

    def get_net(self):
        return Sequential(
            Linear(self.embedding_dim, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            #
            Linear(1024, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            #
            Linear(1024, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            #
            Linear(1024, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            #
            Linear(1024, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            #
            Linear(1024, self.embedding_dim * 2),
            BatchNorm1d(self.embedding_dim * 2),
            Tanh()
        )


class DeeperGenerator(BaseGenerator):
    def __init__(self, embedding_dim, device):
        super().__init__(embedding_dim, device)

    def get_net(self):
        return Sequential(
            Linear(self.embedding_dim, 4096),
            BatchNorm1d(4096),
            LeakyReLU(),
            #
            Linear(4096, 4096),
            BatchNorm1d(4096),
            LeakyReLU(),
            #
            Linear(4096, 2048),
            BatchNorm1d(2048),
            LeakyReLU(),
            #
            Linear(2048, 2048),
            BatchNorm1d(2048),
            LeakyReLU(),
            #
            Linear(2048, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            #
            Linear(1024, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            #
            Linear(1024, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            #
            Linear(1024, self.embedding_dim * 2),
            BatchNorm1d(self.embedding_dim * 2),
            Tanh()
        )


class DeeperNoBatchnormGenerator(BaseGenerator):
    def __init__(self, embedding_dim, device):
        super().__init__(embedding_dim, device)

    def get_net(self):
        return Sequential(
            Linear(self.embedding_dim, 4096),
            LeakyReLU(),
            #
            Linear(4096, 4096),
            LeakyReLU(),
            #
            Linear(4096, 2048),
            LeakyReLU(),
            #
            Linear(2048, 2048),
            LeakyReLU(),
            #
            Linear(2048, 1024),
            LeakyReLU(),
            #
            Linear(1024, 1024),
            LeakyReLU(),
            #
            Linear(1024, 1024),
            LeakyReLU(),
            #
            Linear(1024, self.embedding_dim * 2),
            Tanh()
        )
