from torch.nn import (
    Module,
    Sequential,
    Linear,
    LeakyReLU,
    Conv3d,
    BatchNorm3d,
    ConvTranspose3d,
    BatchNorm1d,
    Tanh
)
from torch import split, exp, randn_like
from .utils import Flatten


class Encoder(Module):
    def __init__(self, embedding_dim, max_log_var=0.1):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_log_var = max_log_var
        self.emb_range_limit = Tanh()
        self.net = Sequential(
            Conv3d(1, 8, kernel_size=3, stride=1),
            BatchNorm3d(8),
            LeakyReLU(),
            Conv3d(8, 16, kernel_size=3, stride=1),
            BatchNorm3d(16),
            LeakyReLU(),
            Conv3d(16, 32, kernel_size=3, stride=1),
            BatchNorm3d(32),
            LeakyReLU(),
            Conv3d(32, 64, kernel_size=3, stride=1),
            BatchNorm3d(64),
            LeakyReLU(),
            Conv3d(64, 64, kernel_size=3, stride=2),
            BatchNorm3d(64),
            LeakyReLU(),
            Conv3d(64, 32, kernel_size=3, stride=1),
            BatchNorm3d(32),
            LeakyReLU(),
            Conv3d(32, 16, kernel_size=3, stride=1),
            BatchNorm3d(16),
            LeakyReLU(),
            Conv3d(16, 8, kernel_size=3, stride=1),
            BatchNorm3d(8),
            LeakyReLU(),
            Conv3d(8, 4, kernel_size=3, stride=1),
            BatchNorm3d(4),
            LeakyReLU(),
            Flatten()
        )

    def forward(self, input):
        output = self.net(input)
        mu, logvar = split(output, self.embedding_dim, dim=1)

        return mu, logvar * self.max_log_var

    def reparameterize(self, input):
        mu, logvar = self(input)
        std = exp(logvar)
        eps = randn_like(std)
        output = self.emb_range_limit(mu + eps * std)
        return output


class Decoder(Module):
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.mlp = Sequential(
            Linear(self.embedding_dim, 1024),
            BatchNorm1d(1024),
            LeakyReLU(),
            Linear(1024, 2048),
            BatchNorm1d(2048),
            LeakyReLU(),
            Linear(2048, 4096),
            BatchNorm1d(4096),
            LeakyReLU()
        )
        self.deconv = Sequential(
            ConvTranspose3d(64, 64, kernel_size=3, stride=1),
            BatchNorm3d(64),
            LeakyReLU(),
            ConvTranspose3d(64, 64, kernel_size=3, stride=1),
            BatchNorm3d(64),
            LeakyReLU(),
            ConvTranspose3d(64, 32, kernel_size=3, stride=3),
            BatchNorm3d(32),
            LeakyReLU(),
            ConvTranspose3d(32, 16, kernel_size=3, stride=1),
            BatchNorm3d(16),
            LeakyReLU(),
            ConvTranspose3d(16, 8, kernel_size=3, stride=1),
            BatchNorm3d(8),
            LeakyReLU(),
            ConvTranspose3d(8, 4, kernel_size=3, stride=1),
            BatchNorm3d(4),
            LeakyReLU(),
            ConvTranspose3d(4, 2, kernel_size=3, stride=1),
            BatchNorm3d(2),
            LeakyReLU(),
            ConvTranspose3d(2, 1, kernel_size=3, stride=1),
            BatchNorm3d(1),
            LeakyReLU(),
            ConvTranspose3d(1, 1, kernel_size=3, stride=1),
            BatchNorm3d(1),
            LeakyReLU(),
            ConvTranspose3d(1, 1, kernel_size=3, stride=1),
            BatchNorm3d(1),
            LeakyReLU(),
            ConvTranspose3d(1, 1, kernel_size=3, stride=1),
            BatchNorm3d(1),
            Tanh()
        )

    def forward(self, input):
        output = self.mlp(input)
        output = output.view(input.size()[0], 64, 4, 4, 4)
        return self.deconv(output)
