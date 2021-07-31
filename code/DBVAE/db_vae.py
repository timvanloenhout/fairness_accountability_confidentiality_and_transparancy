""" DB-VAE class."""
import torch
import torch.nn as nn

class Flatten(nn.Module):
    """ Reshapes a 4d matrix to a 2d matrix. """
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """ Reshapes a 2d matrix to a 4d matrix. """
    def __init__(self, filter_size):
        super().__init__()
        self.filter_size = filter_size

    def forward(self, input):
        """Forward pass"""
        return input.view(input.size(0), 6*self.filter_size, 4, 4)


class Encoder(nn.Module):
    """ Encoder part of the DB_VAE. """
    def __init__(self, z_dim, filter_size):
        """
        Set up the encoding part of the network. Four convolutional layers and
        four linear layers.
        :param z_dim: Number of latent variables.
        :param filter_size: Depth of CNN.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, filter_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(filter_size),
            nn.Conv2d(filter_size, 2*filter_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(2*filter_size),
            nn.Conv2d(2*filter_size, 4*filter_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(4*filter_size),
            nn.Conv2d(4*filter_size, 6*filter_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(6*filter_size),
            Flatten(),
            nn.Linear(4 * 4 * 6 * filter_size, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000)
        )

        self.mean = nn.Linear(1000, z_dim)
        self.logvar = nn.Linear(1000, z_dim)
        self.y = nn.Linear(1000, 1)

        # initialize weights with xavier initialization
        for layer in self.encoder.children():
            try:
                nn.init.xavier_uniform_(layer.weight)
            except (ValueError, AttributeError):
                pass

        nn.init.xavier_uniform_(self.mean.weight)
        nn.init.xavier_uniform_(self.logvar.weight)
        nn.init.xavier_uniform_(self.y.weight)

    def forward(self, input):
        """
        Returns mean and std with shape [batch_size, z_dim]. Additionally
        outputs the classification prediction.
        """
        out = self.encoder(input.float())  # [batch_size, 1000]

        mean = self.mean(out)
        logvar = self.logvar(out)
        y = self.y(out)

        return mean, logvar, y


class Decoder(nn.Module):
    """ Decoder part of the DB_VAE. """
    def __init__(self, z_dim, filter_size):
        """
        Initialize all layers.
        :param z_dim: Size input of latent variables vector
        :param filter_size: Depth of the CNN
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 4 * 4 * 6 * filter_size),
            nn.ReLU(),
            nn.BatchNorm1d(4 * 4 * 6 * filter_size),
            UnFlatten(filter_size),
            nn.ConvTranspose2d(6*filter_size, 4*filter_size, 5, 2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4*filter_size),
            nn.ConvTranspose2d(4*filter_size, 2*filter_size, 5, 2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2*filter_size),
            nn.ConvTranspose2d(2*filter_size, filter_size, 5, 2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filter_size),
            nn.ConvTranspose2d(filter_size, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z_vec):
        """
        Perform forward pass of decoder. Returns reconstructed image
        of size [batch_size, channels, height, width].
        """
        return self.decoder(z_vec)


class DBVAE(nn.Module):
    """ Combines the Encoder and the Decoder. """
    def __init__(self, z_dim):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(z_dim, 12)
        self.decoder = Decoder(z_dim, 12)

    def forward(self, input):
        """
        Perform a forward pass with the Encoder. Sample a vector z with the
        parameterization trick. Perform a forward pass with the Decoder.
        :return:
            Encoder output: mean, log variance, classification.
            Reparameterization: z vector.
            Decoder: reconstructed image.
        """
        # Encoding step
        mean, logvar, pred_y = self.encoder(input)

        # Reparameterization step
        std = torch.exp(logvar / 2)
        epsilon = torch.randn(input.shape[0], self.z_dim).to(input.device)
        z_vec = mean + std * epsilon

        # Decoder step
        reconstruction = self.decoder(z_vec)

        return mean, logvar, pred_y, reconstruction, z_vec
