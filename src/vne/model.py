"""
Code for generating synthetic data sample using driver parameters.

"""

import torch.nn as nn
import torch
from .constants import nsensors, nencoded

class simpleModel(nn.Module):
    """
    Encoder Class

    Parameters
    -----------

    n_sensors: int
        number of  dimension, a scalar
    n_encoded: int
        number of dimension, a scalar

    Returns
    -------------
    simpleModel: torch.nn.module
        an instance of the encoder

    Methods
    -------
    forward: computes the forward pass through the encoder model.
    """

    def __init__(self,
                 n_sensors = nsensors,
                 n_encoded = nencoded):
        super(simpleModel, self).__init__()
        self.input_args = {
            'n_sensors': n_sensors,
            'n_encoded': n_encoded,
        }
        # Build a 'lil baby neural network
        # self.gen_model = nn.Linear(n_sensors, n_encoded)


        # Build a great big neural network
        self.gen_model = nn.Sequential(
            nn.Linear(n_sensors, n_sensors))

    def forward(self, sensors):
        """
        Function for completing a forward pass of the encoder:
        Returns encoded sensor data.

        Parameters
        --------------

        sensors: torch.Tensor, optional
            the input sensor data with dimension (n_batch, latent_dim)

        Returns
        -------
        encoded:torch.Tensor
            tensor of encoded sensor data
        """

        encoded = self.gen_model(sensors)
        return encoded

def init_weights_simple(m):
    """function to re-randomize initial weights so the model passes signals straight through"""
    if isinstance(m, nn.Linear):
        torch.nn.init.eye_(m.weight)
        m.weight.data.multiply_(0.9)
        m.bias.data.fill_(0.00)
    # if isinstance(m, nn.Linear):
    #     torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
    #     m.weight.data.multiply_(nsensors/torch.sum(m.weight))
    #     m.bias.data.fill_(0.00)
    # # if isinstance(m, nn.Linear):
    #     torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
    #     m.weight.data.multiply_(nsensors/torch.sum(m.weight))
    #     m.bias.data.fill_(0.00)
