
from .persist import load_model
from .model import init_weights_simple
from .constants import nfeat, nsensors
import numpy as np
import torch


def encode_load(name="simpleModel_ini", dir_path="../src/vne/models"):
    """
    This function will load a saved model, test it, then save a persistent version (a file)
    Parameters
    ----------

    name:
        the filename of the saved model

    dir_path:
        the directory path to the folder containing the file with the saved model
    Returns
    -------
    model:
        the network model contained in memory
    """
    # load the saved model
    model = load_model(name, dir_path)

    # do some stuff, optional

    # save the model
    model.apply(init_weights_simple)
    return model

def mat2vne(sensor_signals=None, model = None):
    """
    This function can be called from MATLAB and used to send signals through the vne encoder
    Parameters
    ----------
    sensor_signals:
        MATLAB array of shape 1 x nsensors
    model:
        reference to previously loaded model (preload for better performance)

    Returns
    -------
    model:
        the network model contained in memory
    """

    # use the model
    if sensor_signals is None:
        sensor_signals = np.random.random(nsensors)
    sig = np.zeros(1, nfeat, nsensors)
    sig[0, 0, :] = sensor_signals

    sig = torch.tensor(sig.astype(np.float32)).to('cpu')
    enc = model(sig)

    return enc.numpy()

