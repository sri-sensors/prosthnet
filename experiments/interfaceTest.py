from vne.interface import encode_load
from vne.model import init_weights_simple
import numpy as np
from vne.constants import nfeat, nsensors
import torch

if __name__ == "__main__":
    model = encode_load()

    # use the model
    sig = np.random.random([1, nfeat, nsensors])
    sig = torch.tensor(sig.astype(np.float32)).to('cpu')
    enc = model(sig)
    print("signal = {}".format(sig))
    print("encoded = {}".format(enc))

    # use the model
    sig = np.random.random([1, nfeat, nsensors])
    sig = torch.tensor(sig.astype(np.float32)).to('cpu')
    enc = model(sig)
    print("signal = {}".format(sig))
    print("encoded = {}".format(enc))

    # use the model
    sig = np.random.random([1, nfeat, nsensors])
    sig = torch.tensor(sig.astype(np.float32)).to('cpu')
    enc = model(sig)
    print("signal = {}".format(sig))
    print("encoded = {}".format(enc))