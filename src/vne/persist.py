"""
This module supports persistence of the generator and discriminator.

It saves two files a parameters file and a configuration file.

It also supports persisting of multiple modules.

To be persistable in this way the module must have a property
containing a json serializable input dictionary as
mdl.input_args
"""

from vne import model
import json
import torch
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, 'models')


def fullname(inst):
    tp = type(inst)
    # check if instance is actually a type
    if isinstance(inst, type):
        tp = inst
    return tp.__module__ + '.' + tp.__name__


# list types of persistable object
types = [
    model.simpleModel
]
# register types of objects that can be persisted
type_dict = {fullname(t): t for t in types}


def save_model(encoder, name='infomax_test', dir_path=None):
    """
    Persists vne encoder. Persists both input arguments and parameters.


    Parameters
    ----------
    encoder: torch.nn.Module
        module for the encoder
    name: str, optional
        name of the saved configuration
    dir_path: str, optional
        default is the models directory. None assumes
        file is in local directory.

    The encoder must be registered in persist.py and must
    have a parameter 'input_args' that specifies their input
    arguments as a dictionary.
    """
    if dir_path is not None:
        name = os.path.join(dir_path, name)
    jdict = {}
    jdict['enc_input'] = encoder.input_args
    jdict['enc_type'] = fullname(encoder)
    # verify types are registered
    for k, v in jdict.items():
        if k.endswith('type'):
            if v not in type_dict:
                print("bad dict type v = {}".format(v))
                raise ValueError(f"Unknown module type {v}.")
    with open(name+'.json', 'w+') as f:
        json.dump(jdict, f, indent=2)
    state_dict = {}
    state_dict['enc'] = encoder.state_dict()
    torch.save(state_dict, name + '.pt')


def load_model(name='infomax_test', dir_path=None):
    """
    Load a pre-trained generator and discriminator.

    Parameters
    --------------
    name: str, optional
        name of the configuration to load, as saved by
        persist. default: 'infomax_test'
    dir_path: str, optional
        default is the models directory. None assumes
        file is in local directory.

    Returns
    -------------
    encoder: torch.nn.module
        reference to the saved encoder

    Modules must have previously been saved. All modules are
    loaded on the cpu, they can subsequently be moved.
    """
    if dir_path is not None:
        name = os.path.join(dir_path, name)
    with open(name + '.json', 'r') as f:
        jdict = json.load(f)
    enc_in = jdict['enc_input']
    enc_type = type_dict[jdict['enc_type']]
    encoder = enc_type(**enc_in)
    state_dict = torch.load(name + '.pt',
                            map_location=torch.device('cpu'))
    encoder.load_state_dict(state_dict['enc'])

    return encoder
