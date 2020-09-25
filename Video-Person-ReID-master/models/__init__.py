from __future__ import absolute_import

from .ResNet import *
#from .classifier import Temporalpooling ,parts_model
from .basemodel import *
from .partclassifier import parts_model

__factory = {
    'tst': TST,
    'resnet50tp': ResNet50TP,
    'resnet50ta': ResNet50TA,
    'resnet50rnn': ResNet50RNN,
    'classifier': parts_model,
    'base': Resenetglobal,



}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
