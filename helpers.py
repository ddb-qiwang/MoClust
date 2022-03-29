import math
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def npy(t, to_cpu=True):
    """
    Convert a tensor to a numpy array.
    """
    if isinstance(t, (list, tuple)):
        # We got a list. Convert each element to numpy
        return [npy(ti) for ti in t]
    elif isinstance(t, dict):
        # We got a dict. Convert each value to numpy
        return {k: npy(v) for k, v in t.items()}
    # Assuming t is a tensor.
    if to_cpu:
        return t.cpu().detach().numpy()
    return t.detach().numpy()


def ensure_iterable(elem, expected_length=1):
    if isinstance(elem, (list, tuple)):
        assert len(elem) == expected_length, f"Expected iterable {elem} with length {len(elem)} does not have " \
                                             f"expected length {expected_length}"
    else:
        elem = expected_length * [elem]
    return elem


def dict_means(dicts):

    return pd.DataFrame(dicts).mean(axis=0).to_dict()


def add_prefix(dct, prefix, sep="/"):

    return {prefix + sep + key: value for key, value in dct.items()}


def ordered_cmat(labels, pred):

    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    return acc, ordered

def he_init_weights(module):

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


def num2tuple(num):
    return num if isinstance(num, (tuple, list)) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):

    h_w, kernel_size, stride, = num2tuple(h_w), num2tuple(kernel_size), num2tuple(stride)
    pad, dilation = num2tuple(pad), num2tuple(dilation)

    h = math.floor((h_w[0] + 2 * pad[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + 2 * pad[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w
