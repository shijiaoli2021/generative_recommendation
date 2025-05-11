"""
normalize util
"""

import paddle.nn as nn
from paddle.nn import functional as F


def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, axis=dim, epsilon=eps)

class L2Normalization(nn.Layer):

    def __init__(self, dim=-1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return l2norm(x, self.dim, self.eps)
