"""
distribution util
"""
import paddle.nn as nn
import paddle
from paddle import Tensor
from typing import Tuple
from paddle.nn import functional as F

def sample_gumbel(shape: Tuple, device: paddle.device, eps=1e-20):
    """sample from gumbel (0, 1)"""
    U = paddle.rand(shape, dtype="float32").to(device)
    return -paddle.log(-paddle.log(U + eps) + eps)

def gumbel_softmax_sample(logits: Tensor, temperature: float, device: paddle.device) -> Tensor:

    y = logits + sample_gumbel(logits.shape, device)
    sample = F.softmax(y / temperature, axis=-1)
    return sample
