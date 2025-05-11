"""
loss function
"""
import paddle
import paddle.nn as nn
from paddle import Tensor


class ReconstructionLoss(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat:Tensor, x:Tensor)->Tensor:
        return ((x_hat - x) ** 2).sum(axis=-1)


class QuantizeLoss(nn.Layer):
    def __init__(self, commitment_weight: float = 1.0):
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query:Tensor, value:Tensor):
        emb_loss = ((query.detach() - value) ** 2).sum(axis=[-1])
        query_loss = ((query - value.detach()) ** 2).sum(axis=[-1])
        return emb_loss + query_loss * self.commitment_weight