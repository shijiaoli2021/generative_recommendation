"""
quantize for semantic IDs
"""
import paddle
import paddle.nn as nn
from enum import Enum

from anyio import sleep

from gumbel import *
from loss import QuantizeLoss
from paddle import Tensor
from typing import NamedTuple

class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3

class QuantizeOut(NamedTuple):
    embeddings: Tensor
    loss: Tensor
    ids: Tensor

class Quantize(nn.Layer):
    def __init__(
        self,
        embed_dim:int,
        num_embeddings:int,
        do_kmeans_init: bool = True,
        commitment_weight: float = 0.25,
        sim_vq: bool = False,
        forward_mode: int = QuantizeForwardMode.GUMBEL_SOFTMAX

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.forward_mode = forward_mode
        self.kmeans_init = False
        self.do_kmeans_init = do_kmeans_init
        self.out_codebook = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim) if sim_vq else nn.Identity()
        )
        self.quantizeLoss = QuantizeLoss(commitment_weight=commitment_weight)


    def require_codebook(self)->paddle.Tensor:
        return self.out_codebook(self.embedding.weight)

    def _kmeans_init(self, x)->None:
        pass

    def _device(self):
        return self.embedding.weight.place

    def forward(self, x:paddle.Tensor, temperature):

        # kmeans-init
        if self.do_kmeans_init:
            self._kmeans_init(x)

        # codebook
        codebook = self.require_codebook()

        # calculate the distance between x and codebook

        # 1) l2
        dist = (x ** 2).sum(axis=1, keepdim=True) + (codebook.T ** 2).sum(axis=0, keepdim=True) - 2 * x @ codebook.T

        # the min dis
        ids = (dist.detach()).argmin(axis=1)

        if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:

            weights = gumbel_softmax_sample(-dist, temperature=temperature, device=self._device())

            emb = weights @ codebook

            emb_out = emb

        else:
            raise Exception("Unsupported Quantize forward mode.")

        loss = self.quantizeLoss(x, emb)

        return QuantizeOut(emb_out, loss, ids)


