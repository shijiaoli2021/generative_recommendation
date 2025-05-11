"""
RQ-VAE model: transfer between input and semantic IDs
"""

import paddle
import paddle.nn as nn
from encoder import MLP
from quantize import *
from typing import NamedTuple
from normalize import *
from loss import ReconstructionLoss
from rqvaeargs import RqvaeArgs


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor

class RqVaeLoss(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor

class RqVae(nn.Layer):
    def __init__(
        self,
        args: RqvaeArgs
        ):
        super().__init__()
        self.input_dim = args.input_dim
        self.embed_dim = args.embed_dim
        self.hidden_dims = args.hidden_dims
        self.codebook_size = args.codebook_size
        self.codebook_sim_vq = args.codebook_sim_vq
        self.codebook_normalization = args.codebook_normalization
        self.num_layers = args.num_layers
        self.commitment_weight = args.commitment_weight
        self.do_kmeans_init = args.do_kmeans_init
        self.quantize_forward_mode = args.quantize_mode

        # encoder
        self.encoder = MLP(
            input_dim = self.input_dim,
            hidden_dims = self.hidden_dims,
            output_dim = self.embed_dim,
            normalize = self.codebook_normalization
        )

        # decoder
        self.decoder = MLP(
            input_dim = self.embed_dim,
            hidden_dims = self.hidden_dims,
            output_dim = self.input_dim,
            normalize = True
        )

        # quantize
        self.quantize_layers = nn.LayerList(sublayers=[
            Quantize(
                embed_dim = self.embed_dim,
                num_embeddings = self.codebook_size,
                do_kmeans_init = self.do_kmeans_init,
                commitment_weight = self.commitment_weight,
                sim_vq= self.codebook_sim_vq,
                forward_mode = self.quantize_forward_mode
            )
        ])

        # reconstruction loss
        self.reconstruction_loss = ReconstructionLoss()


    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def get_semantic_ids(self, x: Tensor, gumbel_t:float = 0.001):

        # encode
        res = self.encode(x)

        #quantize loss
        quantize_loss = 0

        embs, sem_ids, residuals = [], [], []

        # quantize
        for layer in self.quantize_layers:
            residuals.append(res)
            quantize = layer(res, gumbel_t)
            quantize_loss += quantize.loss
            emb, ids = quantize.embeddings, quantize.ids
            res = res - emb
            embs.append(emb)
            sem_ids.append(ids)

        embs = paddle.stack(embs, axis=0).transpose(perm=(1, 2, 0))
        sem_ids = paddle.stack(sem_ids, axis=0).transpose(perm=(1, 0))
        residuals = paddle.stack(residuals, axis=0).transpose(perm=(1, 2, 0))

        return RqVaeOutput(
            embs,
            residuals,
            sem_ids,
            quantize_loss
        )

    def forward(self, batch, gumbel_t:float = 0.001):
        x = batch.x

        # quantize
        quantize = self.get_semantic_ids(batch, gumbel_t)
        embs, residuals = quantize.embeddings, quantize.residuals

        # reconstruction loss
        x_hat = self.decode(embs.sum(axis=-1))
        # l2 norm
        x_hat = l2norm(x_hat)
        reconstruction_loss = self.reconstruction_loss.forward(x_hat, x)

        # quantize loss
        rqvae_loss = quantize.quantize_loss

        # loss
        loss = (reconstruction_loss + rqvae_loss).mean()

        return RqVaeLoss(
            loss = loss,
            reconstruction_loss= reconstruction_loss.mean(),
            rqvae_loss= rqvae_loss.mean()
        )



