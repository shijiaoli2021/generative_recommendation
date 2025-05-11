"""
encoder util here
"""
import paddle
from normalize import *

class MLP(nn.Layer):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 dropout:float= 0.1,
                 normalize:bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        dims = [input_dim] + hidden_dims + [output_dim]
        self.mlp = nn.Sequential()
        for i, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):

            self.mlp.append(nn.Linear(input_dim, output_dim))
            if i != len(dims)-2:
                self.mlp.append(nn.Silu())
                if dropout != 0:
                    self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(L2Normalization())

    def forward(self, x:paddle.Tensor):
        return self.mlp(x)


