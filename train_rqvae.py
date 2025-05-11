import numpy as np
import pandas as pd
import paddle
from paddle.io import DataLoader

import rqvae
from rqvae_dataset import ItemDataset
from rqvae import *
from quantize import QuantizeForwardMode
from rqvaeargs import RqvaeArgs




if __name__ == '__main__':

    # data


    args = RqvaeArgs()
    # model
    model = rqvae.RqVae(args)

    # optimizer
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, epsilon=args.eps)

