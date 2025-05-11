"""
function for dataset
"""

import paddle
import paddle.nn as nn
from util import *
from paddle.io import Dataset, DataLoader

class ItemDataset(Dataset):
    def __init__(
        self,
        data,
        ):
        super().__init__()
        self.item_embeddings = data

    def __len__(self):
        return len(self.item_embeddings)

    def __getitem__(self, idx):
        return self.item_embeddings[idx]



