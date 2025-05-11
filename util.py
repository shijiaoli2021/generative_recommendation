"""
util for data preprocess
"""

import numpy as np
import pandas as pd
from tqdm import *

AD_TOTAL_COUNT = 1300000

def read_ad_data(data_path, ratio=0.7):
    ad_unit_dict = dict()
    limit_count = int(AD_TOTAL_COUNT * ratio)
    with open(data_path, 'r') as f:
        for line in tqdm(f):
            parts = line.split("\t")
            ad_id = parts[0]
            embedding = np.array(list(map(np.float32, parts[2].split(","))))
            ad_unit_dict[ad_id] = embedding
    return ad_unit_dict



