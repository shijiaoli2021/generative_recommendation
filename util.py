"""
util for data preprocess
"""

import numpy as np
import pandas as pd
from tqdm import *
import threading

AD_TOTAL_COUNT = 1300000

class AdDataThread(threading.Thread):
    def __init__(self, data_path, ratio:float=1.0):
        super().__init__()
        self.data_path = data_path
        self.ratio = ratio
        self.res = None

    def run(self):
        self.res = read_ad_data(self.data_path, self.ratio)

    def get_res(self):
        return self.res


def read_ad_data(data_path, ratio=0.7):
    ad_unit_dict = dict()
    limit_count = int(AD_TOTAL_COUNT * ratio)
    with open(data_path, 'r') as f:
        print(f"load ad_data begin, limit count:{limit_count}...")
        cnt = 1
        for line in tqdm(f):
            parts = line.split("\t")
            ad_id = parts[0]
            embedding = np.array(list(map(np.float32, parts[2].split(","))))
            ad_unit_dict[ad_id] = embedding
            cnt += 1
            if cnt == limit_count:
                break
        print(f"load ad_data over, count:{len(ad_unit_dict)}")
    return ad_unit_dict

def read_ad_data_flash(prefix_path:str, file_num:int, ratio:float = 1.0):
    # 创建线程
    threads = [AdDataThread(prefix_path+str(i+1)+".txt") for i in range(file_num)]

    # 启动线程
    [thread.start() for thread in threads]

    # wait
    [thread.join() for thread in threads]

    # merge result
    res = dict()
    for thread in threads:
        res.update(thread.get_res())
    print(f"merge ad_data over, totally {len(res)}")
    return res





