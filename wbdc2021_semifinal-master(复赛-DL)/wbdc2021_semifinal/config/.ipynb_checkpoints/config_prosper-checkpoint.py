# coding: utf-8
import os
import pickle
import time
import torch
import numpy as np
import random
import pandas as pd
import collections
# py 文件需要去掉“.notebook”
from tqdm import tqdm as tqdm
import warnings 
warnings.filterwarnings("ignore")
seed = 100
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
print('BASE_DIR(目录):',SRC_DIR)

# 存储数据的根目录
ROOT_PATH = os.path.join(SRC_DIR, 'data')
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, "wedata/wechat_algo_data2")
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action_all.pkl")
USER_ACTION_FUSAI = os.path.join(DATASET_PATH,'user_action.pkl')
USER_ACTION_CSV = os.path.join(DATASET_PATH, "user_action_all.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.pkl")
FEED_INFO_DEAL = os.path.join(DATASET_PATH, "feed_info_deal.pkl")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# B榜测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")
# 保存模型文件的目录
MODEL_PATH = os.path.join(ROOT_PATH, 'model')
# 保存结果文件的目录
SUMIT_DIR = os.path.join(ROOT_PATH, 'submission')
#
SUBMIT = os.path.join(DATASET_PATH,'submit_demo_semi_a.csv')
#特征目录


# 复赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
#权重
WEIGHT_DICT = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                   "comment": 1, "follow": 1}
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 每个行为的负样本下采样比例(下采样后负样本数/原负样本数)
ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.1, "comment": 0.1, "follow": 0.1, "favorite": 0.1}

# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
# 各个行为构造训练数据的天数
ACTION_DAY_NUM = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 5, "comment": 5, "follow": 5, "favorite": 5}


FEATURE_PATH = os.path.join(SRC_DIR,'src/src1/prepare/user_data')

#参数配置
END_DAY = 15
FEATDAYS = 5

BATCH_SIZE = 2048
SEED = 2021
DEBUG  = False
DEVICE  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PAD,UNK = 0,-1

KEY_WORDS_MAX = 27271 + 1
TAG_MAX = 352 + 1
USERID_MAX = 250248 + 1

