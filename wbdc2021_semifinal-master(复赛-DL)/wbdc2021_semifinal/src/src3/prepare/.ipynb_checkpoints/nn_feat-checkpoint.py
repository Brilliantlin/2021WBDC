import sys
sys.path.append('..')
# sys.path.append('../config/')
sys.path.append('../../config/')
from config_inger import MyArgparse
from tqdm import tqdm
from collections import defaultdict
import gc
import time
import pickle
import os 
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from deepctr_torch.inputs import build_input_features,get_feature_names

def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(data, f,protocol = 4)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data
args = MyArgparse()


def test_feat(test_path):
    df = pd.read_csv(test_path)
    feed_info = pd.read_csv(args.FEED_INFO)
    feed_info[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)

    df = df.merge(feed_info[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                  on='feedid')
    dense_features = ['videoplayseconds']
    df[dense_features] = np.log(df[dense_features] + 1.0)
    graph_emb8 = load_pickle(args.emb_path+'graph_walk_emb_8.pkl')
    feed_emb_16 = load_pickle(args.emb_path+'feed_embeddings_16.pkl')
    weight_emb8 = load_pickle(args.emb_path+'user_weight_emd_8.pkl')
    weight_emb8 = weight_emb8.drop('user_date_weight_emd',axis = 1)
    keyword_w2v_8 = load_pickle(args.emb_path+'keyword_w2v_8.pkl')
    userid_feedid_d2v_all_16 = load_pickle(args.emb_path+'userid_feedid_d2v_all_16.pkl')##加了初赛数据
    all_text_data_v8 = load_pickle(args.emb_path+'all_text_data_v8.pkl')
    userid_authorid_d2v_all_16 = load_pickle(args.emb_path+'userid_authorid_d2v_all_16.pkl')

    df = df.merge(graph_emb8, how='left',
                  on='userid')
    df = df.merge(feed_emb_16, how='left',
                  on='feedid')
    df = df.merge(weight_emb8, how='left',
                  on='userid')
    df = df.merge(keyword_w2v_8, how='left',
                  on='feedid')
    df = df.merge(userid_feedid_d2v_all_16, how='left',
                  on='userid')
    df = df.merge(all_text_data_v8, how='left',
                  on='feedid')
    df = df.merge(userid_authorid_d2v_all_16, how='left',
                  on='userid')


    IDS = ['userid',
        'feedid',
        'authorid',
        'bgm_song_id' ,
        'bgm_singer_id']

    #encoder
    for col in IDS:
        lbe = load_pickle(os.path.join(args.encoder_path,'ID_col_%s_encoder.pkl' % (col)))
        df[col] = lbe.transform(df[col])
    categorical_feature = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id']
    
     ##转为torch data
    fixlen_feature_columns = load_pickle(args.fixlen_feature_columns)
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
    y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

    test_model_input = {name: df[name] for name in feature_names}
    x = [test_model_input[feature] for feature in feature_index]
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    x_torch_data = torch.from_numpy(
            np.concatenate(x, axis=-1))
    return x_torch_data
# test_model_input = test_feat('../../data/wedata/wechat_algo_data2/test_a.csv')