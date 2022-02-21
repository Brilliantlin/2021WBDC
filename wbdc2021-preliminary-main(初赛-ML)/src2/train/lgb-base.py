#!/usr/bin/env python
# coding: utf-8

# # import and settings

# In[1]:

import logging

logging.basicConfig(
    level=logging.DEBUG,  #控制台打印的日志级别
    filename='train_log.txt',
    filemode='w',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    format=
    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    #日志格式
)

logger = logging.getLogger('mytrain')
logger.setLevel(logging.INFO)
import sys
sys.path.append('..')
import pandas as pd
from mytools.utils.myfile import makedirs
import numpy as np

from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from lightgbm.sklearn import LGBMClassifier

from collections import defaultdict

import  pickle

import gc

import time

import os

# 默认配置logging写入本地文件
logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S',
            filename='mytrain.log',
            filemode='w')


pd.set_option('display.max_columns', None)

def savePkl(config, filepath):
    f = open(filepath, 'wb')
    pickle.dump(config, f)
    f.close()

def loadPkl(filepath):
    f = open(filepath, 'rb')
    config = pickle.load(f)
    return config

def reduce_mem(df, cols):

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in tqdm(cols):

        col_type = df[col].dtypes

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    gc.collect()

    return df



## 从官方baseline里面抽出来的评测函数

def uAUC(labels, preds, user_id_list):

    """Calculate user AUC"""

    user_pred = defaultdict(lambda: [])

    user_truth = defaultdict(lambda: [])

    for idx, truth in enumerate(labels):

        user_id = user_id_list[idx]

        pred = preds[idx]

        truth = labels[idx]

        user_pred[user_id].append(pred)

        user_truth[user_id].append(truth)



    user_flag = defaultdict(lambda: False)

    for user_id in set(user_id_list):

        truths = user_truth[user_id]

        flag = False

        # 若全是正样本或全是负样本，则flag为False

        for i in range(len(truths) - 1):

            if truths[i] != truths[i + 1]:

                flag = True

                break

        user_flag[user_id] = flag



    total_auc = 0.0

    size = 0.0

    for user_id in user_flag:

        if user_flag[user_id]:

            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))

            total_auc += auc 

            size += 1.0

    user_auc = float(total_auc)/size

    return user_auc


# In[2]:


ONLINE = True
DEBUG = False
TARGETS = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]


# # base data

# In[3]:


y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

max_day = 15


## 读取训练集
DATASET_PATH = "../../data/wedata/wechat_algo_data1"
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
train = pd.read_csv(USER_ACTION)

print(train.shape)
for y in y_list:
    print(y, train[y].mean())
## 读取测试集
TEST_FILE1 = os.path.join(DATASET_PATH, "test_a.csv")
TEST_FILE2 = os.path.join(DATASET_PATH, "test_b.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
test1 = pd.read_csv(TEST_FILE1)
test2 = pd.read_csv(TEST_FILE2)
test = pd.concat([test2])
print('测试集大小：',test.shape)
test['date_'] = max_day
print(test.shape)

## 合并处理
df = pd.concat([train, test], axis=0, ignore_index=True)
print(df.head(3))

## 读取视频信息表

feed_info = pd.read_csv(FEED_INFO)

## 此份baseline只保留这三列

feed_info = feed_info[[

    'feedid', 'authorid', 'videoplayseconds','bgm_song_id','bgm_singer_id'

]]


# In[4]:


play_cols = [

    'is_finish', 'play_times', 'play', 'stay'

]
if not os.path.exists('user_data/baseline_data.pkl'):
    print('baseline data 不存在，开始提特征。。。')
    df = df.merge(feed_info, on='feedid', how='left')
    ## 视频时长是秒，转换成毫秒，才能与play、stay做运算

    df['videoplayseconds'] *= 1000

    ## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）

    df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')

    df['play_times'] = df['play'] / df['videoplayseconds']

    play_cols = [

        'is_finish', 'play_times', 'play', 'stay'

    ]



    ## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）

    n_day = 5

    for stat_cols in tqdm([

        ['userid'],

        ['feedid'],

        ['authorid'],

        ['userid', 'authorid'],

       ['bgm_song_id'],

       ['bgm_singer_id'],

    ]):

        f = '_'.join(stat_cols)

        stat_df = pd.DataFrame()

        for target_day in range(2, max_day + 1):

            left, right = max(target_day - n_day, 1), target_day - 1



            tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)

            tmp['date_'] = target_day



            tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count') #n 天的记录数



            g = tmp.groupby(stat_cols) # 

            tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean') #完成率



            feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)] # 记录总数、完成率


            #
            for x in play_cols[1:]:

                for stat in ['max', 'mean']:

                    tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)

                    feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))



            for y in y_list[:4]:

                tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')

                tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')

                feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])



            tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)

            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)

            del g, tmp

        df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')

        del stat_df

        gc.collect()





    ## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

    for f in tqdm(['userid', 'feedid', 'authorid',]):

        df[f + '_count'] = df[f].map(df[f].value_counts())

    for f1, f2 in tqdm([

        ['userid', 'feedid'],

        ['userid', 'authorid']

    ]):

        df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')

        df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

    for f1, f2 in tqdm([

        ['userid', 'authorid']

    ]):

        df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')

        df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)

        df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

    df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')

    df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')

    df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')
    df = reduce_mem(df,df.columns)
    #去掉第一天数据
#    df = df[df['date_']>1]
    savePkl(df,'user_data/baseline_data.pkl')
    ## 内存够用的不需要做这一步
else :
    print('读取baseline_data.pkl...')
    df = loadPkl('user_data/baseline_data.pkl')


# # load data

# In[5]:


def readValidData(hist_days=5):
    valid_date = 14
    valid_data = loadPkl('user_data/Date_%s/%sdays_feature.pkl' %
                         (valid_date, hist_days))
    return valid_data

def readTrainData(hist_days=5):
    train_dates = list(range(2,16))
    train_data = []
    for d in tqdm(train_dates):
        train_data.append(
            loadPkl('user_data/Date_%s/%sdays_feature.pkl' %
                    (d, hist_days)))
    train_data = pd.concat(train_data)
    return train_data

def readTestData(hist_days=5):
    test_date = 15
    test_data = loadPkl('user_data/Date_%s/%sdays_feature.pkl' %
                        (test_date, hist_days))
    return test_data



my_data = readTrainData()


black = [
    'date_','description', 'ocr', 'asr', 'manual_keyword_list', 'machine_keyword_list',
    'manual_tag_list', 'machine_tag_list', 'description_char', 'ocr_char',
    'asr_char', 'play_times', 'play', 'stay', 'is_finish'
] + TARGETS

cate_cols =[ x for x in my_data.columns if x in [
    'userid', 'feedid', 'device', 'authorid',
]]


if DEBUG:
    df = pd.concat([df.head(1000),df.tail(1000),df[df['date_']==14].head(1000)])
    
add_columns = []
all_cols = set(my_data.columns)
for k in [
       'max_date_', 'pooled', 'list_id', ]:
    for col in my_data.columns:
        if k in col and col not in black:
            add_columns.append(col)
add_columns = list(set(add_columns))
print(list(set(add_columns)))
print('拼接前：', df.shape, len(add_columns))
my_data_cols = list(
   set(['feedid', 'userid', 'date_', 'device']) | set(add_columns)
   | set(cate_cols) -
   set(['authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']))
df = df.merge(my_data[my_data_cols],
             on=['feedid', 'userid', 'date_', 'device'],
             how='left')
print('拼接完成后：', df.shape)
print('df 添加特征完毕！')



# # 添加embedding

# In[10]:


feed_embeddings = loadPkl('user_data/feedembedings.pkl')
for col in ['description','ocr','asr','manual_tag_list', 'machine_keyword_list', 'manual_keyword_list']:
    save_file = 'user_data/w2v/%s_embeddings.pkl' % (col)
    embedings = loadPkl(save_file)
    df = df.merge(embedings,on=['feedid'],how='left')
df = df.merge(feed_embeddings,on=['feedid'],how='left')
print('df 加载embedding 完毕！')

# # 添加svd

# In[11]:


svd_embedding = loadPkl('user_data/svd_userid_feedid_embedding.pkl')
df  = df.merge(svd_embedding,on = ['userid'],how='left')

svd_embedding = loadPkl('user_data/svd_userid_authorid_embedding.pkl')
df  = df.merge(svd_embedding,on = ['userid'],how='left')

# for c in ['manual_tag_list', 'machine_keyword_list', 'manual_keyword_list']:
#     file_name = 'user_data/'+'svd_userid_'+ c + '_embedding'+'.pkl'
#     svd_embedding = loadPkl(file_name)
#     df  = df.merge(svd_embedding,on = ['userid'],how='left')

# # 数据划分

# In[12]:


df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])


# In[13]:

logger.info('begain:')
train = df[~df['read_comment'].isna()].reset_index(drop=True)

test = df[df['read_comment'].isna()].reset_index(drop=True)
print('ceshiji shape',test.shape)
logger.info('ceshiji %s' % (test.shape[0]))
cols = [f for f in df.columns if f not in ['date_'] + play_cols + y_list]
logger.info(train[cols].shape)
trn_x = train[train['date_'] < 14].reset_index(drop=True)
val_x = train[train['date_'] == 14].reset_index(drop=True)


# # run offline

# In[14]:
makedirs('../results')
makedirs('../models')
##################### 线下验证 #####################
for i in range(10):

    logger.info('开始训练%s折' % (i))
    uauc_list = []
    r_list = []

    feature_importance_df = pd.DataFrame()

    feature_importance_df["Feature"] = cols

    for y in y_list[:4]:

        print('=========', y, '=========')

        t = time.time()

        clf = LGBMClassifier(learning_rate=0.04,
                             n_estimators=5000,
                             num_leaves=100,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             random_state=i,
                             metric='None')

        clf.fit(trn_x[cols],
                trn_x[y],
                eval_set=[(val_x[cols], val_x[y])],
                eval_metric='auc',
                early_stopping_rounds=100,
                verbose=50)

        feature_importance_df[y] = clf.feature_importances_

        val_x[y + '_score'] = clf.predict_proba(val_x[cols])[:, 1]

        val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])

        uauc_list.append(val_uauc)

        logger.info(val_uauc)

        r_list.append(clf.best_iteration_)

        logger.info('runtime: {}\n'.format(time.time() - t))

    weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[
        2] + 0.1 * uauc_list[3]
    feature_importance_df.to_csv('user_data/feature_importance_df_%s' %
                                 (weighted_uauc))
    logger.info(uauc_list)
    logger.info(weighted_uauc)

    # # run online

    # In[16]:

    ##################### 全量训练 #####################
    if ONLINE:

        r_dict = dict(zip(y_list[:4], r_list))

        for y in y_list[:4]:

            print('=========', y, '=========')

            t = time.time()

            clf = LGBMClassifier(learning_rate=0.04,
                                 n_estimators=r_dict[y],
                                 num_leaves=100,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 random_state=i)
            clf.fit(train[cols],
                    train[y],
                    eval_set=[(train[cols], train[y])],
                    early_stopping_rounds=r_dict[y],
                    verbose=100)
            savePkl(clf, '../models/model_%s_seed%s.pkl' % (y, i))
            test[y] = clf.predict_proba(test[cols])[:, 1]
            print('runtime: {}\n'.format(time.time() - t))

        test[['userid', 'feedid'] +
             y_list[:4]].to_csv('../results/sub_%.6f_%.6f_%.6f_%.6f_%.6f.csv' %
                                (weighted_uauc, uauc_list[0], uauc_list[1],
                                 uauc_list[2], uauc_list[3]),
                                index=False)