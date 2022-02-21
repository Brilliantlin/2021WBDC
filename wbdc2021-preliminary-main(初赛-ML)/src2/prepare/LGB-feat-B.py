#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('..')
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers = 32)
import numpy as np
import gc
from mytools.utils.myfile import makedirs
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
import os
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import collections
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from tqdm import tqdm as tqdm
# 存储数据的根目录
DATASET_PATH = "../../data/wedata/wechat_algo_data1"
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE1 = os.path.join(DATASET_PATH, "test_a.csv")
TEST_FILE2 = os.path.join(DATASET_PATH, "test_b.csv")


# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
TARGETS = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]

#参数配置
DEBUG = False
END_DAY = 15
FEATDAYS = 5
SEED = 2021
PAD,UNK = 0,-1
# 历史行为特征
HIST_FEAT = ['feedid'] 
# feed 特征
FEA_feed = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id',]
SUBMIT = os.path.join(DATASET_PATH, "submit_demo_初赛a.csv")
makedirs('user_data')

# In[2]:


import pandas as pd
import lightgbm as lgb
import numpy as np
import multiprocessing
from collections import Counter

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.model_selection import KFold
import gc
from sklearn import preprocessing
from scipy.stats import entropy
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer, roc_auc_score,classification_report,f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import datetime
import time
from itertools import product
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle 
# from feature_selector import FeatureSelector
warnings.filterwarnings('ignore')


# # function

# In[3]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
def savePkl(config, filepath):
    f = open(filepath, 'wb')
    pickle.dump(config, f)
    f.close()

def loadPkl(filepath):
    f = open(filepath, 'rb')
    config = pickle.load(f)
    return config


# ## encoder

# In[4]:


class myencoder():
    def __init__(self,):
        self.w2i = {}
    def fit(self,inputs):
        inputs = list(set(inputs) - set([PAD])) 
        self.w2i = dict(zip(inputs,range(1,len(inputs))))
        self.w2i.update({PAD:0})
    def transform(self,inputs):
        if not isinstance(inputs,pd.Series):
            inputs = pd.Series(inputs)
        res = inputs.apply(lambda x:self.w2i[x] if x in self.w2i else self.w2i[PAD])
#         res = [self.w2i[x] if x in self.w2i else self.w2i[PAD] for x in inputs]
        return np.array(res)
    def fit_transform(self,inputs):
        self.fit(inputs)
        return self.transform(inputs)


# In[5]:


def Xy_concat(X1,X2,y1,y2):
    assert len(set(X1.keys() - set(X2.keys()))) ==0
    for k in X1.keys():
        X1[k] = np.concatenate([X1[k],X2[k]],0)
    
    y1 = y1.append(y2)
    return X1,y1
def getencodedlabel(s):
    '''input pd.Series'''
    l_encoder = LabelEncoder()
    return l_encoder.fit_transform(s) + 1,l_encoder

    


def getFeedEmbedings(feed_embeddings):
    '''input feedid_encode '''
    embedings = feed_embeddings.sort_values('feedid_encode')
    embedings = embedings.feed_embedding.parallel_apply(lambda x:[float(i) for i in x.split()])
    embedings = np.array(embedings.to_list())
    embedings = np.concatenate([embedings,np.array([[0] * embedings.shape[1]])],0)
    return embedings

def get_random_list(seed):
    np.random.seed(seed)
    random_seeds = np.random.randint(0, 10000, size=5)
    print('Gnerate random seeds:', random_seeds)
    return random_seeds


def getfeedembedings():
    '''获取feedembeding'''
    feed_embeddings = pd.read_csv(FEED_EMBEDDINGS)
    feed2id = dict(zip(feed_embeddings.feedid,range(feed_embeddings.shape[0])))
    embedings = feed_embeddings['feed_embedding'].apply(lambda x:np.array([float(i) for i in x.split()]))
    return feed2id,np.array(embedings.values.tolist())


# #  read data

# In[6]:


# 测试集
submit = pd.read_csv(SUBMIT)
user_action = pd.read_csv(USER_ACTION) if not DEBUG else pd.read_csv(USER_ACTION).groupby('date_').head(1000)
test1 = pd.read_csv(TEST_FILE1)
test2 = pd.read_csv(TEST_FILE2)
test = pd.concat([test2])
print('训练集大小：',user_action.shape,user_action.date_.nunique())
print('测试集大小：',test.shape)
feed_info = pd.read_csv(FEED_INFO)
feed_embeddings = pd.read_csv(FEED_EMBEDDINGS)     
# 存储各种ID信息全部
IDS = {
    'userid':user_action.userid.unique().tolist() + [-1],
    'feedid':feed_info.feedid.unique().tolist() + [-1],
    'authorid':feed_info.authorid.unique().tolist() + [-1],
    'bgm_song_id':feed_info.bgm_song_id.unique().tolist() + [-1],
    'bgm_singer_id':feed_info.bgm_singer_id.unique().tolist() + [-1],
}
# 
VAR_IDS = {}
DENSE = ['videoplayseconds']


# # PCA  embedding
# 切分
from sklearn.decomposition import PCA
feed_embeddings = pd.read_csv(FEED_EMBEDDINGS) 
feed_embeddings['feed_embedding'] = feed_embeddings['feed_embedding'].apply(lambda x:np.array([float(i) for i in x.split()]))
x = np.array(feed_embeddings['feed_embedding'].tolist())
#pca
pca = PCA(n_components=32,svd_solver='full')
new_x = pca.fit_transform(x)
print(pca.explained_variance_ratio_.sum())
new_x_ = pd.DataFrame(new_x)
new_x_.columns = ['pca_embedding_' + str(x) for x in new_x_.columns]
feed_embeddings = pd.concat([feed_embeddings[['feedid']],new_x_],axis=1)
savePkl(feed_embeddings,'user_data/feedembedings.pkl')





feed_embeddings_dict = dict(zip(feed_embeddings['feedid'],new_x))





feed_info['videoplayseconds'] *= 1000
feed_info.fillna(UNK,inplace = True)


# # get feat


user_action = reduce_mem_usage(user_action)
feed_embeddings = reduce_mem_usage(feed_embeddings)


# In[11]:


def myPivot(feat_feild,index,values, aggfunc):
    t = feat_feild.pivot_table(index=index, values=values, aggfunc=aggfunc)
    columns = ['_'.join(index)+ '_' + fun_name + '_' + v for fun_name,v in t.columns ]
    print(columns)
    t.columns = columns
    t = t.reset_index()
    return t, columns

def getTagrate(x,c,user_tag_item):
    userid = x.userid
    if x[c]==-1:
        return 0
    tags = x[c].split(';')
    score = 0 
    if tags[0]!=-1:
        for tag in tags:
            tag = float(tag)
            if (userid,tag) in user_tag_item :
                score += user_tag_item[(userid,tag)]
            else:
                score += 0
        score = np.mean(score)
        return score
    else :
        return -1


# In[12]:


use_col = ['userid', 'feedid', 'device'] + ACTION_LIST 


user_action = user_action.merge(feed_info,on='feedid',how='left')
test = test.merge(feed_info,on='feedid',how='left')


test['date_'] = 15
user_action = pd.concat([user_action,test])


# # SVD 特征

# In[15]:


file_name = 'user_data/svd_userid_feedid_embedding.pkl'
if not os.path.exists(file_name):
    tmp = user_action[['feedid','userid']]
    tmp['feedid'] = tmp['feedid'].astype(str)
    t,cols = myPivot(tmp,index=['userid'],values=['feedid'],aggfunc=[list])
    t['userid_list_feedid'] = t['userid_list_feedid'].apply(lambda x:' '.join(x))

    user_feed_dim = 32
    tfidf_clf = TfidfVectorizer(ngram_range=(1, 3), min_df=0.01, max_df=0.99)
    tfidf_vector = tfidf_clf.fit_transform(t['userid_list_feedid'].tolist())
    print(tfidf_vector.shape)
    svd = TruncatedSVD(n_components=user_feed_dim, n_iter=7, random_state=42)
    svd_vector = svd.fit_transform(tfidf_vector)
    print(svd.explained_variance_ratio_.sum())
    svd_embedding = pd.DataFrame(svd_vector,
                                 index=t['userid'],
                                 columns=[
                                     'svd_userid_feedid_embedding' + '_' + str(x)
                                     for x in range(user_feed_dim)
                                 ]).reset_index()
    savePkl(svd_embedding,file_name)


# In[16]:


file_name = 'user_data/svd_userid_authorid_embedding.pkl'
if not os.path.exists(file_name):
    tmp = user_action[['authorid','userid']]
    tmp['authorid'] = tmp['authorid'].astype(str)
    t,cols = myPivot(tmp,index=['userid'],values=['authorid'],aggfunc=[list])
    t['userid_list_authorid'] = t['userid_list_authorid'].apply(lambda x:' '.join(x))

    user_author_dim = 16
    tfidf_clf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.01, max_df=0.99)
    tfidf_vector = tfidf_clf.fit_transform(t['userid_list_authorid'].tolist())
    print(tfidf_vector.shape)
    svd = TruncatedSVD(n_components=user_author_dim, n_iter=7, random_state=42)
    svd_vector = svd.fit_transform(tfidf_vector)
    print(svd.explained_variance_ratio_.sum())
    svd_embedding = pd.DataFrame(svd_vector,
                                 index=t['userid'],
                                 columns=[
                                     'svd_userid_authorid_embedding' + '_' + str(x)
                                     for x in range(user_author_dim)
                                 ]).reset_index()
    savePkl(svd_embedding,file_name)


# 标签

# In[17]:



for c in [
        'manual_tag_list', 'machine_keyword_list', 'manual_keyword_list'
]:
    file_name = 'user_data/'+'svd_userid_'+ c + '_embedding'+'.pkl'
    if os.path.exists(file_name):
        continue
    dim = 16
    tmp = user_action
    split_ = tmp[c].str.split(';', expand=True)  # 切分当前特征列
    split_.columns = [c + str(x) for x in split_.columns]
    split_ = split_.astype(np.float32)
    tmp = pd.concat([tmp, split_], axis=1)  #拼接
    t = tmp[['userid', 'feedid'] +
                    split_.columns.tolist()].set_index(
                        ['userid', 'feedid'])
    t = t.stack(0, dropna=True).reset_index()
    t[c] = t[0].astype(str)
    t,cols = myPivot(t,index=['userid'],values=[c],aggfunc=[list])
    t[cols[0]] = t[cols[0]].apply(lambda x:' '.join(x))
    

    tfidf_clf = TfidfVectorizer(ngram_range=(1, 1), min_df=0.01, max_df=0.99)
    tfidf_vector = tfidf_clf.fit_transform(t[cols[0]].tolist())
    print(tfidf_vector.shape)
    svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
    svd_vector = svd.fit_transform(tfidf_vector)
    print(svd.explained_variance_ratio_.sum())
    svd_embedding = pd.DataFrame(svd_vector,
                                 index=t['userid'],
                                 columns=[
                                     'svd_userid_'+ c + '_embedding' + '_' + str(x)
                                     for x in range(dim)
                                 ]).reset_index()
    savePkl(svd_embedding,file_name)
#     break


# 文本

# In[18]:


if not os.path.exists('user_data/texts_svd_embedding.pkl'):
    text_dim = 16
    tmp = feed_info[['feedid','description','ocr','asr']].astype(str)
    texts  =tmp.apply(lambda x: x.description + ' # ' + x.ocr + ' # '+ x.asr ,axis=1).tolist()

    tfidf_clf = TfidfVectorizer(ngram_range=(1,3),min_df=0.05,max_df=0.99)
    tfidf_vector = tfidf_clf.fit_transform(texts)

    svd = TruncatedSVD(n_components=text_dim, n_iter=7, random_state=42)
    svd_vector = svd.fit_transform(tfidf_vector)

    svd_embedding = pd.DataFrame(svd_vector,index=tmp.feedid,columns=['svd_text_embedding' + '_' + str(x) for x in range(text_dim)]).reset_index()

    savePkl(svd_embedding,'user_data/texts_svd_embedding.pkl')


# # w2v 特征

# In[19]:


#标签
makedirs('user_data/w2v')
for col in ['manual_tag_list', 'machine_keyword_list', 'manual_keyword_list']:
    print('w2ving %s' %(col))
    save_file = 'user_data/w2v/%s_embeddings.pkl' % (col)
    if os.path.exists(save_file):
        continue
    else:
        texts = feed_info[col].str.split(';').tolist()
    texts = [x if isinstance(x, list) else [str(x)] for x in texts]
    w2v = Word2Vec(
        sentences=texts,
        size = 8,
    )  # 训练word2vec模型
    tmp = []
    for x in tqdm(texts):
        v1 = np.sum([w2v[i] for i in x if i in w2v], 0)
        v2 = np.mean([w2v[i] for i in x if i in w2v], 0)
        v = np.append(v1,v2)
        if isinstance(v, np.float64):
            tmp.append(np.zeros(w2v.vector_size*2))
        else:
            tmp.append(v)
#     pca = PCA(n_components=32,svd_solver='full')
#     tmp = pca.fit_transform(tmp)
    w2v_embedings = pd.DataFrame(tmp,columns=[col + '_embedding' + str(i) for i in range(w2v.vector_size*2)])
    w2v_embedings['feedid'] = feed_info['feedid']
    savePkl(w2v_embedings,save_file)


# In[20]:


# 文本
texts_dict = {}
texts = []
for col in ['description','ocr','asr']:
    t = feed_info[col].str.split('').tolist()
    t = [x if isinstance(x, list) else [str(x)] for x in t]
    texts_dict[col] = t
    texts += t
# 转换为二维 ，处理单值
w2v = Word2Vec(
        sentences=texts,
        size = 12
    )  # 训练word2vec模型
for col,texts in texts_dict.items(): 
    save_file = 'user_data/w2v/%s_embeddings.pkl' % (col)
    if os.path.exists(save_file):
        continue
    tmp = []
    for x in tqdm(texts):
        v1 = np.sum([w2v[i] for i in x if i in w2v], 0)
        v2 = np.mean([w2v[i] for i in x if i in w2v], 0)
        v = np.append(v1,v2)
        if isinstance(v, np.float64):
            tmp.append(np.zeros(w2v.vector_size*2))
        else:
            tmp.append(v)
    
    w2v_embedings = pd.DataFrame(tmp,columns=[col + '_embedding' + str(i) for i in range(w2v.vector_size*2)])
    w2v_embedings['feedid'] = feed_info['feedid']
    savePkl(w2v_embedings,save_file)


# # debug 

# In[21]:


user_action['is_finish'] = (user_action['play'] >= user_action['videoplayseconds']*0.95).astype('int8')
user_action['play_times'] = user_action['play'] / user_action['videoplayseconds']
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]


# In[22]:


makedirs('user_data/hist_data')


# In[23]:


def runFeat(begain, end, user_action, ACTION_LIST, feed_embeddings_dict,
            use_col):
    hist_days = 5
    for label_feild_date in list(range(begain, end)):
        file_name = 'user_data/Date_%s/%sdays_feature.pkl' % (
            label_feild_date, hist_days)
        makedirs('user_data/Date_%s/' % (label_feild_date))
        feat_feild_date = list(
            range(max(label_feild_date - hist_days, 1), label_feild_date))
        label_feild = user_action[user_action.date_ == label_feild_date]

        feat_feild = user_action[user_action.date_.isin(feat_feild_date)]
        feat_feild = feat_feild[feat_feild.userid.isin(label_feild.userid)]
        feat_feild = feat_feild.drop_duplicates(['userid', 'feedid'] +
                                                ACTION_LIST,
                                                keep='last')


        #最后一次操作，距离现在是几天
        for f in [['userid'], ['feedid'], ['authorid'], ['bgm_song_id'],
                  ['bgm_singer_id'], ['userid', 'authorid']]:
            for target in TARGETS:
                t, cols = myPivot(feat_feild[feat_feild[target] == 1],
                                  index=f,
                                  values=['date_'],
                                  aggfunc=['max'])
                t[cols] = label_feild_date - t[cols]
                cols = [x + target for x in cols]
                t.columns = f + cols
                print(cols)
                label_feild = label_feild.merge(t.fillna(-1), on=f,
                                                how='left')  # 没有记录 补-1
                use_col += cols
        # 历史序列feedid mean embedding
        t = feat_feild.groupby('userid').apply(lambda x: np.array(
            [feed_embeddings_dict[i] for i in x.feedid]).mean(axis=0).tolist())
        t = pd.DataFrame(list(t.values), index=t.index)
        t.columns = ['feedid_pooled_embeding_' + str(i) for i in t.columns]
        print(list(t.columns))
        use_col += list(t.columns)
        feedid_pooled_embeding = t.reset_index()
        label_feild = label_feild.merge(feedid_pooled_embeding,
                                        on=['userid'],
                                        how='left')
        # 作者多少个视频被曝光
        t, cols = myPivot(feat_feild,
                          index=['authorid'],
                          values=['feedid'],
                          aggfunc=['nunique'])
        label_feild = label_feild.merge(t, on=['authorid'],
                                        how='left').drop_duplicates()
        use_col += cols
        #tag keyword 转化率
        for c in [
                'manual_tag_list', 'machine_keyword_list',
                'manual_keyword_list'
        ]:
            split_ = feat_feild[c].str.split(';', expand=True)  # 切分当前特征列
            split_.columns = [c + str(x) for x in split_.columns]
            split_ = split_.astype(np.float32)
            feat_feild_ = pd.concat([feat_feild, split_], axis=1)  #拼接
            t = feat_feild_[['userid', 'feedid'] +
                            split_.columns.tolist()].set_index(
                                ['userid', 'feedid'])
            t = t.stack(0, dropna=True).reset_index()
            t = t.merge(feat_feild[[
                'userid',
                'feedid',
            ] + ACTION_LIST],
                        how='left',
                        on=['userid', 'feedid'])[[
                            'userid',
                            'feedid',
                            0,
                        ] + ACTION_LIST]
            t.columns = [
                'userid',
                'feedid',
                c + '_id',
            ] + ACTION_LIST

            user_transform_rate, cols = myPivot(t,
                                                index=['userid', c + '_id'],
                                                values=ACTION_LIST,
                                                aggfunc=['mean', 'sum'])
            use_col += cols
            #转化为字典
            user_transform_rate = user_transform_rate.set_index(
                ['userid', c + '_id']).to_dict()
            for col in cols:
                user_tag_item = user_transform_rate[col]
                label_feild[col] = label_feild.apply(
                    lambda x: getTagrate(x, c, user_tag_item), axis=1)
        del feat_feild_
        del feat_feild
        gc.collect()
        label_feild = label_feild.fillna(-1)
        label_feild = reduce_mem_usage(label_feild)
        savePkl(label_feild, file_name)


for i  in tqdm(list(range(2,16))):
    runFeat(i, i+1, user_action, ACTION_LIST, feed_embeddings_dict,use_col)

