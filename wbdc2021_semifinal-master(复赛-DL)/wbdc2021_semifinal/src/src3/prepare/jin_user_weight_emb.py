import pickle
import random
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


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
seed_everything()



train = pd.read_csv('../../data/wedata/wechat_algo_data2/user_action.csv')

user_feed_date_list = train.groupby(['userid','date_']).apply(lambda x: [str(i) for i in list(x['feedid'])]).reset_index()
user_feed_date_sum = user_feed_date_list.groupby(['userid']).apply(lambda x:x['date_'].sum()).reset_index()
user_feed_date_sum.columns = ['userid','date_sum']
user_feed_date_list = user_feed_date_list.merge(user_feed_date_sum,on = 'userid')
user_feed_date_list['rate'] = user_feed_date_list['date_']/user_feed_date_list['date_sum']
emd = pd.read_csv('../../data/wedata/wechat_algo_data2/feed_embeddings.csv')
emd.index = emd.feedid
emd_dict = {}
for index in tqdm(emd.index):
    emd_dict[index] = [float(i) for i in emd.loc[int(index),'feed_embedding'].split(' ')[0:-1]]
    
def getemb(x):
    emds = []
    for i in x:
        emds.append(emd_dict[int(i)])
    emds = np.array(emds)
    emds = np.mean(emds,axis = 0)
    return list(emds)
user_feed_date_list['emd'] = user_feed_date_list[0].apply(getemb)
user_feed_date_list['emd'] = user_feed_date_list['emd'].apply(lambda x:np.array(x))
user_feed_date_list['date_emd'] = user_feed_date_list['emd']*user_feed_date_list['rate']
user_feed_date_list['emd'] = user_feed_date_list['emd'].apply(lambda x:np.array(x))
user_feed_date_list['date_emd'] = user_feed_date_list['emd']*user_feed_date_list['rate']
user_date_emd = user_feed_date_list.groupby('userid').apply(lambda x:sum(x['date_emd'])).reset_index()
user_date_emd.columns = ['userid','user_date_weight_emd']
emb = user_date_emd.copy()
save_pickle(user_date_emd,'../train/data/wechat_algo_data2/user_date_weigh_emd.pkl')

for i in range(512):
    user_date_emd['user_date_weight_emd_{}'.format(i)] = user_date_emd['user_date_weight_emd'].apply(lambda x:x[i])

from sklearn.decomposition import PCA
dim = 8
pca = PCA(n_components=dim)
pca_result = pca.fit_transform(user_date_emd[['user_date_weight_emd_{}'.format(i) for i in range(512)]].values)
for i in range(dim):
    emb['user_weight_emd_{}'.format(i)] =  pca_result[:,i]
# emb.drop('user_date_weight_emd',axis = 1,inplace = True)
save_pickle(emb,'../train/data/wechat_algo_data2/user_weight_emd_{}.pkl'.format(dim))