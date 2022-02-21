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


#emd = load_pickle('../data/wechat_algo_data1/feed_embeddings_512.pkl')
emd = pd.read_csv('../../data/wedata/wechat_algo_data2/feed_embeddings.csv')
for i in tqdm(range(512)):
    emd['emd_{}'.format(i) ] = emd['feed_embedding'].apply(lambda x:float(x.split(' ')[i]))

from sklearn.decomposition import PCA
dim = 16
pca = PCA(n_components=dim)
pca_result = pca.fit_transform(emd[['emd_{}'.format(i) for i in range(512)]].values)
emx = pd.read_csv('../../data/wedata/wechat_algo_data2/feed_embeddings.csv')
for i in range(dim):
    emx['emd_{}'.format(i)] =  pca_result[:,i]
emx.drop('feed_embedding',axis = 1,inplace = True)
if not os.path.exists('../train/data/wechat_algo_data2/'):os.makedirs('../train/data/wechat_algo_data2/')
save_pickle(emx,'../train/data/wechat_algo_data2/feed_embeddings_{}.pkl'.format(dim))
