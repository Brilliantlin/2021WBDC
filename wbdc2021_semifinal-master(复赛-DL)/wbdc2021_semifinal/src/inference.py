import sys
import os
import torch
import pandas as pd
import numpy as np
from deepctr_torch.inputs import build_input_features,get_feature_names
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
import pickle
import time
import sys
import os
import gc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../config'))
sys.path.append(os.path.join(BASE_DIR,'src3/train/'))
sys.path.append(os.path.join(BASE_DIR,'src3/model/'))

# sys.path.append('../config')
from deepfm_batch import MyDeepFM
from config_inger import MyArgparse
from Mytools import init_logger,seed_everything,save_pickle,load_pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, help='Source language')
parser_args = parser.parse_args()


args = MyArgparse()

logger = init_logger('test_log.txt')


class CustomDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,data,action,batch_size,mode = 'train'):
        # TODO
        # 1. Initialize file path or list of file names.
        self.data = data
        self.data.index = range(len(self.data)) 
        self.aciton = action
        self.mode = mode
        self.batch_size = batch_size
        self.indexes = np.arange(self.data.shape[0])
        if mode=='train':
            np.random.shuffle(self.indexes)
        self.graph_emb8 = load_pickle(args.emb_all_path+'graph_walk_emb_16.pkl')
        self.feed_emb_16 = load_pickle(args.emb_all_path+'feed_embeddings_16.pkl')
        self.weight_emb8 = load_pickle(args.emb_all_path+'user_weight_emd_16.pkl')
        self.weight_emb8 = self.weight_emb8.drop('user_date_weight_emd',axis = 1)
        self.keyword_w2v_8 = load_pickle(args.emb_all_path+'keyword_w2v_8.pkl')
        self.userid_feedid_d2v_all_16 = load_pickle(args.emb_all_path+'userid_feedid_d2v_all_16.pkl')##加了初赛数据
        self.all_text_data_v8 = load_pickle(args.emb_all_path+'all_text_data_v8.pkl')
        self.userid_authorid_d2v_all_16 = load_pickle(args.emb_all_path+'userid_authorid_d2v_all_16.pkl')
        self.feed_info = pd.read_csv(args.FEED_INFO)
        self.feed_info[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
        self.feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = self.feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
        self.dense_features = ['videoplayseconds']
        self.feed_info[self.dense_features] = np.log(self.feed_info[self.dense_features] + 1.0)
        self.feature_names = load_pickle(args.DATASET_PATH+'feature_names_batch.pkl')
        self.feature_index = load_pickle(args.DATASET_PATH+'feature_index_batch.pkl')
    def __getitem__(self, index):
        # TODO
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) *
                                    self.batch_size]
        x = self.data.loc[batch_indexs]
        if self.mode=='train':
            label = self.getLabelData(x)
            return self.mergeFeat(x),label
        else:
            return self.mergeFeat(x)
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return (self.data.shape[0] // self.batch_size) + 1
    def getLabelData(self,batch):
        y = batch[[self.aciton]].values
        y_torch_data =torch.from_numpy(y)
        return y_torch_data 
    def mergeFeat(self,batch):
        data = batch.copy()
        data.index =  range(len(data))
        df = data.merge(self.feed_info[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                      on='feedid')
        df = df.merge(self.graph_emb8, how='left',
                      on='userid')
        df = df.merge(self.feed_emb_16, how='left',
                      on='feedid')
        df = df.merge(self.weight_emb8, how='left',
                      on='userid')
        df = df.merge(self.keyword_w2v_8, how='left',
                      on='feedid')
        df = df.merge(self.userid_feedid_d2v_all_16, how='left',
                      on='userid')
        df = df.merge(self.all_text_data_v8, how='left',
                      on='feedid')
        df = df.merge(self.userid_authorid_d2v_all_16, how='left',
                      on='userid')
        IDS = ['userid',
            'feedid',
            'authorid',
            'bgm_song_id' ,
            'bgm_singer_id']

    #     encoder
        for col in IDS:
            lbe = load_pickle(os.path.join(args.encoder_all_path,'ID_col_%s_encoder.pkl' % (col)))
            df[col] = lbe.transform(df[col])

         ##转为torch data
        test_model_input = {name: df[name] for name in self.feature_names}
        x = [test_model_input[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        x_torch_data = torch.from_numpy(
                np.concatenate(x, axis=-1))
        return x_torch_data

    

def getLabelData(data,action):
    y = data[[action]].values
    y_torch_data =torch.from_numpy(y)
    return y_torch_data  



def getTrainLoader(data_path,aciton,batch_size,num_workers,mode = 'train_all'):
    def collate_fn(batch):
        label = batch[0][1]
        x = batch[0][0]
        return x,label
    data = load_pickle(data_path)
    if mode=='train_base':
        data = data[data.date_<14]
    dataset = CustomDataset(data,aciton,batch_size,mode = 'train')
    
    sampler = RandomSampler(dataset)
    train_loader = DataLoader(
            dataset=dataset, sampler=sampler, batch_size=1,num_workers = num_workers,collate_fn =collate_fn)
    return train_loader

def getTestLoader(data_path,action,batch_size,num_workers,mode = 'test'):
    def collate_fn(batch):
        x= batch[0]
        return x.squeeze(0)
     
    if mode=='test':
        data = pd.read_csv(data_path)
        dataset = CustomDataset(data,action,batch_size,mode = 'test')
    else:
        data = load_pickle(data_path)
        dataset =CustomDataset(data,action,batch_size,mode = 'val')
    test_sampler = SequentialSampler(dataset)
    test_loader = DataLoader(
                dataset=dataset, sampler=test_sampler, batch_size=1,num_workers = num_workers,collate_fn =collate_fn)
    return test_loader

def get_fixlen_feature_columns():
    sparse_features = ['userid',
                    'feedid',
                    'authorid',
                    'bgm_song_id' ,
                    'bgm_singer_id']
    user_action_all = load_pickle('../../data/wedata/wechat_algo_data2/user_action_all.pkl')
    feed_info = pd.read_csv(args.FEED_INFO)
    feed_info[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)

    IDS = {
        'userid':user_action_all.userid.unique().tolist() ,
        'feedid':feed_info.feedid.unique().tolist(),
        'authorid':feed_info.authorid.unique().tolist(),
        'bgm_song_id':feed_info.bgm_song_id.unique().tolist(),
        'bgm_singer_id':feed_info.bgm_singer_id.unique().tolist(),
    }
    graph_emb8 = load_pickle(args.emb_all_path+'graph_walk_emb_16.pkl')
    dense_features = graph_emb8.columns.tolist()

    feed_emb_16 = load_pickle(args.emb_all_path+'feed_embeddings_16.pkl')
    dense_features =dense_features+feed_emb_16.columns.tolist()

    weight_emb8 = load_pickle(args.emb_all_path+'user_weight_emd_16.pkl')
    weight_emb8 = weight_emb8.drop('user_date_weight_emd',axis = 1)
    dense_features =dense_features+weight_emb8.columns.tolist()

    keyword_w2v_8 = load_pickle(args.emb_all_path+'keyword_w2v_8.pkl')
    dense_features =dense_features+keyword_w2v_8.columns.tolist()

    userid_feedid_d2v_all_16 = load_pickle(args.emb_all_path+'userid_feedid_d2v_all_16.pkl')##加了初赛数据
    dense_features =dense_features+userid_feedid_d2v_all_16.columns.tolist()

    all_text_data_v8 = load_pickle(args.emb_all_path+'all_text_data_v8.pkl')
    dense_features =dense_features+all_text_data_v8.columns.tolist()

    userid_authorid_d2v_all_16 = load_pickle(args.emb_all_path+'userid_authorid_d2v_all_16.pkl')
    dense_features =dense_features+userid_authorid_d2v_all_16.columns.tolist()


    dense_features =dense_features+["videoplayseconds"]
    dense_features = [i for i in dense_features if i not in sparse_features]

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=len(IDS[feat]) + 1, embedding_dim=16)
                                  for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]
     
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
    save_pickle(fixlen_feature_columns,args.DATASET_PATH+'fixlen_feature_columns_batch.pkl')
    save_pickle(feature_names,args.DATASET_PATH+'feature_names_batch.pkl')
    save_pickle(feature_index,args.DATASET_PATH+'feature_index_batch.pkl')
    return fixlen_feature_columns




args.num_train_epochs = 10
args.model_name = 'deepfm_batch_l2_1e1'
batch_size =4096
num_workers = 14
sta_time = time.time()


args.TEST_FILE = parser_args.test_file
submit1 = pd.read_csv(args.TEST_FILE)[['userid', 'feedid']]
test_loader = getTestLoader(args.TEST_FILE,'like',batch_size*20,num_workers)
ACTION_LIST =["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]

for action in ACTION_LIST:
    
    fixlen_feature_columns = load_pickle(args.DATASET_PATH+'fixlen_feature_columns_batch.pkl')
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    
#     test_loader = getTestLoader(args.TEST_FILE,action,batch_size*20,num_workers)
    
 
    
    model = MyDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                     seed = 2021,
                    l2_reg_embedding=1e-1, 
                     device=device,
                     gpus=[0, 1])

    model.compile("adagrad", "binary_crossentropy", metrics=["uAUC"])
    model_save_path = args.model_batch_path+args.model_name+'/{}_best.pth'.format(action)
    
    import collections
    x = torch.load(model_save_path)
    y = collections.OrderedDict()
    for key in x.keys():
        y[key[7:]] = x[key]
    model.load_state_dict(y)
    del x,y
    gc.collect()
  
    pred_ans = model.predict(test_loader)
    submit1[action] = pred_ans
    torch.cuda.empty_cache()
    del model
    gc.collect()
end_time = time.time()

logger.info('预测时间为：{}'.format(end_time-sta_time))


# 保存提交文件
import time
time_str = time.strftime('%d_%H_%M_%S', time.localtime(time.time())) 
# submit1.to_csv(args.submit_path+"test.csv".format(time_str), index=False)
submit2 = pd.read_csv(args.submit_path + 'share_bottom.csv')

def rankFusion(file1,file2,out):
    res1 = file1.sort_values(['userid','feedid'])
    res1 = getRankPred(res1)
    
    res2 = file2.sort_values(['userid','feedid'])
    res2 = getRankPred(res2)
    res = res1.copy()
    y_list = res1.columns.tolist()[2:]
    for col in y_list:
        weight = weight_dict[col]
        res[col] = weight*res1[col]+(1-weight)*res2[col]
    res.to_csv(out,index = None)
    return res

def getRankPred(re):
    res = re.copy()
    y_list = res.columns.tolist()[2:]
    for col in y_list:
        res[col+'_rank'] = res[col].rank()
        mi,ma = res[col+'_rank'].min(),res[col+'_rank'].max()   
        res[col] = res[col+'_rank'].apply(lambda x:(ma-x)/(ma - mi))
        res[col] = 1 - res[col]
        res = res.drop(col+'_rank',axis = 1)
    return res 

weight_dict = {'read_comment':0.5,
 'like':0.5,
 'click_avatar':0.5,
 'forward':0.5,
 'comment':0.5,
 'follow':0.5,
 'favorite':0.5}


rankFusion(submit1,submit2,args.submit_path + 'result.csv')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('test file', help='测试集路径 绝对路径')