import sys
import os
sys.path.append('./config')
from config_prosper import *
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(SRC_DIR,'src/src1/train/'))

from mytools.utils.myfile import savePkl,loadPkl
import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf

from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat,build_input_features,input_from_feature_columns

from mytools.utils.myfile import savePkl,loadPkl
from mmoe_tf import MMOE,MMOE_FefM,MMOE_mutihead,Shared_Bottom,MMOE_FefM_multihead
from evaluation import evaluate_deepctr
from tensorflow.python.keras.utils import multi_gpu_model
from tqdm import tqdm as tqdm
import warnings
import tensorflow as tf
print(tf.test.is_gpu_available())
# GPU相关设置
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# SEED = 100
def loadFeedinfo():
    feed = loadPkl(FEED_INFO_DEAL)
    feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
    feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')
    print('feedinfo loading over...')
    return feed
class myDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data: pd.DataFrame,batch_size=2048, shuffle=True,mode = 'train'):
        
        
        assert mode == 'train' or mode == 'test'
        if mode == 'test' and shuffle == True :
            raise ValueError('测试数据打乱了！')
            
        self.data = data.copy()
        self.data = self.data.reset_index(drop = True)
        self.target = ACTION_LIST
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(self.data.shape[0])
        self.feedinfo = loadFeedinfo()
        self.sparse_features = list(set(['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id' 
                  ] +  [x for x in self.feedinfo.columns if 'manual_tag_list' in x 
                  ] + [x for x in self.feedinfo.columns if 'manual_keyword_list' in x 
                  ] + [x for x in self.feedinfo.columns if 'machine_keyword_list' in x]))
        
        self.var_len_features = ['manual_tag_list', 'manual_keyword_list', 'machine_keyword_list'] 
        self.dense_features = ['videoplayseconds',]
        
        

        # dense 特征处理
#         self.data['videoplayseconds'] = self.data['videoplayseconds'].fillna(0,)
#         self.data['videoplayseconds'] = np.log(self.data['videoplayseconds'] + 1.0)
        

#         self.feed_embeddings = loadPkl(os.path.join(FEATURE_PATH,'feedembedings.pkl'))
#         self.user_feed_svd_embedding = loadPkl(os.path.join(FEATURE_PATH,'svd_userid_feedid_embedding.pkl'))
#         self.user_author_svd_embedding = loadPkl(os.path.join(FEATURE_PATH,'svd_userid_authorid_embedding.pkl'))
#         self.text_svd_embedding = loadPkl(os.path.join(FEATURE_PATH,'texts_svd_embedding.pkl'))
#         self.text_svd_embedding['feedid'] = self.text_svd_embedding['feedid'].astype(int)

        self.graph_emb8 = loadPkl(os.path.join(MODEL_PATH,'emb/graph_walk_emb_8.pkl'))
        self.feed_emb_16 = loadPkl(os.path.join(MODEL_PATH,'emb/feed_embeddings_16.pkl'))
        self.weight_emb8 = loadPkl(os.path.join(MODEL_PATH,'emb/user_weight_emd_8.pkl'))
        self.weight_emb8 = self.weight_emb8.drop('user_date_weight_emd',axis = 1)
        self.keyword_w2v_8 = loadPkl(os.path.join(MODEL_PATH,'emb/keyword_w2v_8.pkl'))
        self.userid_feedid_d2v_all_16 = loadPkl(os.path.join(MODEL_PATH,'emb/userid_feedid_d2v_all_16.pkl'))##加了初赛数据
        self.all_text_data_v8 = loadPkl(os.path.join(MODEL_PATH,'emb/all_text_data_v8.pkl'))
        self.userid_authorid_d2v_all_16 = loadPkl(os.path.join(MODEL_PATH,'emb/userid_authorid_d2v_all_16.pkl'))
        
        if mode == 'train':
            self.dnn_feature_columns = self.getFeatureColumns()
            self.feature_names = get_feature_names(self.dnn_feature_columns)
            self.feature_index = build_input_features(self.dnn_feature_columns)
            savePkl(self.dnn_feature_columns,os.path.join(MODEL_PATH,'feature_columns_all.pkl'))
            print('feature columns have saved')
        else :
            self.dnn_feature_columns = loadPkl(os.path.join(MODEL_PATH,'feature_columns_all.pkl'))
            self.feature_names = get_feature_names(self.dnn_feature_columns)
            self.feature_index = build_input_features(self.dnn_feature_columns)
            print('load feature columns' ,os.path.join(MODEL_PATH,'feature_columns_all.pkl'))
        
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)

    def __len__(self):

        return (self.data.shape[0] // self.batch_size) + 1

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) *
                                    self.batch_size]
        batch_data = self.data.iloc[batch_indexs, :]
        
        return self.get_feature_on_batch(batch_data)

    def on_epoch_end(self):
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)
    def on_epoch_begain(self):
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)

    def get_feature_on_batch(self, batch):
        
#         batch = batch.merge(self.user_feed_svd_embedding,on='userid',how='left')
#         batch = batch.merge(self.user_author_svd_embedding,on='userid',how='left')
#         batch = batch.merge(self.text_svd_embedding,on='feedid',how='left')
#         batch = batch.merge(self.feed_embeddings,on='feedid',how='left')
        import time
        t = time.time()
        batch = batch.merge(self.graph_emb8, how='left',
              on='userid')
        batch = batch.merge(self.feed_emb_16, how='left',
                      on='feedid')
        batch = batch.merge(self.weight_emb8, how='left',
                      on='userid')
        batch = batch.merge(self.keyword_w2v_8, how='left',
                      on='feedid')
        batch = batch.merge(self.userid_feedid_d2v_all_16, how='left',
                      on='userid')
        batch = batch.merge(self.all_text_data_v8, how='left',
                      on='feedid')
        batch = batch.merge(self.userid_authorid_d2v_all_16, how='left',
                      on='userid')
        batch = batch.merge(self.feedinfo[[ x for x in self.feedinfo.columns if x in self.var_len_features + self.sparse_features + self.dense_features]],
                            how='left',
                            on='feedid')             
#         print('get batch cost time: %s' % (time.time() - t))
        x = {name: batch[name].values for name in self.feature_names}
        for col in ['manual_tag_list','manual_keyword_list','machine_keyword_list']:
            x[col] = np.array(batch[col].tolist())
        y = [batch[y].values for y in ACTION_LIST]
#         print('get batch cost time: %s' % (time.time() - t))
        return x,y
        
    def getFeatureColumns(self,):
        embedding_dim = 16
        sparse_features = [ x for x in self.sparse_features if '_list' not in x] #排除变长特征的单独列
        dense_features = self.dense_features 
         
        
        ###dense
        for df in [
                self.graph_emb8, 
                self.feed_emb_16, 
                self.weight_emb8,
                self.keyword_w2v_8, 
                self.userid_feedid_d2v_all_16,
                self.all_text_data_v8, 
                self.userid_authorid_d2v_all_16
        ]:
            dense_features += [
                x for x in df.columns if x not in ['userid', 'feedid']
            ]
            
        ### user id  and varlen
        userid_columns = [
            SparseFeat('userid',
                       vocabulary_size=USERID_MAX,
                       embedding_dim=embedding_dim)
        ]
        
        tag_columns = [
            VarLenSparseFeat(SparseFeat('manual_tag_list',
                                        vocabulary_size=TAG_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4)
        ]
        
        key_words_columns = [
            VarLenSparseFeat(SparseFeat('manual_keyword_list',
                                        vocabulary_size=KEY_WORDS_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4),
            VarLenSparseFeat(SparseFeat('machine_keyword_list',
                                        vocabulary_size=KEY_WORDS_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4),
        ]
        
        # sparse
        fixlen_feature_columns = [
            SparseFeat(feat,
                       vocabulary_size=self.feedinfo[feat].max() + 1,
                       embedding_dim=embedding_dim) for feat in sparse_features
            if feat !='userid'
        ] + [SparseFeat('manual_tag_list' + str(x),
                       vocabulary_size=TAG_MAX ,
                       embedding_dim=embedding_dim) for x in range(4)  # 
        ] + [SparseFeat('manual_keyword_list' + str(x),
                       vocabulary_size=KEY_WORDS_MAX,
                       embedding_dim=embedding_dim) for x in range(4)
        ] + [SparseFeat('machine_keyword_list' + str(x),
                       vocabulary_size=KEY_WORDS_MAX,
                       embedding_dim=embedding_dim) for x in range(4)
        ]
        
        
        ### dense feature
        dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

        dnn_feature_columns = fixlen_feature_columns + tag_columns + key_words_columns + dense_feature_columns + userid_columns
        return dnn_feature_columns
class myDataGenerator_v2(tf.keras.utils.Sequence):
    def __init__(self, data: pd.DataFrame,batch_size=2048, shuffle=True,mode = 'train'):
        
        
        assert mode == 'train' or mode == 'test'
        if mode == 'test' and shuffle == True :
            raise ValueError('测试数据打乱了！')
            
        self.data = data.copy()
        self.data = self.data.reset_index(drop = True)
        self.target = ACTION_LIST
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(self.data.shape[0])
        self.feedinfo = loadFeedinfo()
        self.sparse_features = list(set(['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']))
        
        self.var_len_features = ['manual_tag_list', 'manual_keyword_list', 'machine_keyword_list'] 
        self.dense_features = ['videoplayseconds',]
        

        self.graph_emb8 = loadPkl(os.path.join(MODEL_PATH,'emb/graph_walk_emb_8.pkl'))
        self.feed_emb_16 = loadPkl(os.path.join(MODEL_PATH,'emb/feed_embeddings_16.pkl'))
        self.weight_emb8 = loadPkl(os.path.join(MODEL_PATH,'emb/user_weight_emd_8.pkl'))
        self.weight_emb8 = self.weight_emb8.drop('user_date_weight_emd',axis = 1)
        self.keyword_w2v_8 = loadPkl(os.path.join(MODEL_PATH,'emb/keyword_w2v_8.pkl'))
        self.userid_feedid_d2v_all_16 = loadPkl(os.path.join(MODEL_PATH,'emb/userid_feedid_d2v_all_16.pkl'))##加了初赛数据
        self.all_text_data_v8 = loadPkl(os.path.join(MODEL_PATH,'emb/all_text_data_v8.pkl'))
        self.userid_authorid_d2v_all_16 = loadPkl(os.path.join(MODEL_PATH,'emb/userid_authorid_d2v_all_16.pkl'))
        
        if mode == 'train':
            self.dnn_feature_columns = self.getFeatureColumns()
            self.feature_names = get_feature_names(self.dnn_feature_columns)
            self.feature_index = build_input_features(self.dnn_feature_columns)
            savePkl(self.dnn_feature_columns,os.path.join(MODEL_PATH,'feature_columns_all.pkl'))
            print('feature columns have saved')
        else :
            self.dnn_feature_columns = loadPkl(os.path.join(MODEL_PATH,'feature_columns_all.pkl'))
            self.feature_names = get_feature_names(self.dnn_feature_columns)
            self.feature_index = build_input_features(self.dnn_feature_columns)
            print('load feature columns' ,os.path.join(MODEL_PATH,'feature_columns_all.pkl'))
        
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)

    def __len__(self):

        return (self.data.shape[0] // self.batch_size) + 1

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) *
                                    self.batch_size]
        batch_data = self.data.iloc[batch_indexs, :]
        
        return self.get_feature_on_batch(batch_data)

    def on_epoch_end(self):
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)
    def on_epoch_begain(self):
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)

    def get_feature_on_batch(self, batch):
        
        batch = batch.merge(self.graph_emb8, how='left',
              on='userid')
        batch = batch.merge(self.feed_emb_16, how='left',
                      on='feedid')
        batch = batch.merge(self.weight_emb8, how='left',
                      on='userid')
        batch = batch.merge(self.keyword_w2v_8, how='left',
                      on='feedid')
        batch = batch.merge(self.userid_feedid_d2v_all_16, how='left',
                      on='userid')
        batch = batch.merge(self.all_text_data_v8, how='left',
                      on='feedid')
        batch = batch.merge(self.userid_authorid_d2v_all_16, how='left',
                      on='userid')
        batch = batch.merge(self.feedinfo[[ x for x in self.feedinfo.columns if x in self.var_len_features + self.sparse_features + self.dense_features]],
                            how='left',
                            on='feedid')             

        x = {name: batch[name].values for name in self.feature_names}
        for col in ['manual_tag_list','manual_keyword_list','machine_keyword_list']:
            x[col] = np.array(batch[col].tolist())
        y = [batch[y].values for y in ACTION_LIST]
        return x,y
        
    def getFeatureColumns(self,):
        embedding_dim = 16
        sparse_features = [ x for x in self.sparse_features if '_list' not in x] #排除变长特征的单独列
        dense_features = self.dense_features 
         
        
        ###dense
        for df in [
                self.graph_emb8, 
                self.feed_emb_16, 
                self.weight_emb8,
                self.keyword_w2v_8, 
                self.userid_feedid_d2v_all_16,
                self.all_text_data_v8, 
                self.userid_authorid_d2v_all_16
        ]:
            dense_features += [
                x for x in df.columns if x not in ['userid', 'feedid']
            ]
            
        ### user id  and varlen
        userid_columns = [
            SparseFeat('userid',
                       vocabulary_size=USERID_MAX,
                       embedding_dim=embedding_dim)
        ]
        
        tag_columns = [
            VarLenSparseFeat(SparseFeat('manual_tag_list',
                                        vocabulary_size=TAG_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4)
        ]
        
        key_words_columns = [
            VarLenSparseFeat(SparseFeat('manual_keyword_list',
                                        vocabulary_size=KEY_WORDS_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4),
            VarLenSparseFeat(SparseFeat('machine_keyword_list',
                                        vocabulary_size=KEY_WORDS_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4),
        ]
        
        # sparse
        fixlen_feature_columns = [
            SparseFeat(feat,
                       vocabulary_size=self.feedinfo[feat].max() + 1,
                       embedding_dim=embedding_dim) for feat in sparse_features
            if feat !='userid'
        ] 
        
        
        ### dense feature
        dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

        dnn_feature_columns = fixlen_feature_columns + tag_columns + key_words_columns + dense_feature_columns + userid_columns
        return dnn_feature_columns
class myDataGenerator_base(tf.keras.utils.Sequence):
    def __init__(self, data: pd.DataFrame,batch_size=2048, shuffle=True,mode = 'train'):
        
        
        assert mode == 'train' or mode == 'test'
        if mode == 'test' and shuffle == True :
            raise ValueError('测试数据打乱了！')
            
        self.data = data.copy()
        self.data = self.data.reset_index(drop = True)
        self.target = ACTION_LIST
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(self.data.shape[0])
        self.feedinfo = loadFeedinfo()
        self.sparse_features = list(set(['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id' 
                  ] +  [x for x in self.feedinfo.columns if 'manual_tag_list' in x 
                  ] + [x for x in self.feedinfo.columns if 'manual_keyword_list' in x 
                  ] + [x for x in self.feedinfo.columns if 'machine_keyword_list' in x]))
        
        self.var_len_features = ['manual_tag_list', 'manual_keyword_list', 'machine_keyword_list'] 
        self.dense_features = ['videoplayseconds',]
        
        
        
        if mode == 'train':
            self.dnn_feature_columns = self.getFeatureColumns()
            self.feature_names = get_feature_names(self.dnn_feature_columns)
            self.feature_index = build_input_features(self.dnn_feature_columns)
            savePkl(self.dnn_feature_columns,os.path.join(MODEL_PATH,'feature_columns_base.pkl'))
            print('feature columns have saved')
        else :
            self.dnn_feature_columns = loadPkl(os.path.join(MODEL_PATH,'feature_columns_base.pkl'))
            self.feature_names = get_feature_names(self.dnn_feature_columns)
            self.feature_index = build_input_features(self.dnn_feature_columns)
            print('load feature columns' ,os.path.join(MODEL_PATH,'feature_columns_base.pkl'))
        
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)

    def __len__(self):

        return (self.data.shape[0] // self.batch_size) + 1

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) *
                                    self.batch_size]
        batch_data = self.data.iloc[batch_indexs, :]
        
        return self.get_feature_on_batch(batch_data)

    def on_epoch_end(self):
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)
    def on_epoch_begain(self):
        if self.shuffle:
            print('shuffle data index ing...')
            np.random.shuffle(self.indexes)

    def get_feature_on_batch(self, batch):
        
        batch = batch.merge(self.feedinfo[[ x for x in self.feedinfo.columns if x in self.var_len_features + self.sparse_features + self.dense_features]],
                            how='left',
                            on='feedid')             
        x = {name: batch[name].values for name in self.feature_names}
        for col in ['manual_tag_list','manual_keyword_list','machine_keyword_list']:
            x[col] = np.array(batch[col].tolist())
        y = [batch[y].values for y in ACTION_LIST]
        return x,y
        
    def getFeatureColumns(self,):
        embedding_dim = 16
        sparse_features = [ x for x in self.sparse_features if '_list' not in x] #排除变长特征的单独列
        dense_features = self.dense_features 
         
        ### user id  and varlen
        userid_columns = [
            SparseFeat('userid',
                       vocabulary_size=USERID_MAX,
                       embedding_dim=embedding_dim)
        ]
        
        tag_columns = [
            VarLenSparseFeat(SparseFeat('manual_tag_list',
                                        vocabulary_size=TAG_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4)
        ]
        
        key_words_columns = [
            VarLenSparseFeat(SparseFeat('manual_keyword_list',
                                        vocabulary_size=KEY_WORDS_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4),
            VarLenSparseFeat(SparseFeat('machine_keyword_list',
                                        vocabulary_size=KEY_WORDS_MAX,
                                        embedding_dim=embedding_dim),
                             maxlen=4),
        ]
        
        # sparse
        fixlen_feature_columns = [
            SparseFeat(feat,
                       vocabulary_size=self.feedinfo[feat].max() + 1,
                       embedding_dim=embedding_dim) for feat in sparse_features
            if feat !='userid'
        ] + [SparseFeat('manual_tag_list' + str(x),
                       vocabulary_size=TAG_MAX ,
                       embedding_dim=embedding_dim) for x in range(4)  # 
        ] + [SparseFeat('manual_keyword_list' + str(x),
                       vocabulary_size=KEY_WORDS_MAX,
                       embedding_dim=embedding_dim) for x in range(4)
        ] + [SparseFeat('machine_keyword_list' + str(x),
                       vocabulary_size=KEY_WORDS_MAX,
                       embedding_dim=embedding_dim) for x in range(4)
        ]
        
        
        ### dense feature
        dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]

        dnn_feature_columns = fixlen_feature_columns + tag_columns + key_words_columns + dense_feature_columns + userid_columns
        return dnn_feature_columns

    
def get_MMOE_MutiHead_v2(dnn_feature_columns):
    num_tasks = len(ACTION_LIST)
    train_model = MMOE_mutihead(dnn_feature_columns, 
                   num_tasks=num_tasks,
                   task_types = ['binary' for i in range(num_tasks)],
                   task_names = ACTION_LIST,
                   num_experts=5,
                   tower_dnn_units_lists = [[64,32] for i in range(num_tasks) ],
                   dnn_hidden_units=(512, 512),
                   expert_dim=32,
                   multi_head_num = 3,
                  )
    train_model.compile('adagrad', loss='binary_crossentropy')
    return train_model    
    
def get_Shared_Bottom(dnn_feature_columns):
    num_tasks = len(ACTION_LIST)
    train_model = Shared_Bottom(
                       dnn_feature_columns=dnn_feature_columns,
                       num_tasks=num_tasks,
                       bottom_dnn_units=[512,512],
                       task_types = ['binary' for i in range(num_tasks)],
                       task_names = ACTION_LIST,
                       tower_dnn_units_lists = [[64,32] for i in range(num_tasks) ],
    )
#     train_model.summary()
#     len(train_loader)
#     train_model = multi_gpu_model(train_model, gpus=2)
#     optimizer = tf.keras.optimizers.Adagrad(
#         lr=0.05, epsilon=1e-07,
#     )
    train_model.compile('adagrad', loss='binary_crossentropy')
    return train_model

def get_MMOE_FEFM(dnn_feature_columns):
    num_tasks = len(ACTION_LIST)
    train_model = MMOE_FefM(
                   dnn_feature_columns=dnn_feature_columns,
                   num_tasks=num_tasks,
                   task_types = ['binary' for i in range(num_tasks)],
                   task_names = ACTION_LIST,
                   num_experts=7,
                   tower_dnn_units_lists = [[64,32] for i in range(num_tasks) ],
                   dnn_hidden_units=(512, 512),
                   expert_dim=32,)
    train_model.compile('adagrad', loss='binary_crossentropy')
    return train_model

def get_MMOE_MutiHead(dnn_feature_columns):
    num_tasks = len(ACTION_LIST)
    train_model = MMOE_mutihead(dnn_feature_columns, 
                   num_tasks=num_tasks,
                   task_types = ['binary' for i in range(num_tasks)],
                   task_names = ACTION_LIST,
                   num_experts=7,
                   tower_dnn_units_lists = [[64,32] for i in range(num_tasks) ],
                   dnn_hidden_units=(512, 512),
                   expert_dim=32,
                   multi_head_num = 3,
                  )
    train_model.compile('adagrad', loss='binary_crossentropy')
    return train_model

def get_MMOE_FEFM_mutihead_base(dnn_feature_columns):
    num_tasks = len(ACTION_LIST)
    train_model = MMOE_FefM_multihead(
                   dnn_feature_columns=dnn_feature_columns,
                   num_tasks=num_tasks,
                   task_types = ['binary' for i in range(num_tasks)],
                   task_names = ACTION_LIST,
                   num_experts=7,
                   tower_dnn_units_lists = [[64,32] for i in range(num_tasks) ],
                   dnn_hidden_units=(128, 128),
                    multi_head_num = 5,
                   expert_dim=32,)
    train_model.compile('adagrad', loss='binary_crossentropy')
    return train_model

def infer(test_loader,model,model_weights_path,):
    t1 = time.time()
    sub = test_loader.data.copy()
    model.load_weights(model_weights_path)
    print('model weights load from %s' % (model_weights_path))
    pred_ans = model.predict(test_loader,workers = 4,use_multiprocessing=True,max_queue_size=200)
    for i, action in enumerate(ACTION_LIST):
        sub[action] = pred_ans[i]
    t2 = time.time()
    print('7个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    print('7个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)
    return sub[['userid', 'feedid'] + ACTION_LIST]

SEED = 100
import time
if __name__ == "__main__":
    argv = sys.argv
#     argv = ['python','../../data/wedata/wechat_algo_data2/test_a.csv']
#     params = xdeepfm_params
    t = time.time() 
    test_path = ''
    if len(argv)==2:
        test_path = argv[1]
        print(test_path)
        t1 = time.time()
        test = pd.read_csv(test_path)
        test[ACTION_LIST] = 0
        test_loader = myDataGenerator(test,shuffle=False,batch_size=4096*40,mode ='test')
        dnn_feature_columns = test_loader.dnn_feature_columns
        print('Get test input cost: %.4f s'%(time.time()-t1))
    
    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None
    
    print('开始预测share bottom...')
    share_bottom_model = get_Shared_Bottom(dnn_feature_columns)
    submission1 = infer(test_loader,share_bottom_model,os.path.join(MODEL_PATH,'tf_models/share_bottom/model_seed%s' % (SEED)))
    
#     print('开始预测MMOE FEFM...')
#     mmoe_fefm_model = get_MMOE_FEFM(dnn_feature_columns)
#     submission2 = infer(test_loader,mmoe_fefm_model,os.path.join(MODEL_PATH,'tf_models/MMOE_FEFM/model_seed%s' % (SEED)))
    
#     print('开始预测MMOE MUTI_HEAD...')
#     mmoe_multihead_model = get_MMOE_MutiHead(dnn_feature_columns)
#     submission3 = infer(test_loader,mmoe_multihead_model,os.path.join(MODEL_PATH,'tf_models/MMOE_MutiHead/model_seed%s' % (SEED)))
    
#     print('开始预测 MMOE base ...')
#     test_loader2 = myDataGenerator_base(test,shuffle=False,batch_size=4096*40, mode = 'test')
#     dnn_feature_columns = test_loader2.dnn_feature_columns
#     mmoe_base_model = get_MMOE_FEFM_mutihead_base(dnn_feature_columns)
#     submission4 = infer(test_loader2,mmoe_base_model,os.path.join(MODEL_PATH, 'tf_models/MMOE_FEFM_base/model_seed%s' % (200)))
    
#     print('开始预测 MMOE v2 ...')
#     test_loader2 = myDataGenerator(test,shuffle=False,batch_size=4096*40, mode = 'test')
#     dnn_feature_columns_v2 = test_loader2.dnn_feature_columns
#     mmoe_base_model = get_MMOE_MutiHead_v2(dnn_feature_columns_v2)
#     submission5 = infer(test_loader2,mmoe_base_model,os.path.join(MODEL_PATH, 'tf_models/MMOE_MutiHead_v2/model_seed%s' % (100)))
    
    
    submission1.to_csv(os.path.join(SUMIT_DIR,'share_bottom.csv'),index=None)
#     submission2.to_csv(os.path.join(SUMIT_DIR,'MMOE_FEFM.csv'),index=None)
#     submission3.to_csv(os.path.join(SUMIT_DIR,'MMOE_MutiHead.csv'),index=None)
#     submission4.to_csv(os.path.join(SUMIT_DIR,'mmoe_base.csv'),index=None)
#     submission5.to_csv(os.path.join(SUMIT_DIR,'mmoe_v2.csv'),index=None)
    
    print('Time cost: %.2f s'%(time.time()-t))