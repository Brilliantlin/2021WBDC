import sys
import os
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from config_prosper import *
from mytools.utils.seed import seed_everything
from mytools.utils.myfile import savePkl,loadPkl
from pandarallel import pandarallel
seed_everything(SEED)
pandarallel.initialize(nb_workers = 8)
def getFeedembeddings(df):
    #feedembeddings 降维

    feed_embedding_path = os.path.join(FEATURE_PATH,'feedembedings.pkl')
    feed_embeddings = loadPkl(feed_embedding_path)
    df = df.merge(feed_embeddings,on='feedid',how='left')
    dense = [x for x in list(feed_embeddings.columns) if x != 'feedid' ]
    
    return df,dense

def getSvdembeddings(df):
    dense = []
    #userid-feedid svd
    svd_embedding = loadPkl(os.path.join(FEATURE_PATH,'svd_userid_feedid_embedding.pkl'))
    df = df.merge(svd_embedding,on = ['userid'],how='left')
    dense += [x for x in list(svd_embedding.columns) if x not in ['userid']]
                            
    #userid_authorid svd
    svd_embedding = loadPkl(os.path.join(FEATURE_PATH,'svd_userid_authorid_embedding.pkl'))
    df  = df.merge(svd_embedding,on = ['userid'],how='left')
    dense += [x for x in list(svd_embedding.columns) if x not in ['userid']]
    
    #text svd
    svd_embedding = loadPkl(os.path.join(FEATURE_PATH,'texts_svd_embedding.pkl'))
    svd_embedding['feedid']  = svd_embedding['feedid'].astype(np.int32) 
    df  = df.merge(svd_embedding,on = ['feedid'],how='left')
    dense += [x for x in list(svd_embedding.columns) if x not in ['feedid']]
    
    return df, dense
def myLeftjoin(left,right,on):
    return left.merge(right[right[on].isin(left[on])].set_index(on),how='left',left_on=on,right_index=True)
def getHistFeatures(df,hist_features):
    dense = [x for x in hist_features.columns if x not in df.columns and  'hist_seq' not in x ]
    varlen = [x for x in hist_features.columns if 'hist_seq' in x]
    df = df.merge(hist_features[hist_features.userid.isin(df.userid.unique())][['userid','feedid','date_','device'] + dense],how = 'left',on = ['userid','feedid','date_','device'])
    return (df,dense)
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

def getAllFeat(data_path,mode):
    print('Mode : %s dealing feature...' % (mode))
    assert mode == 'train' or mode == 'test'
    feed_info = reduce_mem_usage(loadPkl(FEED_INFO)) 
    feed_embeddings = reduce_mem_usage(pd.read_csv(FEED_EMBEDDINGS))
    if mode =='train':
        # 存储各种ID信息全部
        user_action_all = loadPkl(USER_ACTION) if not DEBUG else loadPkl(USER_ACTION).head(20000000)
        IDS = {
            'userid':user_action_all.userid.unique().tolist() + [UNK],
            'feedid':feed_info.feedid.unique().tolist() + [UNK],
            'authorid':feed_info.authorid.unique().tolist() + [UNK],
            'bgm_song_id':feed_info.bgm_song_id.unique().tolist() + [UNK],
            'bgm_singer_id':feed_info.bgm_singer_id.unique().tolist() + [UNK],
        }
        
        VAR_IDS = {}
        DENSE = ['videoplayseconds']
        all_dfs = []
        
        #encoder
        for col in IDS:
            encoder = LabelEncoder()
            encoder.fit(IDS[col])
            savePkl(encoder,os.path.join(MODEL_PATH,'ID_col_%s_encoder.pkl' % (col)))

        begain_index = 0
        window_size = 15000000
        user_action_all = user_action_all.sample(frac=1)
        user_action_all = user_action_all[user_action_all.date_ >=5]
        hist_features = loadPkl(os.path.join(SRC_DIR, 'src/prepare/5days_feature.pkl' ))
        hist_features = hist_features.fillna(0)
        print(user_action_all.shape[0])
        valid_data = []
        while begain_index  < user_action_all.shape[0]:
            print('处理数据%s - %s ......' % (begain_index,begain_index + window_size))
            user_action = user_action_all.iloc[begain_index:begain_index+window_size]
            df = user_action.merge(feed_info,on=['feedid'],how='left')
            del user_action
            df = df[['userid','feedid','date_','device'] + ACTION_LIST + ['authorid','bgm_song_id','bgm_singer_id','videoplayseconds']]
            gc.collect()
            #### merge feat #####
            df,dense_tmp = getFeedembeddings(df)
            DENSE += dense_tmp

            df,dense_tmp = getSvdembeddings(df)
            DENSE += dense_tmp

            print('hist feature ...')
            df,dense_tmp = getHistFeatures(df,hist_features)
            DENSE += dense_tmp
            
            print('encoding ID 特征......')
            df[list(IDS.keys())] = df[list(IDS.keys())].fillna(-1)
            for col in IDS:
                encoder = loadPkl(os.path.join(MODEL_PATH,'ID_col_%s_encoder.pkl' % (col)))
                df[col] = encoder.transform(df[col])
            
            # 特征名定义
            feature_columns = []
            feature_columns += [SparseFeat(k,len(IDS[k]) + 1,embedding_dim=8) for k,v in IDS.items()]
            feature_columns += [DenseFeat(k, 1) for k in DENSE]
            feature_columns += [VarLenSparseFeat(SparseFeat(k,len(VAR_IDS[k]) + 1,embedding_dim = 8,),maxlen=MAX_LNE[k],combiner = 'mean')  for k,v in VAR_IDS.items()]
            
            df = reduce_mem_usage(df)
            valid_data.append(df[df.date_==14])
            df = df[df.date_!=14]
#             dataloader = getDataloader(df,feature_columns,BATCH_SIZE)
            #end
            print('saving...')
            savePkl(df,"train_data_part%s.pkl" % (begain_index//window_size))
            begain_index += window_size
#             break

        del user_action_all
        gc.collect()
        valid_data = pd.concat(valid_data)
        savePkl(valid_data,"valid_data.pkl")
        savePkl((IDS,VAR_IDS,DENSE),os.path.join(MODEL_PATH,'feature_names.pkl'))
        savePkl(feature_columns,os.path.join(MODEL_PATH,'feature_columns.pkl'))
        return 

    else :
        hist_features = loadPkl(os.path.join(SRC_DIR, 'src/prepare/5days_feature_test.pkl' ))
        hist_features = hist_features.fillna(0)
        df = reduce_mem_usage(pd.read_csv(data_path))
        df = df.merge(feed_info,on=['feedid'],how='left')
        df[ACTION_LIST] = 0
        df = df[['userid','feedid','device'] + ACTION_LIST + ['authorid','bgm_song_id','bgm_singer_id','videoplayseconds']]
        df['date_'] = END_DAY

        IDS,VAR_IDS,DENSE = loadPkl(os.path.join(MODEL_PATH,'feature_names.pkl'))

        #### merge feat #####
        df,dense_tmp = getFeedembeddings(df)
        DENSE += dense_tmp

        df,dense_tmp = getSvdembeddings(df)
        DENSE += dense_tmp

        print('hist feature ...')
        df,dense_tmp = getHistFeatures(df,hist_features)
        DENSE += dense_tmp


        df[list(IDS.keys())] = df[list(IDS.keys())].fillna(-1)
        print('测试集encoding ID 特征......')
        for col in IDS:
            encoder = loadPkl(os.path.join(MODEL_PATH,'ID_col_%s_encoder.pkl' % (col)))
            df[col].loc[~df[col].isin(encoder.classes_)] = UNK
            df[col] = encoder.transform(df[col])
        feature_columns = loadPkl(os.path.join(MODEL_PATH,'feature_columns.pkl'))
        return feature_columns,df

    
def getBaseFeat(data_path,mode):
    print('Mode : %s dealing feature...' % (mode))
    assert mode == 'train' or mode == 'test'
    feed_info = reduce_mem_usage(loadPkl(FEED_INFO)) 
    feed_embeddings = reduce_mem_usage(pd.read_csv(FEED_EMBEDDINGS))
    if mode =='train':
        # 存储各种ID信息全部
        user_action_all = loadPkl(USER_ACTION) if not DEBUG else loadPkl(USER_ACTION).head(20000000)
        IDS = {
            'userid':user_action_all.userid.unique().tolist() + [UNK],
            'feedid':feed_info.feedid.unique().tolist() + [UNK],
            'authorid':feed_info.authorid.unique().tolist() + [UNK],
            'bgm_song_id':feed_info.bgm_song_id.unique().tolist() + [UNK],
            'bgm_singer_id':feed_info.bgm_singer_id.unique().tolist() + [UNK],
        }

        VAR_IDS = {}
        DENSE = ['videoplayseconds']
        all_dfs = []
        
        #encoder
        for col in IDS:
            encoder = LabelEncoder()
            encoder.fit(IDS[col])
            savePkl(encoder,os.path.join(MODEL_PATH,'ID_col_%s_encoder.pkl' % (col)))

        begain_index = 0
        window_size = 45000000
        user_action_all = user_action_all.sample(frac=1)
#         hist_features = loadPkl('5days_feature.pkl')
#         hist_features = hist_features.fillna(0)
        print(user_action_all.shape[0])
        valid_data = []
        while begain_index  < user_action_all.shape[0]:
            print('处理数据%s - %s ......' % (begain_index,begain_index + window_size))
            user_action = user_action_all.iloc[begain_index:begain_index+window_size]
            df = user_action.merge(feed_info,on=['feedid'],how='left')
            del user_action
            df = df[['userid','feedid','date_','device'] + ACTION_LIST + ['authorid','bgm_song_id','bgm_singer_id','videoplayseconds']]
            gc.collect()
            #### merge feat #####

            
            print('encoding ID 特征......')
            df[list(IDS.keys())] = df[list(IDS.keys())].fillna(-1)
            for col in IDS:
                encoder = loadPkl(os.path.join(MODEL_PATH,'ID_col_%s_encoder.pkl' % (col)))
                df[col] = encoder.transform(df[col])
            
            # 特征名定义
            feature_columns = []
            feature_columns += [SparseFeat(k,len(IDS[k]) + 1,embedding_dim=8) for k,v in IDS.items()]
            feature_columns += [DenseFeat(k, 1) for k in DENSE]
            feature_columns += [VarLenSparseFeat(SparseFeat(k,len(VAR_IDS[k]) + 1,embedding_dim = 8,),maxlen=MAX_LNE[k],combiner = 'mean')  for k,v in VAR_IDS.items()]
            
            df = reduce_mem_usage(df)
            valid_data.append(df[df.date_==14])
            df = df[df.date_!=14]
#             dataloader = getDataloader(df,feature_columns,BATCH_SIZE)
            #end
            print('saving...')
            savePkl(df,"base_train_data_part%s.pkl" % (begain_index//window_size))
            begain_index += window_size
#             break

        del user_action_all
        gc.collect()
        valid_data = pd.concat(valid_data)
        savePkl(valid_data,"valid_data.pkl")
        savePkl((IDS,VAR_IDS,DENSE),os.path.join(MODEL_PATH,'feature_names.pkl'))
        savePkl(feature_columns,os.path.join(MODEL_PATH,'feature_columns.pkl'))
        return 

    else :
        df = reduce_mem_usage(pd.read_csv(data_path))
        df = df.merge(feed_info,on=['feedid'],how='left')
        df[ACTION_LIST] = 0
        df = df[['userid','feedid','device'] + ACTION_LIST + ['authorid','bgm_song_id','bgm_singer_id','videoplayseconds']]
        df['date_'] = END_DAY

        IDS,VAR_IDS,DENSE = loadPkl(os.path.join(MODEL_PATH,'feature_names.pkl'))

        #### merge feat #####
#         df,dense_tmp = getFeedembeddings(df)
#         DENSE += dense_tmp

#         df,dense_tmp = getSvdembeddings(df)
#         DENSE += dense_tmp

#         print('hist feature ...')
#         df,dense_tmp = getHistFeatures(df,hist_features)
#         DENSE += dense_tmp


        df[list(IDS.keys())] = df[list(IDS.keys())].fillna(-1)
        print('测试集encoding ID 特征......')
        for col in IDS:
            encoder = loadPkl(os.path.join(MODEL_PATH,'ID_col_%s_encoder.pkl' % (col)))
            df[col] = df[col].map(lambda x: x if x in encoder.classes_ else UNK )
            df[col] = encoder.transform(df[col])
        feature_columns = loadPkl(os.path.join(MODEL_PATH,'feature_columns.pkl'))
        return feature_columns,df 

if __name__ == "__main__":
    t1 = time.time()
    feature_columns,train_data = getFeat('None','train')
    savePkl(feature_columns,os.path.join(MODEL_PATH,'feature_columns.pkl'))
    train_data = reduce_mem_usage(train_data)
    savePkl(train_data,'train_data.pkl')
    print(train_data.info())
    print('Time cost: %.2f s'%(time.time()-t1))