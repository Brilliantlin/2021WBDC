

# load deepfm ------------------
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from scipy import sparse
import os
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
from evaluation import uAUC

os.environ['NUMEXPR_MAX_THREADS'] = '16'
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)



# 存储数据的根目录
ROOT_PATH = "./data/wedata"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_b.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']



def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
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
    print('start memory {:.2f} Mb, end memory {:.2f} Mb reduced ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


feed_info = reduce_mem(pd.read_csv(FEED_INFO))
user_action_df = reduce_mem(pd.read_csv(USER_ACTION))
# feed_embed = pd.read_csv(FEED_EMBEDDINGS)
test = reduce_mem(pd.read_csv(TEST_FILE))
test['date_'] = 15


feed_use_feature = ['feedid','authorid','videoplayseconds','bgm_song_id', 'bgm_singer_id']
feed_info_use = feed_info[feed_use_feature] 

data_df = pd.concat([user_action_df, test], axis=0, ignore_index=True)
data_df = pd.merge(data_df, feed_info_use, on=['feedid'],how='left',copy=False)


## user feed embed -----------------------------------------------------------------

userid_feedid_d2v_max_wd = pd.read_pickle('./data/wedata/userid_feedid_d2v_b.pkl')

data_df = pd.merge(data_df, userid_feedid_d2v_max_wd, on=['userid'],how='left',copy=False)


## user author embed -----------------------------------------------------------------

userid_authorid_d2v_max_wd = pd.read_pickle('./data/wedata/userid_authorid_d2v_b.pkl')

data_df = pd.merge(data_df, userid_authorid_d2v_max_wd, on=['userid'],how='left',copy=False)


del userid_feedid_d2v_max_wd, userid_authorid_d2v_max_wd






# ## count feature --------------------------------------------------------

data_df[['bgm_song_id','bgm_singer_id']]=data_df[['bgm_song_id','bgm_singer_id']].fillna(0)
data_df[['bgm_song_id','bgm_singer_id']]=data_df[['bgm_song_id','bgm_singer_id']].fillna(0)

## 视频时长是秒，转换成毫秒，才能与play、stay做运算 -----------------------------------------------------

data_df['videoplayseconds'] *= 1000
## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
data_df['is_finish'] = (data_df['play'] >= 0.9*data_df['videoplayseconds']).astype('int8')
data_df['play_times'] = data_df['play'] / data_df['videoplayseconds']
play_cols = ['is_finish', 'play_times', 'play', 'stay']


## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
FEA_COLUMN_LIST = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15
n_day = 5

for stat_cols in tqdm([
    ['userid'],
    ['feedid'],
    ['authorid'],
    # ['bgm_song_id'],
    # ['bgm_singer_id'],
    ['userid', 'authorid']
    # ['userid', 'bgm_song_id'],
    # ['userid', 'bgm_singer_id']
]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1

        tmp = data_df[((data_df['date_'] >= left) & (data_df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

        g = tmp.groupby(stat_cols)
        
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean').astype(np.float32)
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

        for x in play_cols[1:]:
            for stat in ['max', 'mean']:
                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

        for y in FEA_COLUMN_LIST[:4]:
            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum').astype(np.float32)
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean').astype(np.float32)
            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])

        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)

        del g, tmp

    data_df = data_df.merge(stat_df, on=stat_cols + ['date_'], how='left')

    del stat_df
    gc.collect()

## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

# 曝光 -----------------------------------------------------------
for f in tqdm(['userid', 'feedid', 'authorid']):

    data_df[f + '_count'] = data_df[f].map(data_df[f].value_counts())

# number of unique ------------------------------------------------------
for f1, f2 in tqdm([
    ['userid', 'feedid'],
    ['userid', 'authorid']
]):
    data_df['{}_in_{}_nunique'.format(f1, f2)] = data_df.groupby(f2)[f1].transform('nunique')
    data_df['{}_in_{}_nunique'.format(f2, f1)] = data_df.groupby(f1)[f2].transform('nunique')

# 偏好 ----------------------------------------------------------------
for f1, f2 in tqdm([
    ['userid', 'authorid']
]):
    data_df['{}_{}_count'.format(f1, f2)] = data_df.groupby([f1, f2])['date_'].transform('count')
    data_df['{}_in_{}_count_prop'.format(f1, f2)] = data_df['{}_{}_count'.format(f1, f2)] / (data_df[f2 + '_count'] + 1)
    data_df['{}_in_{}_count_prop'.format(f2, f1)] = data_df['{}_{}_count'.format(f1, f2)] / (data_df[f1 + '_count'] + 1)

# ----------------------------------------------------------------------------------
data_df['videoplayseconds_in_userid_mean'] = data_df.groupby('userid')['videoplayseconds'].transform('mean')
data_df['videoplayseconds_in_authorid_mean'] = data_df.groupby('authorid')['videoplayseconds'].transform('mean')
# 作者有多少视频数
data_df['feedid_in_authorid_nunique'] = data_df.groupby('authorid')['feedid'].transform('nunique')


# keyword_data -------------------------------------------------

keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']

keyword_data = pd.read_csv('./data/wedata/keyword_w2v_16.csv', index_col=0)


# # keyctr -------------------------------------------------------------------

keyctr_data = pd.read_pickle('./data/wedata/keyword_ctr_final.pkl')

keyctr_data = keyctr_data.fillna(0)

data_df = data_df.merge(keyctr_data, how='left', on=['userid','feedid','date_'])

del keyctr_data



#d2v ----------------------------------------------------------------------------------
# text doc
text_feature = ['description','ocr','asr']

text_data = pd.read_csv('./data/wedata/all_text_data_20v.csv', index_col=0)

text_data = text_data.drop('feedid', axis=1)
text_data.columns = [i+'_d2v' for i in text_data.columns]


# merge embedding ---------------------------------------------------------------------------
feed_embed_processed = pd.read_csv('./data/wedata/feed_embeddings_use_rd_32.csv', index_col=0)
data_df = pd.merge(data_df, feed_embed_processed, on=['feedid'],how='left',copy=False)



feed_info_use = pd.concat([feed_info_use, keyword_data, text_data], axis=1)


data_df = pd.merge(data_df, feed_info_use[['feedid']+keyword_data.columns.tolist()+text_data.columns.tolist()], on=['feedid'],how='left',copy=False)




# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']

play_cols = ['is_finish', 'play_times', 'play', 'stay']
drop_columns = ['play', 'stay']
keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']
text_feature = ['description','ocr','asr']

data_df = data_df.drop(play_cols, axis=1)





train_use_all = data_df.iloc[0:user_action_df.shape[0],:]
test_use_all = data_df.iloc[user_action_df.shape[0]:,:]

train_use_all.shape, test_use_all.shape


ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
training_features = [i for i in data_df.columns if i not in FEA_COLUMN_LIST+['date_']]

categorical_feature = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id']




class MyBaseModel(BaseModel):

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
            #ul = val_x[0]  # --------------------------------------------------
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                try:
                                    temp = metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
                                except Exception:
                                    temp = 0
                                finally:
                                    train_result[name].append(temp)
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result, pred_ans = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
                final_uauc =  uAUC(pd.DataFrame(val_y)[0].tolist(), pd.DataFrame(pred_ans)[0].tolist(), pd.DataFrame(val_x[0]).astype(int)[0].tolist())
                print("val_uAUC:{0: .4f}".format(final_uauc))   # ----------------------
            else:
                final_uauc = 0.0
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name +                                 ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name +                                     ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history, final_uauc

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                temp = metric_fun(y, pred_ans)
            except Exception:
                temp = 0
            finally:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result, pred_ans

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")


class MyDeepFM(MyBaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 256, 256),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0.2,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(MyDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred



ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
submit_df = test_use_all[['userid', 'feedid']+ACTION_LIST].reset_index(drop=True)
submit_df[ACTION_LIST] = 0.0
score = []
seed_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 2049, 2050]

k = len(seed_list)

for seed_num in seed_list:
    for action in ACTION_LIST:

        print("posi prop:")
        print(sum((train_use_all[action]==1)*1)/train_use_all.shape[0])
        
        train_target = train_use_all[action]
        train = train_use_all[training_features]
        test = test_use_all[training_features]
        
        

        data = pd.concat((train, test)).reset_index(drop=True)
        
        sparse_features = categorical_feature
        
        dense_features = [i for i in train.columns if i not in sparse_features] 
        

        data[sparse_features] = data[sparse_features].fillna(0)
        data[dense_features] = data[dense_features].fillna(0)

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
            
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                for feat in dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(drop=True)
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = MyDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    task='binary',
                    l2_reg_embedding=1e-1, device=device,
                    seed = seed_num)

        model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

        # history, final_uauc = model.fit(train_model_input, train_target.values, batch_size=1024, epochs=5, verbose=1,
        #                     validation_split=0.0)

        # score.append(final_uauc)

        # torch.save(model.state_dict(), './data/model/deepfm_14_{}_{}.pt'.format(action, seed_num))

        model.load_state_dict(torch.load('./data/model/deepfm_14_{}_{}.pt'.format(action, seed_num)))
        model.eval()

        pred_ans = model.predict(test_model_input, 1024)

        submit_df[action] = (np.array(submit_df[action]) + pred_ans.reshape(1,-1)/k).reshape(-1,1)
        torch.cuda.empty_cache()

deepfm = submit_df
# get submit_df



# load lgb1 -------------------


# -*- coding: utf-8 -*-
from tqdm import tqdm

import gc
import numpy as np
import lightgbm as lgb
from evaluation import uAUC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
# import seaborn as sns
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# 存储数据的根目录
ROOT_PATH = "./data/wedata"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_b.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']



def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
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
    print('start memory {:.2f} Mb, end memory {:.2f} Mb reduced ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


feed_info = reduce_mem(pd.read_csv(FEED_INFO))
user_action_df = reduce_mem(pd.read_csv(USER_ACTION))
feed_embed = pd.read_csv(FEED_EMBEDDINGS)
test = reduce_mem(pd.read_csv(TEST_FILE))
test['date_'] = 15


feed_use_feature = ['feedid','authorid','videoplayseconds','bgm_song_id', 'bgm_singer_id']
feed_info_use = feed_info[feed_use_feature] # feed_info
# feed_info_use.loc[feed_info_use['videoplayseconds']>=300,'videoplayseconds'] = 300



data_df = pd.concat([user_action_df, test], axis=0, ignore_index=True)
data_df = pd.merge(data_df, feed_info_use, on=['feedid'],how='left',copy=False)




## user feed embed -----------------------------------------------------------------


userid_feedid_d2v_max_wd = pd.read_pickle('./data/wedata/userid_feedid_d2v_b.pkl')

data_df = pd.merge(data_df, userid_feedid_d2v_max_wd, on=['userid'],how='left',copy=False)


## user author embed -----------------------------------------------------------------

userid_authorid_d2v_max_wd = pd.read_pickle('./data/wedata/userid_authorid_d2v_b.pkl')

data_df = pd.merge(data_df, userid_authorid_d2v_max_wd, on=['userid'],how='left',copy=False)



del userid_feedid_d2v_max_wd, userid_authorid_d2v_max_wd






data_df[['bgm_song_id','bgm_singer_id']]=data_df[['bgm_song_id','bgm_singer_id']].fillna(0)
data_df[['bgm_song_id','bgm_singer_id']]=data_df[['bgm_song_id','bgm_singer_id']].fillna(0)

## 视频时长是秒，转换成毫秒，才能与play、stay做运算 -----------------------------------------------------

data_df['videoplayseconds'] *= 1000
## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
data_df['is_finish'] = (data_df['play'] >= 0.9*data_df['videoplayseconds']).astype('int8')
data_df['play_times'] = data_df['play'] / data_df['videoplayseconds']
play_cols = ['is_finish', 'play_times', 'play', 'stay']


# ctr ---------------------------------------------------------------------------------------
## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
FEA_COLUMN_LIST = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15
n_day = 5





for stat_cols in tqdm([
    ['userid'],
    ['feedid'],
    ['authorid'],
    # ['bgm_song_id'],
    # ['bgm_singer_id'],
    ['userid', 'authorid']
    # ['userid', 'bgm_song_id'],
    # ['userid', 'bgm_singer_id']
]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1

        tmp = data_df[((data_df['date_'] >= left) & (data_df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

        g = tmp.groupby(stat_cols)
        
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean').astype(np.float32)
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

        for x in play_cols[1:]:
            for stat in ['max', 'mean']:
                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

        for y in FEA_COLUMN_LIST[:4]:
            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum').astype(np.float32)
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean').astype(np.float32)
            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])

        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)

        del g, tmp

    data_df = data_df.merge(stat_df, on=stat_cols + ['date_'], how='left')

    del stat_df
    gc.collect()





## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

# 曝光 -----------------------------------------------------------
for f in tqdm(['userid', 'feedid', 'authorid']):

    data_df[f + '_count'] = data_df[f].map(data_df[f].value_counts())

# number of unique ------------------------------------------------------
for f1, f2 in tqdm([
    ['userid', 'feedid'],
    ['userid', 'authorid']
]):
    data_df['{}_in_{}_nunique'.format(f1, f2)] = data_df.groupby(f2)[f1].transform('nunique')
    data_df['{}_in_{}_nunique'.format(f2, f1)] = data_df.groupby(f1)[f2].transform('nunique')

# 偏好 ----------------------------------------------------------------
for f1, f2 in tqdm([
    ['userid', 'authorid']
]):
    data_df['{}_{}_count'.format(f1, f2)] = data_df.groupby([f1, f2])['date_'].transform('count')
    data_df['{}_in_{}_count_prop'.format(f1, f2)] = data_df['{}_{}_count'.format(f1, f2)] / (data_df[f2 + '_count'] + 1)
    data_df['{}_in_{}_count_prop'.format(f2, f1)] = data_df['{}_{}_count'.format(f1, f2)] / (data_df[f1 + '_count'] + 1)

# ----------------------------------------------------------------------------------
data_df['videoplayseconds_in_userid_mean'] = data_df.groupby('userid')['videoplayseconds'].transform('mean')
data_df['videoplayseconds_in_authorid_mean'] = data_df.groupby('authorid')['videoplayseconds'].transform('mean')
# 作者有多少视频数
data_df['feedid_in_authorid_nunique'] = data_df.groupby('authorid')['feedid'].transform('nunique')



# ## keyword w2v -------------------------------------------

keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']

keyword_data = pd.read_csv('./data/wedata/keyword_w2v_16.csv', index_col=0)
keyword_data.columns = [i+'w2v' for i in keyword_data.columns]


# # keyctr -------------------------------------------------------------------
# import pickle
# def load_pickle(input_file):
#     '''
#     读取pickle文件
#     :param pickle_path:
#     :param file_name:
#     :return:
#     '''
#     with open(str(input_file), 'rb') as f:
#         data = pickle.load(f)
#     return data

keyctr_data = pd.read_pickle('./data/wedata/keyword_ctr_final.pkl')

keyctr_data = keyctr_data.fillna(0)

data_df = data_df.merge(keyctr_data, how='left', on=['userid','feedid','date_'])

del keyctr_data

# keyword cate -----------------------------------------------------------------------
def process_keywords(colname):
    data = feed_info[colname].str.split(';')
    keyword_array = pd.DataFrame(np.zeros((feed_info.shape[0], 1)), columns=[colname]).astype(object) 
    for i in tqdm(range(data.shape[0])):
        x = feed_info.loc[i, colname]
        if x != np.nan and x != '':
            y = str(x).strip().split(";")
        else:
            y = []
        keyword_array.at[i,colname] = y
    res = pd.concat((feed_info['feedid'], keyword_array), axis=1)
    return res

manual_keyword_list_use  = process_keywords('manual_keyword_list')
manual_tag_list = process_keywords('manual_tag_list')
machine_keyword_list = process_keywords('machine_keyword_list')

# 也可以根据概率的分布，截取概率的threshold
def process_machine_tag_list(colname):
    # data = feed_info[colname].str.split(';')
    keyword_array = pd.DataFrame(np.zeros((feed_info.shape[0], 1)), columns=[colname]).astype(object) 
    for i in tqdm(range(feed_info.shape[0])):
        x = feed_info.loc[i, colname]
        if x != np.nan and x != '' and x !=' ' and x != ';':
            y = [i.split(' ') for i in str(x).strip().split(";")]
            if y[0][0] == 'nan':
                y=[]
            else:
                y = pd.DataFrame(y).astype({0: np.int, 1: np.float32}).sort_values(by=1, ascending=False).reset_index(drop=True)

                if y[1][0]<0.5 and y[1][0]>0.25:
                    y=[y[0][1]]
                elif y[1][0]>=0.5:
                    y = y[y[1]>0.5][0].tolist()
                else:
                    y=[]
        else:
            y = []
        keyword_array.at[i,colname] = y
    res = pd.concat((feed_info['feedid'], keyword_array), axis=1)
    return res

machine_tag_list= process_machine_tag_list('machine_tag_list')





#d2v ----------------------------------------------------------------------------------
# text doc
text_feature = ['description','ocr','asr']

text_data = pd.read_csv('./data/wedata/all_text_data_20v.csv', index_col=0)

text_data = text_data.drop('feedid', axis=1)
text_data.columns = [i+'_d2v' for i in text_data.columns]



# # ## feed embedding




# #------------------


# # ## merge with train and test data


# merge keyword

feed_info_use = pd.concat([feed_info_use, keyword_data, text_data], axis=1)
feed_info_use = pd.concat([feed_info_use, manual_keyword_list_use.iloc[:,1], manual_tag_list.iloc[:,1], machine_keyword_list.iloc[:,1], machine_tag_list.iloc[:,1]], axis=1)



# # merge embedding
feed_embed_processed = pd.read_csv('./data/wedata/feed_embeddings_use_rd_32.csv', index_col=0)
# feed_embed_processed.columns = ['feed_embed'+str(i) for i in range(feed_embed_processed.shape[1]-1)] + ['feedid']
# feed_embed_feature = ['feed_embed'+str(i) for i in range(feed_embed_processed.shape[1])]
data_df = pd.merge(data_df, feed_embed_processed, on=['feedid'],how='left',copy=False)



data_df = pd.merge(data_df, feed_info_use[['feedid']+keyword_data.columns.tolist() +text_data.columns.tolist()+ keyword_feature ], on=['feedid'],how='left',copy=False)



# ### use MultiLabelBinarizer

keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']

keyword_feature_binary = data_df[keyword_feature]



from sklearn.preprocessing import MultiLabelBinarizer
def MultiLabelBinarizer_process(data, target):
    mlb = MultiLabelBinarizer(sparse_output=True)
    encoded = mlb.fit_transform(data[target])
    return encoded

manual_keyword_sparse = MultiLabelBinarizer_process(keyword_feature_binary, 'manual_keyword_list')
machine_keyword_sparse = MultiLabelBinarizer_process(keyword_feature_binary, 'machine_keyword_list')
manual_tag_sparse = MultiLabelBinarizer_process(keyword_feature_binary, 'manual_tag_list')
machine_tag_sparse = MultiLabelBinarizer_process(keyword_feature_binary, 'machine_tag_list')




# description_sparse


# # ### filter out low frequency word

cut_num = 500000  # 1000 10000 70000
print(sum(pd.Series(manual_keyword_sparse.sum(axis=0).tolist()[0]) > 200000),
     sum(pd.Series(machine_keyword_sparse.sum(axis=0).tolist()[0]) > 320000),
     sum(pd.Series(manual_tag_sparse.sum(axis=0).tolist()[0]) > 500000),
     sum(pd.Series(machine_tag_sparse.sum(axis=0).tolist()[0]) > 400000)
     )

manual_keyword_sparse = manual_keyword_sparse[:,(pd.Series(manual_keyword_sparse.sum(axis=0).tolist()[0]) > 200000).values]
machine_keyword_sparse = machine_keyword_sparse[:,(pd.Series(machine_keyword_sparse.sum(axis=0).tolist()[0]) > 320000).values]
manual_tag_sparse = manual_tag_sparse[:,(pd.Series(manual_tag_sparse.sum(axis=0).tolist()[0]) > 500000).values]
machine_tag_sparse = machine_tag_sparse[:,(pd.Series(machine_tag_sparse.sum(axis=0).tolist()[0]) > 400000).values]




manual_keyword_sparse_df = pd.DataFrame(manual_keyword_sparse.toarray(), columns=["manual_keyword_sparse_"+str(i) for i in range(manual_keyword_sparse.shape[1])]).astype(np.float32)
machine_keyword_sparse_df = pd.DataFrame(machine_keyword_sparse.toarray(), columns=["machine_keyword_sparse_"+str(i) for i in range(machine_keyword_sparse.shape[1])]).astype(np.float32)
manual_tag_sparse_df = pd.DataFrame(manual_tag_sparse.toarray(), columns=["manual_tag_sparse_"+str(i) for i in range(manual_tag_sparse.shape[1])]).astype(np.float32)
machine_tag_sparse_df = pd.DataFrame(machine_tag_sparse.toarray(), columns=["machine_tag_sparse_"+str(i) for i in range(machine_tag_sparse.shape[1])]).astype(np.float32)



data_df = pd.concat((data_df, 
                    manual_keyword_sparse_df,
                    machine_keyword_sparse_df,
                    manual_tag_sparse_df,
                    machine_tag_sparse_df),axis=1)



del manual_keyword_sparse_df, machine_keyword_sparse_df, manual_tag_sparse_df, machine_tag_sparse_df


data_df = data_df.drop(keyword_feature, axis=1)


# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']

play_cols = ['is_finish', 'play_times', 'play', 'stay']
drop_columns = ['play', 'stay']
keyword_feature = ['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']
text_feature = ['description','ocr','asr']

data_df = data_df.drop(play_cols, axis=1)

train_use_all = data_df.iloc[0:user_action_df.shape[0],:]
test_use_all = data_df.iloc[user_action_df.shape[0]:,:]
test_use_all['date_'] = 15 # 预测15号



train_use_all.shape, test_use_all.shape


# train_use_all.to_csv('./train_userfeed_keyword.csv')
# test_use_all.to_csv('./test_userfeed_keyword.csv')


# train_use = train_use.drop('date_', axis=1)
# test_use = test_use.drop('date_', axis=1)


ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
training_features = [i for i in data_df.columns if i not in FEA_COLUMN_LIST+['date_']]

categorical_feature = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id'] + [
    "manual_keyword_sparse_"+str(i) for i in range(manual_keyword_sparse.shape[1])] + [
    "machine_keyword_sparse_"+str(i) for i in range(machine_keyword_sparse.shape[1])] + [
    "manual_tag_sparse_"+str(i) for i in range(manual_tag_sparse.shape[1])] + [
    "machine_tag_sparse_"+str(i) for i in range(machine_tag_sparse.shape[1])] 


# categorical_feature = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id']
categorical_feature = [i for i, e in enumerate(data_df.columns.tolist()) if e in categorical_feature]



# # train

trn_x = train_use_all[train_use_all['date_'] < 14].reset_index(drop=True)
val_x = train_use_all[train_use_all['date_'] == 14].reset_index(drop=True)

from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc
import time
import random
import pickle
def savePkl(config, filepath):
    f = open(filepath, 'wb')
    pickle.dump(config, f)
    f.close()


def loadPkl(filepath):
    f = open(filepath, 'rb')
    config = pickle.load(f)
    return config
# callback
##################### 线下验证 #####################
seed_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 2049, 2050]
for item in range(10):

    seed = seed_list[item]

    uauc_list = []
    r_list = []
    for y in ['read_comment','like', 'click_avatar', 'forward']:
        print('=========', y, '=========')
        t = time.time()
        clf = loadPkl('./data/model/lgb_{}_{}.m'.format(y,item))
        test[y] += clf.predict_proba(test[y])[:, 1]/10


lgb1=test


lgb2=pd.read_csv('./sc2/lgb2.csv')

lgb3=pd.read_csv('./sc3/lgb3.csv')

# avg lgb
lgb_all = lgb1.copy()
for col in ['read_comment', 'like', 'click_avatar', 'forward']:
    lgb_all[col] = np.array([x[col].tolist() for x in [lgb1,lgb2,lgb3]]).mean(0)


# fank avg deepfm lgb

# [lgb1,lgb2,lgb3]  deepfm


files = os.listdir('.n-lgb/')
r = dict(
    zip(range(4), [x.sort_values(['userid', 'feedid']).reset_index(drop=True) for x in [deepfm, lgb_all]]))

weight_dict = {
    'read_comment': 0.5,
    'like': 0.5,
    'click_avatar': 0.5,
    'forward': 0.5,
}
submit = r[0].copy()
for col in ['read_comment', 'like', 'click_avatar', 'forward']:
    for k in r:
        r[k][col] = r[k][col].rank(ascending=False)
    submit[col] = np.array([x[col].tolist() for x in r.values()]).mean(0)
    mi, ma = submit[col].min(), submit[col].max()
    submit[col] = submit[col].apply(lambda x: (ma - x) / (ma - mi))

submit = deepfm[['userid','feedid']].merge(submit,on=['userid','feedid'],how='left')
submit.to_csv('./data/submission/result.csv.csv', index=None)
















