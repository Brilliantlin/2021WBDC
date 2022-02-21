
import json
import time
import os

class MyArgparse(object):
        __instance = None
        ##单例模式
        def __new__(cls, *args, **kwargs):  # 这里不能使用__init__，因为__init__是在instance已经生成以后才去调用的
            if cls.__instance is None:
                cls.__instance = super(MyArgparse, cls).__new__(cls, *args, **kwargs)
            return cls.__instance
        """配置参数"""
        def __init__(self,):
            self.debug = False
            self.model_name = 'deepfm'
            self.times = time.strftime('%Y-%m-%d.%H:%M:%S', time.localtime(time.time()))
            # self.times = '2021-05-08.14:11:32'
    
            ####数据文件地址######
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.SRC_DIR = os.path.abspath(os.path.join(self.BASE_DIR, '..'))
            self.ROOT_PATH = os.path.join(self.SRC_DIR, 'data/')
            self.DATASET_PATH = os.path.join(self.ROOT_PATH,'wedata/wechat_algo_data2/')# 比赛数据集路径
            
            
            # 训练集
            self.USER_ACTION = self.DATASET_PATH + "user_action.csv"
            self.FEED_INFO = self.DATASET_PATH + "feed_info.csv"
            self.FEED_EMBEDDINGS = self.DATASET_PATH + "feed_embeddings.csv"
            
            #测试集
            self.TEST_FILE = self.DATASET_PATH + "test_a.csv"#测试集
            
            ###emb特征路径
            self.fixlen_feature_columns = self.ROOT_PATH+'wedata/fixlen_feature_columns_all.pkl'
            self.emb_path = self.ROOT_PATH+'wedata/emb/' ##graph_emb8、feed_emb_16 、weight_emb8、 keyword_w2v_8、 userid_feedid_d2v_all_16、 all_text_data_v8、 userid_authorid_d2v_all_16
            self.emb_all_path = self.ROOT_PATH+'wedata/emb_all/'
            self.encoder_path = self.ROOT_PATH+'wedata/encoder/'
            self.encoder_all_path = self.ROOT_PATH+'wedata/encoder_all/'
            self.history_feat_path = self.ROOT_PATH+'wedata/stat_data/'
            self.train_history_feat_path = self.ROOT_PATH+'wedata/train_feat/'
            
            ##模型保存路径
            self.model_path =  os.path.join(self.ROOT_PATH,'model/deepfm/')

            
    
            ####使用字段
            self.FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]#复赛预测列
            self.sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
            self.dense_features = ['videoplayseconds']
            
            ###输出结果文件路径
            self.model_batch_path = self.ROOT_PATH + 'model/'
            self.submit_path = self.ROOT_PATH + 'submission/'
            self.feat_path = self.ROOT_PATH +'feat/'
    



    
            ####预训练模型######
    
    
            ####模型结构######
            self.embedding_dims = {action:10 for action in self.FEA_COLUMN_LIST}
            self.model = 'deepFM'
    
            ####训练参数######
            self.max_grad_norm = 10
            self.accumulation_steps = 1  # 梯度累计
            self.early_stop_epochs = 3  # 早停轮数
            self.dropout = 0.5  # dropout
            self.num_train_epochs = 10  # epoch数
            self.batch_size = 512  # mini-batch大小
            self.fold_num = 10 #折数
            self.fp16 = False
            self.optim = 'adam'#adamW、swa 、sgd_adam、sgd、adam、adagrad、rmsprop
    
    
            ####优化器相关######
            self.learning_rate = 0.01  # 学习率
            self.warmup_proportion = 0.1
            self.adam_epsilon = 1e-8
            self.correct_bias = True
            self.weight_decay = 0.01
            self.max_grad_norm = 1.0
            self.cycle_epoch = 4 ###快照集成每个周期的轮数
            self.scheduler_on_batch = True##每个batch进行学习率调整
    
            ####训练trick######
            self.adv = 'fgm' #fgm、pgd   todo:freelb,freeAT。。。
            self.lookahead = False
            self.sda = False
            self.swa = False
            self.snap = False
            self.do_ema = False
            self.ema_decay = 0.9#
    
            ####其他######
            self.seed = 2021
            self.run_cv_fold_num = 1#交叉验证跑几折
            ####训练结果记录######
            self.record = {}
    
            
#         def makdir(self,model_name):
#             self.base_sava_path = './output/{}'.format(self.times)+'run_fold_{}_{}'.format(self.run_cv_fold_num,model_name)
#             self.model_sava_dir = self.base_sava_path + '/model_save/'  # 模型保存路径
#             self.reulst_dir = self.base_sava_path + '/prediction_result/'  # 结果保存路径
#             self.config_dir = self.base_sava_path + '/config/'  # config保存路径
#             self.feat_dir = self.base_sava_path + '/feat/'

#             ####创建相关文件夹####
#             if not self.debug:
#                 if not os.path.exists(self.base_sava_path): os.makedirs(self.base_sava_path)
#                 if not os.path.exists(self.model_sava_dir): os.makedirs(self.model_sava_dir)
#                 if not os.path.exists(self.reulst_dir): os.makedirs(self.reulst_dir)
#                 if not os.path.exists(self.config_dir): os.makedirs(self.config_dir)
#                 if not os.path.exists(self.feat_dir): os.makedirs(self.feat_dir)
#                 print('base_sava_path地址为:  {}'.format(self.base_sava_path))
        def __repr__(self):
            return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
    
        def __str__(self):
            return self.__repr__()
        def save_config(self,path):
            f = open(path,'w', encoding='utf-8')
            f.write(json.dumps(self.__dict__, ensure_ascii=False, indent=1))
            f.close()
        def load_config(self,path):
            f = open(path, 'r', encoding='utf-8')
            data = json.load(f)
            f.close()
            for attr in data.keys():
                setattr(self, attr, data[attr])
            return data
