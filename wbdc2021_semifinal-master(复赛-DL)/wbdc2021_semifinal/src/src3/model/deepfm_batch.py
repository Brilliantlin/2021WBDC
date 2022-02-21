from lr_scheduler import get_linear_schedule_with_warmup
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
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
import pickle
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import time
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import traceback
import logging
from numba import njit
from scipy.stats import rankdata


@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)

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
            # auc = fast_auc(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc


def compute_weighted_score(score_dict, weight_dict):
    '''基于多个行为的uAUC值，计算加权uAUC
    Input:
        scores_dict: 多个行为的uAUC值映射字典, dict
        weights_dict: 多个行为的权重映射字典, dict
    Output:
        score: 加权uAUC值, float
    '''
    score = 0.0
    weight_sum = 0.0
    for action in score_dict:
        weight = float(weight_dict[action])
        score += weight*score_dict[action]
        weight_sum += weight
    score /= float(weight_sum)
    # score = round(score, 6)
    return score


def evaluate_deepctr(eval_dict):
    weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                   "comment": 1, "follow": 1}
    weight_auc = compute_weighted_score(eval_dict, weight_dict)
    print("Weighted uAUC: ", weight_auc)
    return weight_auc
def evaluate_auc(eval_dict):
    weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                   "comment": 1, "follow": 1}
    weight_auc = compute_weighted_score(eval_dict, weight_dict)
    print("Weighted uAUC: ", weight_auc)
    return weight_auc


import warnings
warnings.filterwarnings("ignore")
from lr_scheduler import get_linear_schedule_with_warmup
class MyBaseModel(BaseModel):
    def fit(self,args, train_loader, batch_size=None, epochs=2, verbose=1,early_stop = 2,
              model_save_path = 'model_sava/deepfm.pth',user_list = None, initial_epoch=1, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        model_paths = model_save_path.split('/')
        do_validation = True
        if len(validation_data) == 2:
            val_x, val_y = validation_data
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

#         train_loader = DataLoader(
#             dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size,num_workers = 8)

        sample_num = len(train_loader)*batch_size
        steps_per_epoch = (sample_num - 1) // batch_size + 1
        scheduler = self.build_lr_scheduler(optim,steps_per_epoch*args.num_train_epochs)
        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        #Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
           sample_num, len(val_y), len(train_loader)))
        logger.info("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            sample_num, len(val_y), len(train_loader)))
        best_auc = 0
        best_epoch = initial_epoch
        no_improve_epoch = 0
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
                        scheduler.step()
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
#             for name, result in train_result.items():
#                 epoch_logs[name] = np.sum(result) / steps_per_epoch
            if epoch >=4:
                path = model_save_path[0:-4]+"_epoch_{}".format(epoch)+'.pth'
                torch.save(model.state_dict(),path)
            if do_validation:
                eval_result = self.evaluate(val_x, val_y, user_list,batch_size)
                current_auc = eval_result['uAUC']
                # print('current_uAUC :{}  best_uAUC :{} best epoch :{}'.format(current_auc,best_auc,best_epoch))
                if current_auc>=best_auc:
                    best_epoch = epoch
                    best_auc = current_auc
                    torch.save(model.state_dict(),model_save_path)
                    no_improve_epoch = 0
                else:
                    no_improve_epoch+=1
                if no_improve_epoch==early_stop:
                    break

                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch, epochs))
                logger.info('Epoch {0}/{1}'.format(epoch + 1, epochs))
                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

#                 for name in self.metrics:
#                     eval_str += " - " + name + \
#                                 ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])

                print(eval_str)
                logger.info(eval_str)
                # print('best_uAUC :{} best epoch :{}'.format(best_auc, best_epoch))
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break
        if do_validation:
            print('best_uAUC :{} best epoch :{}'.format(best_auc, best_epoch))
            logger.info('best_uAUC :{} best epoch :{}'.format(best_auc, best_epoch))
        callbacks.on_train_end()

        return self.history,best_auc

    def evaluate(self, test_loader, y, user_list = None,batch_size=256):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(test_loader)
        eval_result = {}
        uAUC_score = 0
        acc = 0
        log_loss_score = 1000
        for name, metric_fun in self.metrics.items():
            try:
                if name =='uAUC':
                    uAUC_score = metric_fun(y.cpu().data.numpy(), pred_ans,user_list)
                if name =='acc':
                    acc = metric_fun(y.cpu().data.numpy(), pred_ans,user_list)
                if name=='logloss':
                    log_loss_score = metric_fun(y.cpu().data.numpy(), pred_ans)
            except Exception :
                pass
            finally:
                eval_result['uAUC'] = uAUC_score
                eval_result['acc'] = acc
                eval_result['binary_crossentropy'] = log_loss_score
                return eval_result


    def predict(self, test_loader):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
       

#         tensor_data = Data.TensorDataset(x)
#         test_loader = DataLoader(
#             dataset=tensor_data, shuffle=False, batch_size=batch_size,num_workers = 8)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test.to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")


    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        # self.optim = self._get_optim(optimizer)
        self.optim = self.build_optimizer(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)
    def build_optimizer(self, optim='adam'):
        '''
        Setup the optimizer.
        '''
        if isinstance(optim, str):
            if optim == "sgd":
                optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optim == "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)  # 0.001
            elif optim == "adagrad":
                optimizer = torch.optim.Adagrad(self.parameters())
                # optimizer = torch.optim.Adagrad(optimizer_grouped_parameters, lr=self.args.learning_rate)  # 0.01
            elif optim == "rmsprop":
                optimizer = torch.optim.RMSprop(self.parameters(), lr=args.learning_rate)
            else:
                raise NotImplementedError
        else:
            optimizer = optim
        return optimizer

    def build_lr_scheduler(self, optimizer, t_total):
        '''
        the learning rate scheduler.
        '''
        warmup_steps = int(t_total * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
#         if self.args.snap:
#             ##余弦退火
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
#                                                                              T_0=int(t_total / (self.args.num_train_epochs/self.args.cycle_epoch)),
#                                                                              T_mult=1,
#                                                                              eta_min=1e-7,
#                                                                              last_epoch=-1)
        return scheduler
    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                if metric == 'uAUC':
                    metrics_[metric] = uAUC
                self.metrics_names.append(metric)
        return metrics_
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
