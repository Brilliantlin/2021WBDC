# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from deepctr.layers.core import PredictionLayer,DNN
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.regularizers import l2
from deepctr.layers.activation import activation_layer
from itertools import chain
from deepctr.feature_column import *
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input,reduce_sum
from deepctr.layers.interaction import FEFMLayer

def MMOE_FefM(dnn_feature_columns, num_tasks, task_types, task_names, num_experts=4, expert_dim=8,l2_reg_linear=1e-5,dnn_hidden_units=(128, 128),tower_dnn_units_lists=[[64,32,8],[64,32]],
         l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',dnn_use_bn = True,l2_reg_embedding_field=0.00001,
        fm_group=[DEFAULT_GROUP_NAME],use_fefm_embed_in_dnn = True,exclude_feature_embed_in_dnn = False):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")
    
    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())
    
    linear_logit = get_linear_logit(features, [x for x in dnn_feature_columns if isinstance(x,DenseFeat)], seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)
    dnn_input = combined_dnn_input(list(chain.from_iterable(group_embedding_dict.values())), dense_value_list)
    #####################################################################################################################################
    
    fefm_interaction_embedding = concat_func([FEFMLayer(
        regularizer=l2_reg_embedding_field)(concat_func(v, axis=1))
                                              for k, v in group_embedding_dict.items() if k in [DEFAULT_GROUP_NAME]] ,
                                             axis=1)
    
    if use_fefm_embed_in_dnn:
        if exclude_feature_embed_in_dnn:
            # Ablation3: remove feature vector embeddings from the DNN input
            dnn_input = fefm_interaction_embedding
        else:
            # No ablation
            dnn_input = concat_func([dnn_input, fefm_interaction_embedding], axis=1)
    print('dnn input shape',dnn_input.shape)
    fefm_logit = tf.keras.layers.Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=True))(fefm_interaction_embedding)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed )(dnn_input)
    
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_out)
    
    task_outputs = []
    for task_type, task_name, tower_dnn, mmoe_out in zip(task_types, task_names, tower_dnn_units_lists, mmoe_outs):
        #build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name )(mmoe_out)
        
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        final_logit = add_func([linear_logit, fefm_logit, logit])
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outputs.append(output)
    
    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model
class MMOELayer(Layer):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size,units)``.  
      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, output_dim)`` .
      Arguments
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **output_dim**: integer, the dimension of each output of MMOELayer.
    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, num_tasks, num_experts, output_dim, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.seed = seed
        super(MMOELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(
            name='expert_kernel',
            shape=(input_dim, self.num_experts * self.output_dim),
            dtype=tf.float32,
            initializer=glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                name='gate_weight_'.format(i),
                shape=(input_dim, self.num_experts),
                dtype=tf.float32,
                initializer=glorot_normal(seed=self.seed)))
        super(MMOELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0))
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = tf.tensordot(inputs, self.gate_kernels[i], axes=(-1, 0))
            gate_out = tf.nn.softmax(gate_out)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs.append(output)
        return outputs

    def get_config(self):

        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(MMOELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks
    
    
def MMOE_mutihead(dnn_feature_columns, num_tasks, task_types, task_names, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),tower_dnn_units_lists=[[64,32,8],[64,32]],
         l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',dnn_use_bn = True,multi_head_num = 2):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :return: a Keras model instance
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")


    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    print('dnn input shape',dnn_input.shape)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)
    mmoe_outs_list = [MMOELayer(num_tasks, num_experts, expert_dim,seed = i)(dnn_out) for i in range(multi_head_num)]

    task_outputs = []
    for i, (task_type, task_name, tower_dnn) in  enumerate(zip(task_types, task_names, tower_dnn_units_lists)):
        #build tower layer
        mmoe_out = tf.keras.layers.average([outs[i] for outs in mmoe_outs_list])
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(mmoe_out)
        
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outputs.append(output)


    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model





def MMOE_FefM_multihead(dnn_feature_columns, num_tasks, task_types, task_names, num_experts=4, expert_dim=8,
                        l2_reg_linear=1e-5,dnn_hidden_units=(128, 128),tower_dnn_units_lists=[[64,32,8],[64,32]],
                        multi_head_num = 5,
         l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',dnn_use_bn = True,l2_reg_embedding_field=0.00001,
        fm_group=[DEFAULT_GROUP_NAME],use_fefm_embed_in_dnn = True,exclude_feature_embed_in_dnn = False):

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")
    
    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())
    
    linear_logit = get_linear_logit(features, [x for x in dnn_feature_columns if isinstance(x,DenseFeat)], seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)
    dnn_input = combined_dnn_input(list(chain.from_iterable(group_embedding_dict.values())), dense_value_list)
    #####################################################################################################################################
    
    fefm_interaction_embedding = concat_func([FEFMLayer(
        regularizer=l2_reg_embedding_field)(concat_func(v, axis=1))
                                              for k, v in group_embedding_dict.items() if k in [DEFAULT_GROUP_NAME]] ,
                                             axis=1)
    
    if use_fefm_embed_in_dnn:
        if exclude_feature_embed_in_dnn:
            # Ablation3: remove feature vector embeddings from the DNN input
            dnn_input = fefm_interaction_embedding
        else:
            # No ablation
            dnn_input = concat_func([dnn_input, fefm_interaction_embedding], axis=1)
    print('dnn input shape',dnn_input.shape)
    fefm_logit = tf.keras.layers.Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=True))(fefm_interaction_embedding)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed )(dnn_input)
    mmoe_outs_list = [MMOELayer(num_tasks, num_experts, expert_dim,seed = i)(dnn_out) for i in range(multi_head_num)]
    
    task_outputs = []
    for i, (task_type, task_name, tower_dnn) in  enumerate(zip(task_types, task_names, tower_dnn_units_lists)):
        #build tower layer
        mmoe_out = tf.keras.layers.average([outs[i] for outs in mmoe_outs_list])
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name )(mmoe_out)
        
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        final_logit = add_func([fefm_logit, logit])
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outputs.append(output)
    
    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model