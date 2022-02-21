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


class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
#                 print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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


def MMOE(dnn_feature_columns, num_tasks, task_types, task_names, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),tower_dnn_units_lists=[[64,32,8],[64,32]],
         l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',dnn_use_bn = True):
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
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_out)
    
    task_outputs = []
    for task_type, task_name, tower_dnn, mmoe_out in zip(task_types, task_names, tower_dnn_units_lists, mmoe_outs):
        #build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(mmoe_out)
        
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outputs.append(output)


    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model

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


def Shared_Bottom(dnn_feature_columns, num_tasks, task_types, task_names,
                  bottom_dnn_units=[512, 512,512], tower_dnn_units_lists=[[64,32], [64,32]],
                  l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,dnn_activation='relu', dnn_use_bn=False):
    """Instantiates the Shared_Bottom multi-task learning Network architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks:  integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    :param bottom_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param tower_dnn_units_lists: list, list of positive integer list, its length must be euqal to num_tasks, the layer number and units in each layer of task-specific DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :return: A Keras model instance.
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")
        
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
            
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,seed)
    
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    
    print(dnn_input.shape)
    shared_bottom_output = DNN(bottom_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    tasks_output = []
    for task_type, task_name, tower_dnn in zip(task_types, task_names, tower_dnn_units_lists):
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(shared_bottom_output)
        
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit) #regression->keep, binary classification->sigmoid
        tasks_output.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=tasks_output)
    return model



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



class CGC_layer(Layer):
    def __init__(self,num_tasks,num_experts_specific,num_experts_shared,expert_dnn_units,gate_dnn_units,dnn_activation = 'relu', l2_reg_dnn = 1e-5, dnn_dropout = 0, dnn_use_bn = False, seed=100,**kwargs):
        self.num_tasks = num_tasks
        self.num_experts_specific = num_experts_specific
        self.num_experts_shared = num_experts_shared
        self.expert_dnn_units = expert_dnn_units
        self.gate_dnn_units = gate_dnn_units
        self.dnn_activation = dnn_activation
        self.l2_reg_dnn = l2_reg_dnn
        self.dnn_dropout = dnn_dropout
        self.dnn_use_bn = dnn_use_bn
        self.seed = seed
        
        super(CGC_layer, self).__init__(**kwargs)
        
    def build(self,input_shape):
        
        input_dim = int(input_shape[-1])
        self.experts = []
        self.gate_networks = []
        #build task-specific expert layer
        for i in range(self.num_tasks):
            for j in range(self.num_experts_specific):
                expert_network = DNN(self.expert_dnn_units, self.dnn_activation, self.l2_reg_dnn, self.dnn_dropout, self.dnn_use_bn, seed=self.seed)
                self.experts.append(expert_network)
            gate_network = DNN(self.gate_dnn_units, self.dnn_activation, self.l2_reg_dnn, self.dnn_dropout,self.dnn_use_bn, seed=self.seed)
            self.gate_networks.append(gate_network)
        #build task-shared expert layer
        for i in range(self.num_experts_shared):
            expert_network = DNN(self.expert_dnn_units, self.dnn_activation, self.l2_reg_dnn,self.dnn_dropout, self.dnn_use_bn, seed=self.seed, name='expert_shared_'+str(i))
            self.experts.append(expert_network)
        
        super(CGC_layer,self).build(input_shape)
        
        
    def call(self,dnn_input,**kwargs):
        
        expert_outputs = []
        for expert  in self.experts:
            expert_outputs.append(expert(dnn_input)) 
              
        #build one Extraction Layer
        cgc_outs = []
        for i in range(self.num_tasks): 
            #concat task-specific expert and task-shared expert
            cur_expert_num = self.num_experts_specific + self.num_experts_shared
            cur_experts = expert_outputs[i * self.num_experts_specific:(i + 1) * self.num_experts_specific] + expert_outputs[-int(self.num_experts_shared):] #task_specific + task_shared
            expert_concat = tf.keras.layers.concatenate(cur_experts, axis=1, name='expert_concat_'+str(i))
            expert_concat = tf.keras.layers.Reshape([cur_expert_num, self.expert_dnn_units[-1]], name='expert_reshape_'+str(i))(expert_concat)

            #build gate layers
            gate_network = self.gate_networks[i]
            gate_input = gate_network(dnn_input)
            
            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax', name='gate_softmax_'+str(i))(gate_input)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=-1), [1, 1, self.expert_dnn_units[-1]]) 

            #gate multiply the expert
            gate_mul_expert = tf.keras.layers.Multiply(name='gate_mul_expert_'+str(i))([expert_concat, gate_out]) 
            gate_mul_expert = tf.math.reduce_sum(gate_mul_expert, axis=1) #sum pooling in the expert ndim
            cgc_outs.append(gate_mul_expert)

        return cgc_outs
    
def PLE_CGC(dnn_feature_columns, num_tasks, task_types, task_names, num_experts_specific=8, num_experts_shared=4,
            expert_dnn_units=[64,64],  gate_dnn_units=None, tower_dnn_units_lists=[[16,16],[16,16]],
            l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """Instantiates the Customized Gate Control block of Progressive Layered Extraction architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    
    :param num_experts_specific: integer, number of task-specific experts.
    :param num_experts_shared: integer, number of task-shared experts.
    :param expert_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of expert DNN
    :param gate_dnn_units: list, list of positive integer or None, the layer number and units in each layer of gate DNN, default value is None. e.g.[8, 8].
    :param tower_dnn_units_lists: list, list of positive integer list, its length must be euqal to num_tasks, the layer number and units in each layer of task-specific DNN
    
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :return: a Keras model instance
    """
    
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")
        
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
    cgc_layer = CGC_layer(num_tasks,
                          num_experts_specific,
                          num_experts_shared,
                          expert_dnn_units,
                          gate_dnn_units,
                          dnn_activation = 'relu', 
                          l2_reg_dnn = 1e-5, 
                          dnn_dropout = 0, 
                          dnn_use_bn = False,
                          seed=100) 
    
    cgc_outs = cgc_layer(dnn_input)
    print(cgc_outs[0].shape)

    task_outs = []
    for task_type, task_name, tower_dnn, cgc_out in zip(task_types, task_names, tower_dnn_units_lists, cgc_outs):
        #build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(cgc_out)
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit) 
        task_outs.append(output)
        
    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model



def PLE_CGC_FEFM(dnn_feature_columns, num_tasks, task_types, task_names, num_experts_specific=8, num_experts_shared=4,
            expert_dnn_units=[64,64],  gate_dnn_units=None, tower_dnn_units_lists=[[16,16],[16,16]],
            l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,l2_reg_embedding_field=0.00001,
            fm_group=[DEFAULT_GROUP_NAME],use_fefm_embed_in_dnn = True,exclude_feature_embed_in_dnn = False):
    """Instantiates the Customized Gate Control block of Progressive Layered Extraction architecture.
    
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    
    :param num_experts_specific: integer, number of task-specific experts.
    :param num_experts_shared: integer, number of task-shared experts.
    :param expert_dnn_units: list, list of positive integer, its length must be greater than 1, the layer number and units in each layer of expert DNN
    :param gate_dnn_units: list, list of positive integer or None, the layer number and units in each layer of gate DNN, default value is None. e.g.[8, 8].
    :param tower_dnn_units_lists: list, list of positive integer list, its length must be euqal to num_tasks, the layer number and units in each layer of task-specific DNN
    
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :return: a Keras model instance
    """
    
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")
        
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
            
    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)
    dnn_input = combined_dnn_input(list(chain.from_iterable(group_embedding_dict.values())), dense_value_list)
    #######################################################
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

    cgc_layer = CGC_layer(num_tasks,
                          num_experts_specific,
                          num_experts_shared,
                          expert_dnn_units,
                          gate_dnn_units,
                          dnn_activation = 'relu', 
                          l2_reg_dnn = 1e-5, 
                          dnn_dropout = 0, 
                          dnn_use_bn = False,
                          seed=100) 
    
    cgc_outs = cgc_layer(dnn_input)
    print(cgc_outs[0].shape)

    task_outs = []
    for task_type, task_name, tower_dnn, cgc_out in zip(task_types, task_names, tower_dnn_units_lists, cgc_outs):
        #build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(cgc_out)
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        final_logit = add_func([fefm_logit, logit])
        output = PredictionLayer(task_type, name=task_name)(final_logit) 
        task_outs.append(output)
        
    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
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