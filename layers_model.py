import tensorflow as tf 
import numpy as np 
from tensorflow.python.ops import init_ops
import os 
import cv2 

tf.compat.v1.disable_eager_execution() #Tránh xung đột giữa tensorflow 2 và 1 trong quá trình tính toán

# =========================================
def use_bias_helper(bias_initializer):
    """
    Determine if a layer needs bias
    :param bias_initializer:
    :return:
    """
    if bias_initializer is None:
        return False
    else:
        return True

# ============================================
#Xây dựng các hàm activation sigmod và relu 
def activate(input, name, act_type='relu'):
    with tf.compat.v1.variable_scope(name) as scope:
        if act_type == 'relu':
            out = tf.compat.v1.nn.relu(input)
        elif act_type == 'sigmod':
            out = tf.compat.v1.nn.sigmoid(input)
        else:
            raise ValueError('act_type is not valid.')
        
        return out 

# ============================================
#Hàm Conv2D
def conv2D(input, shape, name, padding='SAME', strides=(1, 1),
        weights_initializer = tf.compat.v1.initializers.glorot_uniform(),
        bias_initializer = init_ops.zeros_initializer(),
        weights_regularizer = None,
        bias_regularizer = None,
        params={}):

    use_bias = use_bias_helper(bias_initializer)
    with tf.compat.v1.variable_scope(name) as scope:
        channel = input.get_shape().as_list()[-1]
        kernel = tf.compat.v1.get_variable(
            name = 'weights',
            shape = [shape[0], shape[1], channel, shape[2]], #(3, 3, 3, 64)
            dtype = tf.float32,
            initializer = weights_initializer,
            regularizer = weights_regularizer
        )
        strides = [1, strides[0], strides[1], 1]
        out = tf.compat.v1.nn.conv2d(input, kernel, strides=strides, padding=padding)

        bias = None
        if use_bias:
            bias = tf.compat.v1.get_variable(
                name = 'biases', 
                shape = [shape[2]], 
                dtype=tf.float32, 
                initializer = bias_initializer, 
                regularizer = bias_regularizer
            )
            out = tf.compat.v1.nn.bias_add(out, bias)

        # print('{} weights: {}, bias: {}, out: {}'.format(name, kernel, bias, out))
        params[name] = [kernel, bias]
    
    return out 

# ============================================
#Hàm MaxPooling2D
def max_pool(input, name, ksize = (2, 2), 
            strides = (2, 2), padding = 'SAME'):
    with tf.compat.v1.variable_scope(name) as scope:
        ksize = [1, ksize[0], ksize[1], 1]
        strides = [1, strides[0], strides[1], 1]
        out = tf.compat.v1.nn.max_pool(input, ksize = ksize, strides = strides, padding = padding)
        # print('{} max pool out: {}'.format(name, out))
    
    return out 

# ============================================
#Hàm Avg_Pooling2D
def avg_pool(input, name, ksize = (2, 2), 
            strides = (2, 2), padding = 'SAME'):
    
    with tf.compat.v1.variable_scope(name) as scope:
        ksize = [1, ksize[0], ksize[1], 1]
        strides = [1, strides[0], strides[1], 1]
        out = tf.compat.v1.nn.avg_pool(input, ksize = ksize, strides = strides, padding=padding)
        # print('{} avg pool out: {}'.format(name, out))

    return out

# ============================================
#Hàm Dense (Fully Connected)
def fully_connected(input, num_neuron, name, 
                    weights_initializer = tf.compat.v1.initializers.glorot_uniform(), 
                    bias_initializer = init_ops.zeros_initializer(), 
                    weights_regularizer = None, 
                    bias_regularizer = None, 
                    params = {}):
    
    use_bias = use_bias_helper(bias_initializer)
    with tf.compat.v1.variable_scope(name) as scope:
        input_dim = int(np.prod(input.get_shape().as_list()[1:]))
        kernel = tf.compat.v1.get_variable(
            name = 'weights', 
            shape = [input_dim, num_neuron], 
            dtype = tf.float32, 
            initializer = weights_initializer, 
            regularizer = weights_regularizer
        )

        flat = tf.compat.v1.reshape(input, [-1, input_dim])
        out = tf.compat.v1.matmul(flat, kernel)

        bias = None 
        if use_bias:
            bias = tf.compat.v1.get_variable(
                name = 'biases', 
                shape = num_neuron, 
                dtype = tf.float32, 
                initializer = bias_initializer, 
                regularizer = bias_regularizer
            )
            out = tf.nn.bias_add(out, bias)
        
        # print('{} weights: {}, bias: {}, out: {}'.format(name, kernel, bias, out))
        params[name] = [kernel, bias]

    return out 

# ============================================
def list_vars_in_ckpt(path):
    """List all variables in checkpoint"""
    saved_vars = tf.compat.v1.train.list_variables(path)
    # print(saved_vars)
    return saved_vars
    
# ============================================
def get_restore_var_list(path):
    ''' 
        Get variable list when restore from ckpt. This is mainly for transferring model to another network
    '''
    global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES) #Variables in graph
    # print(global_vars)
    saved_vars = list_vars_in_ckpt(path)
    saved_vars_name = [var[0] for var in saved_vars]
    # print(saved_vars_name)
    restore_var_list = [var for var in global_vars if var.name[:-2] in saved_vars_name] # or 'vgg_' + var.name[:-2] in saved_vars_name]
    # print(restore_var_list)
    return restore_var_list