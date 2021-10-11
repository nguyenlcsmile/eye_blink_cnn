import tensorflow as tf 
import layers_model
from easydict import EasyDict as edict

# tf.compat.v1.disable_eager_execution() #Tránh xung đột giữa tensorflow 2 và 1 trong quá trình tính toán

def get_vgg16_conv5(input, params):
    layers = edict()

    layers.conv1_1 = layers_model.conv2D(input=input, shape=(3, 3, 64), name='conv1_1', params=params)
    layers.conv1_1_relu = layers_model.activate(input=layers.conv1_1, name='conv1_1_relu', act_type='relu')
    layers.conv1_2 = layers_model.conv2D(input=layers.conv1_1_relu, shape=(3, 3, 64), name='conv1_2', params=params)
    layers.conv1_2_relu = layers_model.activate(input=layers.conv1_2, name='conv1_2_relu', act_type='relu')
    layers.pool1 = layers_model.max_pool(input=layers.conv1_2_relu, name='pool1')

    layers.conv2_1 = layers_model.conv2D(input=layers.pool1, shape=(3, 3, 128), name='conv2_1', params=params)
    layers.conv2_1_relu = layers_model.activate(input=layers.conv2_1, name='conv2_1_relu', act_type='relu')
    layers.conv2_2 = layers_model.conv2D(input=layers.conv2_1_relu, shape=(3 , 3, 128), name='conv2_2', params=params)
    layers.conv2_2_relu = layers_model.activate(input=layers.conv2_2, name='conv2_2_relu', act_type='relu')
    layers.pool2 = layers_model.max_pool(input=layers.conv2_2_relu, name='pool2')

    layers.conv3_1 = layers_model.conv2D(input=layers.pool2, shape=(3, 3, 256), name='conv3_1', params=params)
    layers.conv3_1_relu = layers_model.activate(input=layers.conv3_1, name='conv3_1_relu', act_type='relu')
    layers.conv3_2 = layers_model.conv2D(input=layers.conv3_1_relu, shape=(3 , 3, 256), name='conv3_2', params=params)
    layers.conv3_2_relu = layers_model.activate(input=layers.conv3_2, name='conv3_2_relu', act_type='relu')
    layers.pool3 = layers_model.max_pool(input=layers.conv3_2_relu, name='pool3')

    layers.conv4_1 = layers_model.conv2D(input=layers.pool3, shape=(3, 3, 512), name='conv4_1', params=params)
    layers.conv4_1_relu = layers_model.activate(input=layers.conv4_1, name='conv4_1_relu', act_type='relu')
    layers.conv4_2 = layers_model.conv2D(input=layers.conv4_1_relu, shape=(3 , 3, 512), name='conv4_2', params=params)
    layers.conv4_2_relu = layers_model.activate(input=layers.conv4_2, name='conv4_2_relu', act_type='relu')
    layers.pool4 = layers_model.max_pool(input=layers.conv4_2_relu, name='pool4')

    layers.conv5_1 = layers_model.conv2D(input=layers.pool4, shape=(3, 3, 512), name='conv5_1', params=params)
    layers.conv5_1_relu = layers_model.activate(input=layers.conv5_1, name='conv5_1_relu', act_type='relu')
    layers.conv5_2 = layers_model.conv2D(input=layers.conv5_1_relu, shape=(3 , 3, 512), name='conv5_2', params=params)
    layers.conv5_2_relu = layers_model.activate(input=layers.conv5_2, name='conv5_2_relu', act_type='relu')
    layers.conv5_3 = layers_model.conv2D(input=layers.conv5_2_relu, shape=(3 , 3, 512), name='conv5_3', params=params)
    layers.conv5_3_relu = layers_model.activate(input=layers.conv5_3, name='conv5_3_relu', act_type='relu')
    layers.pool5 = layers_model.max_pool(input=layers.conv5_3_relu, name='pool5')

    return layers

def get_vgg16_network(input, params, num_class=1000, is_train=True):
    #Get conv5 and pool5
    layers = get_vgg16_conv5(input, params)

    layers.fc6 = layers_model.fully_connected(input=layers.pool5, num_neuron=4096, name='fc6', params=params)
    if is_train:
        layers.fc6 = tf.compat.v1.nn.dropout(layers.fc6, rate=0.5)
    layers.fc6_relu = layers_model.activate(input=layers.fc6, act_type='relu', name='fc6_relu')

    layers.fc7 = layers_model.fully_connected(input=layers.fc6_relu, num_neuron=2048, name='fc7', params=params)
    if is_train:
        layers.fc7 = tf.compat.v1.nn.dropout(layers.fc7, rate=0.5)
    layers.fc7_relu = layers_model.activate(input=layers.fc7, act_type='relu', name='fc7_relu')

    layers.fc8 = layers_model.fully_connected(input=layers.fc7_relu, num_neuron=num_class, name='fc8', params=params)
    layers.prob = tf.compat.v1.nn.softmax(layers.fc8)

    return layers