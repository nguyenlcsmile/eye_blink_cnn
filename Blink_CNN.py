import tensorflow as tf 
import numpy as np 
import vgg16

tf.compat.v1.disable_eager_execution() #Tránh xung đột giữa tensorflow 2 và 1 trong quá trình tính toán

#Built model
class BlinkCNN(object):
    ''' 
        CNN for eye blinking detection
    '''
    def __init__(self, is_train, img_size=[224, 224, 3], num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes
        self.is_train = is_train

        self.layers = {}
        self.params = {}

    def build(self):
        self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.img_size[0], self.img_size[1], self.img_size[2]])
        self.layers = vgg16.get_vgg16_network(self.input, self.params, self.num_classes, self.is_train)
        # print(self.layers)
        self.prob = self.layers.prob
        # print(self.prob)
        self.gt = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.var_list = tf.compat.v1.trainable_variables()

    #Compute loss
    def loss(self):
        self.net_loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=self.gt, logits=self.layers.fc8)
        self.net_loss = tf.compat.v1.reduce_mean(self.net_loss)
        tf.compat.v1.losses.add_loss(self.net_loss)

        #L2 weight regularize
        self.L2_loss = tf.compat.v1.reduce_mean([0.001*tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() if 'weights' in v.name])
        tf.compat.v1.losses.add_loss(self.L2_loss)
        self.total_loss = tf.compat.v1.losses.get_total_loss()

