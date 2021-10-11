import tensorflow as tf 
import numpy as np 
import cv2 
import os 
import layers_model

tf.compat.v1.disable_eager_execution() #Tránh xung đột giữa tensorflow 2 và 1 trong quá trình tính toán

class Solver(object):
    ''' 
        Solver for training and testing
    '''
    def __init__(self, sess, net, img_size=[224, 224, 3], num_classes=2, summary_dir = "summary/", model_dir = 'ckpt_CNN/'):
        self.sess = sess
        self.net = net
        self.num_classes = num_classes
        self.img_size = img_size
        self.summary_dir = summary_dir
        self.model_dir = model_dir

    def init(self):
        #Tạo folder chứa weight training
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        #Tạo folder chứa dữ liệu vẽ tensorboard
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)

        self.model_path = os.path.join(self.model_dir, 'model.ckpt')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.saver = tf.compat.v1.train.Saver()

        #Initialize the graph
        if self.net.is_train:
            self.num_epoch = 100
            self.learning_rate = 0.01
            self.decay_rate = 0.9
            self.decay_step = 200
            self.net.loss()
            self.set_optimizer()
            # Add summary
            self.loss_summary = tf.compat.v1.summary.scalar('loss_summary', self.net.total_loss)
            self.lr_summary = tf.compat.v1.summary.scalar('learning_rate_summary', self.LR)
            # print(self.loss_summary, type(self.loss_summary))
            # print(self.lr_summary, type(self.lr_summary))
            self.summary = tf.compat.v1.summary.merge([self.loss_summary, self.lr_summary])
            self.writer = tf.compat.v1.summary.FileWriter(self.summary_dir, self.sess.graph)
        
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.load()

    def load(self):
        """Load weights from checkpoint"""
        if os.path.isfile(self.model_path + '.meta'):
            variables_to_restore = layers_model.get_restore_var_list(self.model_path)
            restorer = tf.compat.v1.train.Saver(variables_to_restore)
            restorer.restore(self.sess, self.model_path)
            print('Loading checkpoint {}'.format(self.model_path))
        else:
            print('Loading failed.')

    def save(self, step):
        ''' 
            Save checkpoints
        '''
        save_path = self.saver.save(self.sess, self.model_path, global_step=step)
        print('Model {} saved in file.'.format(save_path))
        
    def set_optimizer(self):
        # Set learning rate decay
        self.LR = tf.compat.v1.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_step,
            decay_rate=self.decay_rate,
            staircase=True
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.LR)
        
        self.train_op = optimizer.minimize(
            loss = self.net.total_loss,
            global_step = self.global_step,
            var_list = None
        )
    
    def train_cnn(self, images, labels):
        #Input
        feed_dict = {
            self.net.input: images,
            self.net.gt: labels
        }
        #Output
        fetch_list = [
            self.train_op,
            self.summary, 
            self.net.prob, 
            self.net.net_loss,
        ]

        return self.sess.run(fetch_list, feed_dict=feed_dict)
    
    def train(self, *args):
        return self.train_cnn(images=args[0], labels=args[1])
    
    def test_cnn(self, images):
        for i, im in enumerate(images):
            images[i] = cv2.resize(im, (self.img_size[0], self.img_size[1]))
        
        feed_dict = {
            self.net.input: images,
        }
        fetch_list = [
            self.net.prob,
        ]

        return self.sess.run(fetch_list, feed_dict=feed_dict)

    def test(self, *args):
        return self.test_cnn(images=args[0])
