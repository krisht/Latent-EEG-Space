import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np



class BrainNet:
    def __init__(self, input_shape=[3,  71, 125], num_output=128, num_classes=6, restore_dir=None):
        self.sess = tf.Session()
        if restore_dir is not None:
            dir = tf.train.Saver()
            dir.restore(self.sess, restore_dir)
        else:
            self.build_model(input_shape, num_output)
        self.num_classes = num_classes

    def triplet_loss(self, total, alpha):
        """Calculate the triplet loss according to the FaceNet paper

        Args:
          anchor: the embeddings for the anchor images.
          positive: the embeddings for the positive images.
          negative: the embeddings for the negative images.

        Returns:
          the triplet loss according to the FaceNet paper as a float tensor.
        """
        anchor, positive, negative = total
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))

            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))

        return loss

    def get_triplets(self, ax=None, px=None, nx=None, choice=None):

        self.artf0 = np.load('artf0.npz')
        self.bckg0 = np.load('bckg0.npz')
        self.eybl0 = np.load('eybl0.npz')
        self.gped0 = np.load('gped0.npz')
        self.spsw0 = np.load('spsw0.npz')
        self.pled0 = np.load('pled0.npz')

        choices = ['bckg', 'eybl']

        ii = random.randint(0, len(self.artf0['arr_0']))
        jj = random.randint(0, len(self.artf0['arr_0']))
        choice = random.choice(choices)

        a = self.artf0['arr_0'][ii]
        a = np.expand_dims(a, 0)
        p = self.artf0['arr_0'][jj]
        p = np.expand_dims(p, 0)

        if choice == 'bckg' and len(self.bckg0['arr_0']) != 0:
            kk = random.randint(0, len(self.bckg0['arr_0']))
            n = self.bckg0['arr_0'][kk]
            n = np.expand_dims(n, 0)
        elif choice == 'eybl' and len(self.eybl0['arr_0']) != 0:
            kk = random.randint(0, len(self.eybl0['arr_0']))
            n = self.eybl0['arr_0'][kk]
            n = np.expand_dims(n, 0)

        return np.vstack([a,p,n])


    def build_model(self, input_shape, num_output):
        self.inputs = tf.placeholder(tf.float32, shape=input_shape, name='input')
        self.num_output = num_output
        self.input_shape = input_shape
        self.keep_prob = 0.5


        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                           weights_initializer=tf.contrib.layers.xavier_initializer(seed=random.random(), uniform=True),
                           weights_regularizer=slim.l2_regularizer(1e-3)):
            net = tf.expand_dims(self.inputs, axis=3)
            print(net)
            net = slim.layers.conv2d(net, num_outputs=32, kernel_size=4, scope='conv1', trainable=True)
            print(net)
            net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool1')
            print(net)
            net = slim.layers.batch_norm(net, trainable=True)
            print(net)
            net = slim.layers.conv2d(net, num_outputs=64, kernel_size=5, scope='conv2', trainable=True)
            print(net)
            net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool2')
            print(net)
            net = slim.layers.batch_norm(net, trainable=True)
            print(net)
            net = slim.layers.flatten(net, scope='flatten');
            print(net)
            net = slim.layers.fully_connected(net, 256, scope='fc1', trainable=True)
            print(net)
            net = slim.layers.fully_connected(net, 1024, scope='fc2', trainable=True)
            print(net)
            net = slim.layers.fully_connected(net, num_output, activation_fn=None, weights_regularizer=None)
            print(net)
            self.net = net


    def train_model(self, learning_rate, keep_prob, train_data, batch_size, train_epoch, outdir=None):
        self.optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize()
        self.sess.run(tf.global_variables_initializer())



        for ii in range(0, batch_size):
            feeder = self.get_triplets()
            out = self.sess.run(self.net, feed_dict={self.inputs: feeder})

            anchor, positive, negative = out

            loss = self.triplet_loss(anchor, positive, negative, alpha=3)

            _, loss = self.sess.run([optim, loss])

            print(loss)



        #
        #
        #
        #
        #
        #
        #     # Train with artf0
        #
        # for ii in range(0, batch_size):
        #     # Train with bckg0
        # for ii in range(0, batch_size):
        #     # Train with eybl0
        # for ii in range(0, batch_size):
        #     # Train with spsw0
        # for ii in range(0, batch_size):
        #     # Train with pled0


    # def infer_embedding(self, data):
    #     data_tuple = self.reshape_data(data)
    #     batch_input = [x[0] for x in data_tuple]
    #     predictions = self.sess.run()
    #     return predictions



model = BrainNet()
model.train_model(learning_rate=1e-3, keep_prob=0.5, train_data="", batch_size=50, train_epoch=10)