import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np



class BrainNet:
    def __init__(self, input_shape=[None,  71, 125], num_output=128, num_classes=6, restore_dir=None):
        self.sess = tf.Session()
        self.num_classes = num_classes
        self.num_output = num_output
        self.input_shape = input_shape
        if restore_dir is not None:
            dir = tf.train.Saver()
            dir.restore(self.sess, restore_dir)
        else:
            #only for inference time usage....
            #i.e you give it a sample and it determines the embedding. This is for use in a k-cluster sort algorithm
            #Not for training
            self.inference_input = tf.placeholder(tf.float32, shape=input_shape)
            self.inference_model = self.get_model(self.inference_input, reuse=False)

    def triplet_loss(self, alpha):
        """Calculate the triplet loss according to the FaceNet paper

        Args:
          anchor: the embeddings for the anchor images.
          positive: the embeddings for the positive images.
          negative: the embeddings for the negative images.

        Returns:
          the triplet loss according to the FaceNet paper as a float tensor.
        """
        self.anchor = tf.placeholder(tf.float32, shape=self.input_shape)
        self.positive = tf.placeholder(tf.float32, shape=self.input_shape)
        self.negative = tf.placeholder(tf.float32, shape=self.input_shape)
        anchor = self.get_model(self.anchor, reuse=True)
        positive = self.get_model(self.positive, reuse=True)
        negative = self.get_model(self.negative, reuse=True)
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


    def get_model(self, input, reuse=False):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                           weights_initializer=tf.contrib.layers.xavier_initializer(seed=random.random(), uniform=True),
                           weights_regularizer=slim.l2_regularizer(1e-3), reuse=reuse):
            net = tf.expand_dims(input, axis=3)
            net = slim.layers.conv2d(net, num_outputs=32, kernel_size=4, scope='conv1', trainable=True)
            net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool1')
            net = slim.layers.batch_norm(net, trainable=True)
            net = slim.layers.conv2d(net, num_outputs=64, kernel_size=5, scope='conv2', trainable=True)
            net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool2')
            net = slim.layers.batch_norm(net, trainable=True)
            net = slim.layers.flatten(net, scope='flatten');
            net = slim.layers.fully_connected(net, 256, scope='fc1', trainable=True)
            net = slim.layers.fully_connected(net, 1024, scope='fc2', trainable=True)
            net = slim.layers.fully_connected(net, self.num_output, activation_fn=None, weights_regularizer=None, scope='output')
            print(net)
            return net


    def train_model(self, learning_rate, keep_prob, train_data, batch_size, train_epoch, outdir=None):
        loss = self.triplet_loss(alpha=3)
        optim= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)
        self.sess.run(tf.global_variables_initializer())



        for ii in range(0, batch_size):
            feeder = self.get_triplets()

            anchor = feeder[0]
            positive = feeder[1]
            negative = feeder[2]

            _, loss = self.sess.run([optim, loss], feed_dict={self.anchor:anchor, self.positive:positive, self.negative:negative})

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