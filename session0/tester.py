import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np


class BrainNet:
    def __init__(self, input_shape=[None,  71, 125], num_output=32, num_classes=6, restore_dir=None):
        self.sess = tf.Session()
        self.num_classes = num_classes
        self.num_output = num_output
        self.input_shape = input_shape
        if restore_dir is not None:
            dir = tf.train.Saver()
            dir.restore(self.sess, restore_dir)
        else:
            # only for inference time usage....
            # i.e you give it a sample and it determines the embedding. This is for use in a k-cluster sort algorithm
            # Not for training
            self.inference_input = tf.placeholder(tf.float32, shape=input_shape)
            self.inference_model = self.get_model(self.inference_input, reuse=False)

        self.artf0 = np.load('artf0.npz')
        self.bckg0 = np.load('bckg0.npz')
        self.eybl0 = np.load('eybl0.npz')
        self.gped0 = np.load('gped0.npz')
        self.spsw0 = np.load('spsw0.npz')
        self.pled0 = np.load('pled0.npz')

    def triplet_loss(self, alpha):
        self.anchor = tf.placeholder(tf.float32, shape=self.input_shape)
        self.positive = tf.placeholder(tf.float32, shape=self.input_shape)
        self.negative = tf.placeholder(tf.float32, shape=self.input_shape)
        self.anchor_out = self.get_model(self.anchor, reuse=True)
        self.positive_out = self.get_model(self.positive, reuse=True)
        self.negative_out = self.get_model(self.negative, reuse=True)
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(self.anchor_out - self.positive_out))
            neg_dist = tf.reduce_sum(tf.square(self.anchor_out - self.negative_out))

            basic_loss =  tf.maximum(0., alpha + pos_dist - neg_dist)
            loss = tf.reduce_mean(basic_loss)
            return loss


    def get_triplets(self, ax=None, px=None, nx=None, choice=None):

        choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled']

        ii = random.randint(0, len(self.artf0['arr_0'])-1)
        jj = random.randint(0, len(self.artf0['arr_0'])-1)

        a = self.artf0['arr_0'][ii]*10e3
        a = np.expand_dims(a, 0)
        p = self.artf0['arr_0'][jj]*10e3
        p = np.expand_dims(p, 0)

        n = []

        while n==[]:
            choice = random.choice(choices)
            if choice == 'bckg' and len(self.bckg0['arr_0']) != 0:
                kk = random.randint(0, len(self.bckg0['arr_0'])-1)
                n = self.bckg0['arr_0'][kk]
            elif choice == 'eybl' and len(self.eybl0['arr_0']) != 0:
                kk = random.randint(0, len(self.eybl0['arr_0'])-1)
                n = self.eybl0['arr_0'][kk]
            elif choice=='spsw' and len(self.spsw0['arr_0']) !=0:
                kk = random.randint(0, len(self.spsw0['arr_0']) -1)
                n = self.spsw['arr_0'][kk]
            elif choice =='pled' and len(self.pled0['arr_0']) !=0:
                kk = random.randint(0, len(self.pled0['arr_0']) -1)
                n = self.pled0['arr_0'][kk]
            elif choice=='gped' and len(self.gped0['arr_0']) !=0:
                kk = random.randint(0, len(self.gped0['arr_0']) -1)
                n = self.gped0['arr_0'][kk]

        n = np.expand_dims(n, 0)*10e3

        return np.vstack([a,p,n])


    def get_model(self, input, reuse=False):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                           weights_initializer=tf.contrib.layers.xavier_initializer(seed=random.random(), uniform=True),
                           weights_regularizer=slim.l2_regularizer(0.05), reuse=reuse):
            net = tf.expand_dims(input, axis=3)
            net = slim.layers.conv2d(net, num_outputs=32, kernel_size=4, scope='conv1', trainable=True)
            net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool1')
            net = slim.layers.conv2d(net, num_outputs=64, kernel_size=5, scope='conv2', trainable=True)
            net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool2')
            net = slim.layers.flatten(net, scope='flatten')
            net = slim.layers.fully_connected(net, 256, scope='fc1', trainable=True)
            net = slim.layers.fully_connected(net, 1024, scope='fc2', trainable=True)
            net = slim.layers.fully_connected(net, self.num_output, activation_fn=None, weights_regularizer=None, scope='output')
            return net


    def train_model(self, learning_rate, keep_prob, train_data, batch_size, train_epoch, outdir=None):
        loss = self.triplet_loss(alpha=0.5)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optim = self.optimizer.minimize(loss=loss)
        self.sess.run(tf.global_variables_initializer())

        count = 0
        ii=0

        while ii <= batch_size:
            ii+=1
            feeder = self.get_triplets()

            anchor = feeder[0]
            anchor = np.expand_dims(anchor, 0)
            positive = feeder[1]
            positive = np.expand_dims(positive, 0)
            negative = feeder[2]
            negative = np.expand_dims(negative, 0)

            temploss = self.sess.run(loss, feed_dict={self.anchor:anchor, self.positive: positive, self.negative:negative})

            if temploss == 0:
                print(temploss)
                ii-=1
                count+=1
                continue

            _, anchor, positive, negative = self.sess.run([self.optim,
                                                           self.anchor_out,
                                                           self.positive_out,
                                                           self.negative_out], feed_dict={self.anchor:anchor,
                                                                                          self.positive: positive,
                                                                                          self.negative:negative})

            d1 = np.linalg.norm(positive - anchor)
            d2 = np.linalg.norm(negative - anchor)

            print("Iteration:", ii, ", Loss: ", temploss, ", Positive Diff: ", d1, ", Negative diff: ", d2)
            print("Iterations skipped: ", count)



model = BrainNet()
model.train_model(learning_rate=1e-2, keep_prob=0.5, train_data="", batch_size=100, train_epoch=10)