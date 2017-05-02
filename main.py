import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np


class BrainNet:
    def __init__(self, input_shape=[None,  71, 125], num_output=32, num_classes=6, restore_dir=None, data='./eeg_data/'
                                                                                                          '/'):
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

        train_percent  = 0.03
        val_percent = 0.1
        test_percent = 0.2

        artf = np.load(data + 'artf_total.npz')
        bckg = np.load(data + 'bckg_total.npz')
        eybl = np.load(data + 'eybl_total.npz')
        gped = np.load(data + 'gped_total.npz')
        spsw = np.load(data + 'spsw_total.npz')
        pled = np.load(data + 'pled_total.npz')

        self.artf_train = artf['arr_0'][:int(len(artf['arr_0'])*train_percent)-1]
        self.bckg_train = bckg['arr_0'][:int(len(bckg['arr_0'])*train_percent)-1]
        self.eybl_train = eybl['arr_0'][:int(len(bckg['arr_0'])*train_percent)-1]
        self.gped_train = gped['arr_0'][:int(len(gped['arr_0'])*train_percent)-1]
        self.spsw_train = spsw['arr_0'][:int(len(spsw['arr_0'])*train_percent)-1]
        self.pled_train = pled['arr_0'][:int(len(pled['arr_0'])*train_percent)-1]

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

    def get_triplets(self):

        choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf']


        choice = random.choice(choices)

        if choice == 'bckg':
            neg_choices = ['eybl', 'gped', 'spsw', 'pled', 'artf']
            ii = random.randint(0, len(self.bckg_train) - 1)
            a = self.bckg_train[ii]

            jj = random.randint(0, len(self.bckg_train) - 1)
            p = self.bckg_train[jj]

        elif choice == 'eybl':
            neg_choices = ['bckg', 'gped', 'spsw', 'pled', 'artf']
            ii = random.randint(0, len(self.eybl_train) - 1)
            a = self.eybl_train[ii]

            jj = random.randint(0, len(self.eybl_train) - 1)
            p = self.eybl_train[jj]

        elif choice == 'gped':
            neg_choices = ['bckg', 'eybl', 'spsw', 'pled', 'artf']
            ii = random.randint(0, len(self.gped_train) - 1)
            a = self.gped_train[ii]

            jj = random.randint(0, len(self.gped_train) - 1)
            p = self.gped_train[jj]

        elif choice == 'spsw':
            neg_choices = ['bckg', 'eybl', 'gped', 'pled', 'artf']
            ii = random.randint(0, len(self.spsw_train) - 1)
            a = self.spsw_train[ii]

            jj = random.randint(0, len(self.spsw_train) - 1)
            p = self.spsw_train[jj]

        elif choice == 'pled':
            neg_choices = ['bckg', 'eybl', 'gped', 'spsw', 'artf']
            ii = random.randint(0, len(self.pled_train) - 1)
            a = self.pled_train[ii]

            jj = random.randint(0, len(self.pled_train) - 1)
            p = self.pled_train[jj]

        else:
            neg_choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled']
            ii = random.randint(0, len(self.artf_train) - 1)
            a = self.artf_train[ii]

            jj = random.randint(0, len(self.artf_train) - 1)
            p = self.artf_train[jj]


        neg_choice = random.choice(neg_choices)

        if neg_choice == 'bckg':
            ii = random.randint(0, len(self.bckg_train) - 1)
            n = self.bckg_train[ii]
        elif neg_choice == 'eybl':
            ii = random.randint(0, len(self.eybl_train) - 1)
            n = self.eybl_train[ii]
        elif neg_choice == 'gped':
            neg_choices = ['bckg', 'eybl', 'spsw', 'pled', 'artf']
            ii = random.randint(0, len(self.gped_train) - 1)
            n = self.gped_train[ii]
        elif neg_choice == 'spsw':
            neg_choices = ['bckg', 'eybl', 'gped', 'pled', 'artf']
            ii = random.randint(0, len(self.spsw_train) - 1)
            n = self.spsw_train[ii]
        elif neg_choice == 'pled':
            neg_choices = ['bckg', 'eybl', 'gped', 'spsw', 'artf']
            ii = random.randint(0, len(self.pled_train) - 1)
            n = self.pled_train[ii]
        else:
            neg_choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled']
            ii = random.randint(0, len(self.artf_train) - 1)
            n = self.artf_train[ii]


        a = np.expand_dims(a, 0)*10e4
        p = np.expand_dims(p, 0)*10e4
        n = np.expand_dims(n, 0)*10e4

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
model.train_model(learning_rate=1e-2, keep_prob=0.5, train_data="./eeg_data/", batch_size=100, train_epoch=10)