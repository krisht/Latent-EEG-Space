import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np



class BrainNet:
    def __init__(self, input_shape=[None, 71, 125], num_output=128, num_classes=6, restore_dir=None):
        self.sess = tf.Session()
        if restore_dir is not None:
            dir = tf.train.Saver()
            dir.restore(self.sess, restore_dir)
        else:
            self.build_model(input_shape, num_output)
        self.num_classes = num_classes

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
            net = slim.layers.conv2d(net, num_outputs=32, kernel_size=5, scope='conv1')
            print(net)
            net = slim.layers.max_pool2d(net, kernel_size=3)
            print(net)
            net = slim.layers.batch_norm(net)
            print(net)


            net = slim.layers.conv2d(net, num_outputs = 64, kernel_size=5, scope='conv2')
            print(net)
            net = slim.layers.max_pool2d(net, kernel_size=3)
            print(net)
            net = slim.layers.batch_norm(net)
            print(net)
            net = slim.layers.flatten(net, scope='flatten')
            print(net)
            net = slim.layers.fully_connected(net, 256)
            print(net)
            net = slim.layers.fully_connected(net, 1024)
            print(net)
            net = slim.layers.dropout(net, keep_prob = self.keep_prob)
            print(net)
            net = slim.layers.fully_connected(net, num_output, activation_fn=None, weights_regularizer=None)
            print(net)
            self.net = net
            self.sess.run(tf.global_variables_initializer())


    def train_model(self, learning_rate, keep_prob, train_data, batch_size, train_epoch, outdir=None):



        

        artf0 = np.load('artf0.npz')
        print(artf0['arr_0'].shape)
        bckg0 = np.load('bckg0.npz')
        print(bckg0['arr_0'].shape)
        eybl0 = np.load('eybl0.npz')
        print(eybl0['arr_0'].shape)
        gped0 = np.load('gped0.npz')
        print(gped0['arr_0'].shape)
        spsw0 = np.load('spsw0.npz')
        print(spsw0['arr_0'].shape)
        pled0 = np.load('pled0.npz')
        print(pled0['arr_0'].shape)

        for ii in range(0, batch_size):
            choices = ['bckg', 'eybl']

            ii = random.randint(0, len(artf0['arr_0']))
            jj =  random.randint(0, len(artf0['arr_0']))
            choice = random.choice(choices)

            anchor = artf0['arr_0'][ii]
            anchor = np.expand_dims(anchor, axis=0)
            positive = artf0['arr_0'][jj]
            positive = np.expand_dims(positive, axis=0)
            negative = [] 

            if choice=='bckg' and len(bckg0['arr_0']) != 0:
                kk = random.randint(0, len(bckg0['arr_0']))
                negative = bckg0['arr_0'][kk]
            elif choice == 'eybl' and len(eybl0['arr_0']) != 0:
                kk = random.randint(0, len(eybl0['arr_0']))
                negative = eybl0['arr_0'][kk]
            # elif choice =='gped' and len(gped0['arr_0']) != 0:
            #     kk = random.randint(0, len(gped0['arr_0']))
            #     negative = gped0['arr_0'][kk]
            # elif choice=='spsw' and len(spsw0['arr_0']) != 0:
            #     kk = random.randint(0, len(spsw0['arr_0']))
            #     negative = spsw0['arr_0'][kk]
            # elif choice=='pled' and len(pled0['arr_0']) != 0:
            #     kk = random.randint(0, len(pled0['arr_0']))
            #     negative = pled0['arr_0'][kk]

            negative = np.expand_dims(negative, axis=0)

            print(negative.shape)


            positiveOut = self.sess.run(self.net, feed_dict={self.inputs: positive})
            negativeOut = self.sess.run(self.net, feed_dict = {self.inputs: negative})
            anchorOut = self.sess.run(self.net, feed_dict={self.inputs: anchor})

            self.positiveLoss = tf.reduce_mean(0.5 * tf.square(positiveOut - anchorOut))
            self.negativeLoss = - tf.reduce_mean(0.5 * tf.square(negativeOut - anchorOut))

            self.positiveOptim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.positiveLoss)
            self.negativeOptim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.negativeLoss)

            positiveloss, negativeloss, _, _ = self.sess.run([self.positiveLoss, self.negativeLoss, self.positiveOptim, self.negativeOptim])
            print(ii, positiveloss, negativeloss)
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