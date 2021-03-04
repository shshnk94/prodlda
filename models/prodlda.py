import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
#import matplotlib.pyplot as plt
import pickle

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return np.random.uniform(size=(fan_in, fan_out), low=low, high=high)

def log_dir_init(fan_in, fan_out,topics=50):
    return tf.log((1.0/topics)*tf.ones([fan_in, fan_out]))

#tf.reset_default_graph()

class VAE(object):

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=100, keep_prob=0.0):

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        #self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob = keep_prob

        self.h_dim = network_architecture["n_z"]
        self.a = 1*np.ones((1 , self.h_dim)).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),1)).T)
        self.var2 = tf.constant(  ( ( (1.0/self.a)*( 1 - (2.0/self.h_dim) ) ).T +
                                ( 1.0/(self.h_dim*self.h_dim) )*np.sum(1.0/self.a,1) ).T  )

        self._create_network()
        self._create_loss_optimizer()

        init = tf.initialize_all_variables()
       
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

        self.sess.run(init)

    def _create_network(self):

        self.z_mean,self.z_log_sigma_sq = self._recognition_network()

        n_z = self.network_architecture["n_z"]
        self.eps = tf.placeholder("float", [None, n_z])
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.eps))
        self.sigma = tf.exp(self.z_log_sigma_sq)

        self.x_reconstr_mean = self._generator_network(self.z)

    def _recognition_network(self):

        # Generate probabilistic encoder (recognition network)
        layer_1 = tf.layers.dense(self.x, self.network_architecture['n_hidden_recog_1'], activation=self.transfer_fct)
        layer_2 = tf.layers.dense(layer_1, self.network_architecture['n_hidden_recog_2'], activation=self.transfer_fct)
        layer_do = tf.nn.dropout(layer_2, self.keep_prob)

        z_mean = tf.contrib.layers.batch_norm(tf.layers.dense(layer_do, self.network_architecture['n_z']))
        z_log_sigma_sq = tf.contrib.layers.batch_norm(tf.layers.dense(layer_do, self.network_architecture['n_z']))

        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, z):

        self.layer_do_0 = tf.nn.dropout(tf.nn.softmax(z), self.keep_prob)

        weights = tf.constant_initializer(xavier_init(self.network_architecture['n_z'], self.network_architecture['n_hidden_gener_1']))
        x_reconstr_mean = tf.nn.softmax(tf.contrib.layers.batch_norm(tf.layers.dense(self.layer_do_0, 
                                                                                     self.network_architecture['n_hidden_gener_1'],
                                                                                     kernel_initializer=weights,
                                                                                     name='beta')))

        return x_reconstr_mean

    def _create_loss_optimizer(self):

        self.x_reconstr_mean+=1e-10

        reconstr_loss = -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean),1)#/tf.reduce_sum(self.x,1)

        latent_loss = 0.5*( tf.reduce_sum(tf.div(self.sigma,self.var2),1)+\
        tf.reduce_sum( tf.multiply(tf.div((self.mu2 - self.z_mean),self.var2),
                  (self.mu2 - self.z_mean)),1) - self.h_dim + tf.reduce_sum(tf.log(self.var2),1)  - tf.reduce_sum(self.z_log_sigma_sq  ,1) )

        self.cost = tf.reduce_mean(reconstr_loss) + tf.reduce_mean(latent_loss) # average over batch
        optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.99)#.minimize(self.cost)

        grads = tf.gradients(self.cost, tf.trainable_variables())
        self.optimizer = optim.apply_gradients(zip(grads, tf.trainable_variables()))

    def partial_fit(self, X):
        
        eps = np.random.randn(X.shape[0], self.network_architecture["n_z"])
        #opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X, self.eps: eps, self.keep_prob: .4})
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X, self.eps: eps})
        return cost 

    def test(self, X):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        X = np.expand_dims(X, axis=0)
        eps = np.random.randn(X.shape[0], self.network_architecture["n_z"])
        #cost = self.sess.run((self.cost),feed_dict={self.x: X, self.eps: eps,self.keep_prob: 1.0})
        cost = self.sess.run((self.cost),feed_dict={self.x: X, self.eps: eps})
        return cost

    def topic_prop(self, X):
        """heta_ is the topic proportion vector. Apply softmax transformation to it before use.
        """
        eps = np.random.randn(X.shape[0], self.network_architecture["n_z"])
        #theta_ = self.sess.run((self.z),feed_dict={self.x: X, self.eps: eps, self.keep_prob: 1.0})
        theta_ = self.sess.run((self.z),feed_dict={self.x: X, self.eps: eps})
        return theta_
