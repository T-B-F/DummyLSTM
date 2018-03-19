#!/usr/bin/env python
from __future__ import division, print_function

import tensorflow as tf


class Model:
    
    def __init__(self, batch_size, sequence_length, hidden_size, 
                 number_of_characters, learning_rate, dropout, 
                 num_layers = 1,
                 is_training=False):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.number_of_characters = number_of_characters
        
        # placeholder for X and Y
        self._inputs = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name ="input")   
        self._targets = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="target") 
        one_hot_inputs = tf.one_hot(self._inputs, depth=self.number_of_characters)  


        # Bi-LSTM
        cell_fw = tf.contrib.rnn.DropoutWrapper(
                            tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True),
                            output_keep_prob=self.dropout)
        cell_bw = tf.contrib.rnn.DropoutWrapper(
                            tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True),
                            output_keep_prob=self.dropout)

        cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw] * num_layers, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw] * num_layers, state_is_tuple=True)

        self._initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32) 
        self._initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)


        lstm_output, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                one_hot_inputs,
                                                                initial_state_fw=self.initial_state_fw,
                                                                initial_state_bw=self.initial_state_bw)

        lstm_output_fw, lstm_output_bw = lstm_output 
        self._final_state_fw, self._final_state_bw = final_state 

        # concatenate fw and bw layer
        lstm_output = tf.concat(lstm_output, axis=2)
        # apply dense to reshape
        lstm_dense = tf.layers.dense(inputs=lstm_output, units=self.hidden_size, activation=tf.nn.tanh) 
        # concatenate with input
        lstm_output = tf.concat((lstm_dense, one_hot_inputs), axis=2)
        # apply dense to reshape
        lstm_output = tf.layers.dense(lstm_output, units=self.number_of_characters, activation=tf.nn.softmax)

        # compute logits and probabilities
        self._logits_flat = tf.reshape(lstm_output, (-1, self.number_of_characters)) 
        probabilities_flat = tf.nn.softmax(self.logits_flat)
        self._probabilities = tf.reshape(probabilities_flat, (self.batch_size, -1, self.number_of_characters)) 
        
        targets_flat = tf.reshape(self.targets, (-1, ))
        #correct_pred = tf.equal(tf.argmax(probabilities_flat, 1), tf.cast(tf.round(targets_flat), tf.int64))

        # compute accuracy
        #self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self._accuracy, self._accuracy_op = tf.metrics.accuracy(labels=tf.cast(tf.round(targets_flat), tf.int64),
                                                                predictions=tf.argmax(probabilities_flat, 1))

        if not is_training:
            return
        
        # compute loss
        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_flat, labels=targets_flat)
        self._cost = tf.reduce_mean(self.loss)

        # optimizer
        trainable_variables = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self._train_op = optimizer.apply_gradients(zip(gradients, trainable_variables))


    @property
    def inputs(self):
        return self._inputs
    
    @property
    def targets(self):
        return self._targets
    
    @property
    def initial_state_fw(self):
        return self._initial_state_fw
    
    @property
    def initial_state_bw(self):
        return self._initial_state_bw
    
    @property
    def final_state_fw(self):
        return self._final_state_fw
    
    @property
    def final_state_bw(self):
        return self._final_state_bw

    @property
    def logits_flat(self):
        return self._logits_flat
    
    @property
    def probabilities(self):
        return self._probabilities

    @property
    def accuracy(self):
        return self._accuracy
    
    @property
    def accuracy_op(self):
        return self._accuracy_op

    @property
    def loss(self):
        return self._loss
    
    @property
    def cost(self):
        return self._cost
    
    @property
    def train_op(self):
        return self._train_op
    