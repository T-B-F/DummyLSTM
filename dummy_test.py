#!/usr/bin/env python
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import os, sys, argparse
from dummy_generator import dummy_sequence, sample_generator
from dummy_model import Model

data_test = dummy_sequence(10000)
data_length = len(data_test ) 

char_set = set()
for ch in data_test:
    char_set.add(ch)

char_list = sorted(list(char_set))

char2idx = dict(zip(char_list, range(len(char_list))))
idx2char = dict(zip(range(len(char_list)), char_list))
         
        
# parameters        
sequence_length = 150
batch_size = 200
number_of_characters = len(char_set)
hidden_size = 32
dropout = 0.8
learning_rate = 2e-3 

# saver
outputdir = "./tmp"
if not os.path.isdir(outputdir):
    os.makesdir(outputdir)

    
########################
# Predict on trained variable 

latest_checkpoint = tf.train.latest_checkpoint(outputdir)

model = Model(1, None, hidden_size, number_of_characters, learning_rate, dropout)

saver = tf.train.Saver(tf.trainable_variables())

initglob_op = tf.global_variables_initializer()
initloc_op = tf.local_variables_initializer()

with tf.Session() as sess: 
    sess.run(initglob_op)
    sess.run(initloc_op)
    saver.restore(sess, latest_checkpoint) 


    for idx in range(0, len(data_test)-(sequence_length+1), sequence_length): 
        idx_target = [char2idx[c] for c in data_test[idx+1:idx+1+sequence_length]]
        idx_query = [char2idx[c] for c in data_test[idx:idx+sequence_length]]
        feed_dict={model.inputs: np.asarray([idx_query]),
                   model.targets: np.asarray([idx_target])
                   }
        fetches = {"probabilities": model.probabilities,
                   "accuracy": model.accuracy,
                   "acc_op": model.accuracy_op}
        outputs = sess.run(fetches, feed_dict=feed_dict)

    print("Global accuracy: {}".format(outputs["accuracy"]))
    
