#!/usr/bin/env python
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import os, sys, argparse     
from dummy_generator import dummy_sequence, sample_generator
from dummy_model import Model

        
########################
# data        

data_seqs = dummy_sequence(100000)
data_length = len(data_seqs) 

char_set = set()
for ch in data_seqs:
    char_set.add(ch)

char_list = sorted(list(char_set))

char2idx = dict(zip(char_list, range(len(char_list))))
idx2char = dict(zip(range(len(char_list)), char_list))

########################
# parameters        
sequence_length = 20
batch_size = 200
number_of_characters = len(char_set)
hidden_size = 32
dropout = 0.8
learning_rate = 2e-3 

# saver
outputdir = "./tmp"
if not os.path.isdir(outputdir):
    os.makedirs(outputdir)

########################
# Train 

model = Model(batch_size, sequence_length, hidden_size, number_of_characters, learning_rate, dropout, is_training=True)

save_path = os.path.join(outputdir, 'model') 

# save weight
#save_final_fw_c = tf.get_variable('state_fw_c', shape=[batch_size, hidden_size])
#save_final_fw_h = tf.get_variable('state_fw_h', shape=[batch_size, hidden_size])
#save_final_bw_c = tf.get_variable('state_bw_c', shape=[batch_size, hidden_size])
#save_final_bw_h = tf.get_variable('state_bw_h', shape=[batch_size, hidden_size])

saver = tf.train.Saver(tf.trainable_variables()) 

initglob_op = tf.global_variables_initializer()
initloc_op = tf.local_variables_initializer()

print("Train")
with tf.Session() as sess:  
    sess.run(initglob_op)
    state_fw = sess.run(model.initial_state_fw)
    state_bw = sess.run(model.initial_state_bw)
    for epoch in range(1):
        # initialize local variable 
        sess.run(initloc_op)
        
        all_acc = list()
        all_loss = list()
        for input_batch, target_batch in sample_generator(data_seqs, char2idx, batch_size, sequence_length):
            feed_dict = {model.inputs: input_batch,
                        model.targets: target_batch}

            #for i, (c, h) in enumerate(model.initial_state_fw):   
                #feed_dict[c] = state_fw[i].c
                #feed_dict[h] = state_fw[i].h
            #for i, (c, h) in enumerate(model.initial_state_bw):   
                #feed_dict[c] = state_bw[i].c
                #feed_dict[h] = state_bw[i].h

            fetches = {"cost": model.cost,
                       "accuracy": model.accuracy,
                       "acc_op": model.accuracy_op,
                       "train_op":model.train_op,
                       #"state_fw": model.final_state_fw,
                       #"state_bw": model.final_state_bw
                       }

            outputs = sess.run(fetches, feed_dict=feed_dict)

            #state_fw = outputs["state_fw"]
            #state_bw = outputs["state_bw"]

            all_loss.append(outputs["cost"])
            #all_acc.append(computed_accuracy)
            #print(sum(all_loss), sum(all_acc))
        print('i: {}, loss: {}, accuracy: {}'.format(epoch, 
                                                     sum(all_loss)/len(all_loss),
                                                     outputs["accuracy"]))
    # only one layer
    #assign_fw_c = tf.assign(save_final_fw_c, state_fw[0][0])
    #assign_fw_h = tf.assign(save_final_fw_h, state_fw[0][1]) 
    #assign_bw_c = tf.assign(save_final_bw_c, state_bw[0][0])
    #assign_bw_h = tf.assign(save_final_bw_h, state_bw[0][1])

    #sess.run(assign_fw_c) 
    #sess.run(assign_fw_h)
    #sess.run(assign_bw_c) 
    #sess.run(assign_bw_h) 
    
    saver.save(sess, save_path)
    
    
