#!/usr/bin/env python
from __future__ import division, print_function

import numpy as np

def dummy_sequence(dummy_length):
    """ generate a round of dummy sequences
    """
    seq = ""
    p_abc_ori = np.asarray([0.7, 0.2, 0.1])
    p_abc_trans = np.asarray(
                      [[0.6, 0.3, 0.1],
                       [0.3, 0.5, 0.2],
                       [0.8, 0.1, 0.1]])
    chars = ["a", "b", "c"]
    positions = {"a": 0, "b": 1, "c": 2}
    c = np.random.choice(chars, p=p_abc_ori)
    seq += c
    for i in range(dummy_length):
        c = np.random.choice(chars, p = p_abc_trans[positions[c]])
        seq += c
    return seq


def sample_generator(data_seqs, char_dict, batch_size, sequence_length):
    data_length = len(data_seqs) 
    length = sequence_length + 1
    num_steps = (data_length // batch_size)

    for step in range(num_steps):
        start_idxs = np.random.random_integers(0, data_length, batch_size)
        input_batch = np.zeros((batch_size, sequence_length), dtype=np.int32)
        target_batch = np.zeros((batch_size, sequence_length), dtype=np.int32)
        for i, start_idx in enumerate(start_idxs):
            sample = [char_dict[data_seqs[i % data_length]] for i in range(start_idx, start_idx+length)]
            input_batch[i, :] = sample[0:sequence_length] 
            target_batch[i, :] = sample[1:sequence_length+1]
            start_idxs = (start_idxs + sequence_length) % data_length 
        yield input_batch, target_batch 
        

def sample_generator2(data_seqs, char_dict, batch_size, sequence_length):
    data_length = len(data_seqs) 
    length = sequence_length + 1
    num_steps = ((data_length // length)  // batch_size)
    print(num_steps)
    
    idx = np.random.random_integers(0, data_length)
    for step in range(num_steps):
        input_batch = np.zeros((batch_size, sequence_length), dtype=np.int32)
        target_batch = np.zeros((batch_size, sequence_length), dtype=np.int32)
        for i in range(batch_size):
            sample = [char_dict[data_seqs[c % data_length]] for c in range(idx, idx+length)]
            input_batch[i, :] = sample[0:sequence_length] 
            target_batch[i, :] = sample[1:sequence_length+1]
            idx += (idx + sequence_length) % data_length 
        yield input_batch, target_batch 
