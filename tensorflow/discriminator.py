import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops.nn import rnn
from tensorflow.python.ops.nn import seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

class Discriminator(object):
    def __init__(self, args, is_training=True):
        self.args = args

        if not is_training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            self.cell = rnn_cell.BasicRNNCell(args.rnn_size)
        elif args.model == 'gru':
            self.cell = rnn_cell.GRUCell(args.rnn_size)
        elif args.model == 'lstm':
            self.cell = rnn_cell.BasicLSTMCell(args.rnn_size)
        else:
            raise Exception('model type not supported: {}'.format(args.model))

        self.cell = rnn_cell.MultiRNNCell([self.cell] * args.num_layers)

        self.input_data    = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets       = tf.placeholder(tf.int32, [args.batch_size, args.seq_length]) # Target replication
        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnn'):
            softmax_w = tf.get_variable('softmax_w', [args.rnn_size, 2])
            softmax_b = tf.get_variable('softmax_b', [2])

            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
                inputs    = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs    = [tf.squeeze(i, [1]) for i in inputs]

            outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, 
                self.cell, loop_function=None)

        output_tf   = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.nn.xw_plus_b(output_tf, softmax_w, softmax_b)
        self.probs  = tf.nn.softmax(self.logits)
        
        loss = seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([args.batch_size * args.seq_length])])

        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.final_state = last_state
        self.lr          = tf.Variable(0.0, trainable = False)
        tvars            = tf.trainable_variables()
        grads, _         = tf.clip_by_global_norm(tf.gradients(self.cost, tvars, aggregation_method=2), args.grad_clip)
        optimizer        = tf.train.AdamOptimizer(self.lr)
        self.train_op    = optimizer.apply_gradients(zip(grads, tvars))

    def predict(self, sess, chars, vocab):
        '''Predict a seqeuence of chars using the current model'''
        state = self.cell.zero_state(self.args.batch_size, tf.float32).eval()
        probabilities, num_seq = [], []

        for char in chars:
            num_seq.append(vocab[char])

        for num in num_seq:
            x = np.array([[num]])
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            probabilities.append(probs)

        return probabilities
