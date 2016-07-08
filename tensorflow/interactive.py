import tensorflow as tf
import numpy as np
import logging
from argparse import ArgumentParser
from gan import GAN
import time
import os
import cPickle
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data',
		help='data directory containing reviews')
	parser.add_argument('--save_dir_gen', type=str, default='models_generator',
		help='directory to store checkpointed generator models')
	parser.add_argument('--save_dir_dis', type=str, default='models_discriminator',
		help='directory to store checkpointed discriminator models')
	parser.add_argument('--rnn_size', type=int, default=128,
		help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=2,
		help='number of layers in the RNN')
	parser.add_argument('--model', type=str, default='lstm',
		help='rnn, gru, or lstm')
	parser.add_argument('--batch_size', type=int, default=5,
		help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=20,
		help='RNN sequence length')
	parser.add_argument('-n', type=int, default=500,
		help='number of characters to sample')
	parser.add_argument('--prime', type=str, default=' ',
		help='prime text')
	parser.add_argument('--num_epochs', type=int, default=5,
		help='number of epochs')
	parser.add_argument('--num_epochs_dis', type=int, default=5,
		help='number of epochs to train discriminator')
	parser.add_argument('--save_every', type=int, default=50,
		help='save frequency')
	parser.add_argument('--grad_clip', type=float, default=5.,
		help='clip gradients at this value')
	parser.add_argument('--learning_rate', type=float, default=0.002,
		help='learning rate')
	parser.add_argument('--learning_rate_dis', type=float, default=0.0002,
		help='learning rate for discriminator')
	parser.add_argument('--decay_rate', type=float, default=0.97,
		help='decay rate for rmsprop')
	parser.add_argument('--keep_prob', type=float, default=0.5,
		help='keep probability for dropout')
	parser.add_argument('--vocab_size', type=float, default=10,
		help='size of the vocabulary (characters)')
	return parser.parse_args()

args = parse_args()
is_training = True

cell_gen = rnn_cell.BasicLSTMCell(args.rnn_size)
cell_dis = rnn_cell.BasicLSTMCell(args.rnn_size)

cell_gen = rnn_cell.MultiRNNCell([cell_gen] * args.num_layers)
cell_dis = rnn_cell.MultiRNNCell([cell_dis] * args.num_layers)

# Pass the generated sequences and targets (1)
input_data  = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
targets     = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

# Both generator and discriminator should start with 0-states
initial_state_gen = cell_gen.zero_state(args.batch_size, tf.float32)
initial_state_dis = cell_dis.zero_state(args.batch_size, tf.float32)

############
# Generator
############
with tf.variable_scope('rnn_generator'):
	softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
	softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
	
	with tf.device('/cpu:0'):
		embedding  = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
		inputs_gen = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, input_data))
		inputs_gen = [tf.squeeze(i, [1]) for i in inputs_gen]

def loop(prev, _):
	prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
	prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
	return tf.nn.embedding_lookup(embedding, prev_symbol)

outputs_gen, last_state = seq2seq.rnn_decoder(inputs_gen, initial_state_gen, 
	cell_gen, loop_function=None if is_training else loop, scope='rnn_generator')

#  Dim: [args.batch_size * args.seq_length, args.rnn_size]
output_gen = tf.reshape(tf.concat(1, outputs_gen), [-1, args.rnn_size])

#  Dim: [args.batch_size * args.seq_length, args.vocab_size]
logits_gen = tf.nn.xw_plus_b(output_gen, softmax_w, softmax_b)
gen_probs  = tf.nn.softmax(logits_gen)
gen_probs  = tf.reshape(gen_probs, [args.batch_size, args.seq_length, args.vocab_size])

################
# Discriminator
################
# Pass a tensor of *probabilities* over the characters to the Discriminator
with tf.variable_scope('rnn_discriminator'):
	softmax_w = tf.get_variable('softmax_w', [args.rnn_size, 2], trainable = False)
	softmax_b = tf.get_variable('softmax_b', [2], trainable = False)

	with tf.device('/cpu:0'):
		embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size], trainable = False)
		
		# TODO:
		# Create appropriate inputs, the probability sequences from Generator
		inputs_dis    = tf.split(1, args.seq_length, gen_probs)
		inputs_dis    = [tf.matmul(tf.squeeze(i, [1]), embedding) for i in inputs_dis]

state   = initial_state_dis
outputs = []

for i, inp in enumerate(inputs_dis):
	if i > 0:
		tf.get_variable_scope().reuse_variables()
	output, state = cell_dis(inp, state)
	outputs.append(output)
last_state = state

output_tf   = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
logits = tf.nn.xw_plus_b(output_tf, softmax_w, softmax_b)
probs  = tf.nn.softmax(logits)

loss = seq2seq.sequence_loss_by_example(
	[logits],
	[tf.reshape(targets, [-1])], 
	[tf.ones([args.batch_size * args.seq_length])],
	2)

cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

final_state = last_state
lr          = tf.Variable(0.0, trainable = False)
tvars 	         = tf.trainable_variables()
grads, _         = tf.clip_by_global_norm(tf.gradients(cost, tvars, aggregation_method = 2), args.grad_clip)
optimizer        = tf.train.AdamOptimizer(lr)
train_op    = optimizer.apply_gradients(zip(grads, tvars))
