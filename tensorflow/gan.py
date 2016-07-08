import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops.nn import rnn
from tensorflow.python.ops.nn import seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.distributions import Categorical

def variable_summaries(var, name):
	'''Attach a lot of summaries to a Tensor.'''
	mean = tf.reduce_mean(var)
	tf.scalar_summary('mean/' + name, mean)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
	tf.scalar_summary('sttdev/' + name, stddev)
	tf.scalar_summary('max/' + name, tf.reduce_max(var))
	tf.scalar_summary('min/' + name, tf.reduce_min(var))
	tf.histogram_summary(name, var)
	
class GAN(object):
	def __init__(self, args, is_training=True):

		if not is_training:
			seq_length = 1
		else:
			seq_length = args.seq_length

		if args.model == 'rnn':
			cell_gen = rnn_cell.BasicRNNCell(args.rnn_size)
			cell_dis = rnn_cell.BasicRNNCell(args.rnn_size)
		elif args.model == 'gru':
			cell_gen = rnn_cell.GRUCell(args.rnn_size)
			cell_dis = rnn_cell.GRUCell(args.rnn_size)
		elif args.model == 'lstm':
			cell_gen = rnn_cell.BasicLSTMCell(args.rnn_size)
			cell_dis = rnn_cell.BasicLSTMCell(args.rnn_size)
		else:
			raise Exception('model type not supported: {}'.format(args.model))

		# Pass the generated sequences and targets (1)
		with tf.name_scope('input'):
			with tf.name_scope('data'):
				self.input_data  = tf.placeholder(tf.int32, [args.batch_size, seq_length])
			with tf.name_scope('targets'):
				self.targets     = tf.placeholder(tf.int32, [args.batch_size, seq_length])

		############
		# Generator
		############
		with tf.variable_scope('generator'):
			self.cell_gen = rnn_cell.MultiRNNCell([cell_gen] * args.num_layers)
			self.initial_state_gen = self.cell_gen.zero_state(args.batch_size, tf.float32)	

			with tf.variable_scope('rnn'):
				softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
				softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
				
				with tf.device('/cpu:0'):
					embedding  = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
					inputs_gen = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
					inputs_gen = [tf.squeeze(i, [1]) for i in inputs_gen]

			outputs_gen, last_state_gen = seq2seq.rnn_decoder(inputs_gen, self.initial_state_gen, 
				self.cell_gen, loop_function=None)
			
			self.logits_sequence = []
			for output_gen in outputs_gen:
				logits_gen  = tf.nn.xw_plus_b(output_gen, softmax_w, softmax_b)
				self.logits_sequence.append(logits_gen)

			self.final_state_gen = last_state_gen

		################
		# Discriminator
		################
		with tf.variable_scope('discriminator'):
			self.cell_dis = rnn_cell.MultiRNNCell([cell_dis] * args.num_layers)
			self.initial_state_dis = self.cell_dis.zero_state(args.batch_size, tf.float32)

			with tf.variable_scope('rnn'):
				softmax_w = tf.get_variable('softmax_w', [args.rnn_size, 2])
				softmax_b = tf.get_variable('softmax_b', [2])

				inputs_dis = []
				embedding  = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
				for logit in self.logits_sequence:
					inputs_dis.append(tf.matmul(logit, embedding))
					# inputs_dis.append(tf.matmul(tf.nn.softmax(logit), embedding))
					
				outputs_dis, last_state_dis = seq2seq.rnn_decoder(inputs_dis, self.initial_state_dis, 
					self.cell_dis, loop_function=None)

			probs, logits = [], []
			for output_dis in outputs_dis:
				logit = tf.nn.xw_plus_b(output_dis, softmax_w, softmax_b)
				prob = tf.nn.softmax(logit)
				logits.append(logit)
				probs.append(prob)

			with tf.name_scope('summary'):
				probs      = tf.pack(probs)
				probs_real = tf.slice(probs, [0,0,1], [args.seq_length, args.batch_size, 1])
				variable_summaries(probs_real, 'probability of real')

			self.final_state_dis = last_state_dis

		#########
		# Train
		#########
		with tf.name_scope('train'):
			gen_loss = seq2seq.sequence_loss_by_example(
				logits,
				tf.unpack(tf.transpose(self.targets)), 
				tf.unpack(tf.transpose(tf.ones_like(self.targets, dtype=tf.float32))))

			self.gen_cost = tf.reduce_sum(gen_loss) / args.batch_size
			tf.scalar_summary('training loss', self.gen_cost)
			self.lr_gen = tf.Variable(0.0, trainable = False)		
			self.tvars 	= tf.trainable_variables()
			gen_vars    = [v for v in self.tvars if not v.name.startswith("discriminator/")]

			if is_training:
				gen_grads            = tf.gradients(self.gen_cost, gen_vars)
				self.all_grads       = tf.gradients(self.gen_cost, self.tvars)
				gen_grads_clipped, _ = tf.clip_by_global_norm(gen_grads, args.grad_clip)
				gen_optimizer        = tf.train.AdamOptimizer(self.lr_gen)
				# gen_optimizer        = tf.train.GradientDescentOptimizer(self.lr_gen)
				self.gen_train_op    = gen_optimizer.apply_gradients(zip(gen_grads_clipped, gen_vars))				


		with tf.name_scope('summary'):
			with tf.name_scope('weight_summary'):
				for v in self.tvars:
					variable_summaries(v, v.op.name)
			if is_training:
				with tf.name_scope('grad_summary'):
					for var, grad in zip(self.tvars, self.all_grads):
						variable_summaries(grad, 'grad/' + var.op.name)

		self.merged = tf.merge_all_summaries()

		
	def generate_samples(self, sess, args, chars, vocab, seq_length = 200, initial = ' ', datafile = 'data/gan/fake_reviews.txt'):
		''' Generate a batch of reviews'''		
		state = self.cell_gen.zero_state(args.batch_size, tf.float32).eval()

		sequence_matrix = []
		for i in xrange(args.batch_size):
			sequence_matrix.append([])
		char_arr = args.batch_size * [initial]
		for n in xrange(seq_length):
			x = np.zeros((args.batch_size, 1))
			for i, char in enumerate(char_arr):
				x[i,0] = vocab[char]    
			feed = {self.input_data: x, self.initial_state_gen: state} 
			sample_op = Categorical(self.logits_sequence[0])
			[sample_indexes, state] = sess.run([sample_op.sample(n = 1), self.final_state_gen], feed)
			char_arr = [chars[i] for i in sample_indexes[0]]
			for i, char in enumerate(char_arr):
				sequence_matrix[i].append(char)

		with open(datafile, 'a+') as f:
			for line in sequence_matrix:
				print ''.join(line)
				print>>f, ''.join(line) 

		return sequence_matrix