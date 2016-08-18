import tensorflow as tf
import numpy as np
import logging
from tensorflow.models.rnn import *
from argparse import ArgumentParser
from batcher_gan import DiscriminatorBatcher, GANBatcher
from gan import GAN
import time
import os
import cPickle

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--real_input_file', type=str, default='real_reviews.txt',
		help='real reviews')
	parser.add_argument('--fake_input_file', type=str, default='fake_reviews.txt',
		help='fake reviews')
	parser.add_argument('--data_dir', type=str, default='data/gan',
		help='data directory containing reviews')
	parser.add_argument('--log_dir', type=str, default='logs',
		help='log directory for TensorBoard')
	parser.add_argument('--vocab_file', type=str, default='simple_vocab.pkl',
		help='data directory containing reviews')
	parser.add_argument('--save_dir_GAN', type=str, default='models_GAN',
		help='directory to store checkpointed GAN models')
	parser.add_argument('--rnn_size', type=int, default=128,
		help='size of RNN hidden state')
	parser.add_argument('--num_layers', type=int, default=1,
		help='number of layers in the RNN')
	parser.add_argument('--model', type=str, default='lstm',
		help='rnn, gru, or lstm')
	parser.add_argument('--batch_size', type=int, default=10,
		help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=30,
		help='RNN sequence length')
	parser.add_argument('--num_epochs_GAN', type=int, default=100,
		help='number of epochs of GAN')
	parser.add_argument('--num_epochs_gen', type=int, default=1,
		help='number of epochs to train generator')
	parser.add_argument('--num_epochs_dis', type=int, default=1,
		help='number of epochs to train discriminator')
	parser.add_argument('--num_batches_gen', type=int, default=10,
		help='number of batches to train generator for each epoch')
	parser.add_argument('--num_batches_dis', type=int, default=100,
		help='number of batches to train discriminator for each epoch')
	parser.add_argument('--num_example_batches', type=int, default=1000,
		help='number of batches to generate')
	parser.add_argument('--save_every', type=int, default=25,
		help='save frequency')
	parser.add_argument('--grad_clip', type=float, default=5.,
		help='clip gradients at this value')
	parser.add_argument('--learning_rate_gen', type=float, default=0.0001,
		help='learning rate')
	parser.add_argument('--learning_rate_dis', type=float, default=0.005,
		help='learning rate for discriminator')
	parser.add_argument('--decay_rate', type=float, default=0.97,
		help='decay rate for rmsprop')
	parser.add_argument('--keep_prob', type=float, default=0.5,
		help='keep probability for dropout')
	parser.add_argument('--vocab_size', type=float, default=5,
		help='size of the vocabulary (characters)')
	return parser.parse_args()



def train_generator(sess, gan, args, train_writer):
	'''Train Generator via GAN.'''
	logging.debug('Training generator...')
		
	for epoch in xrange(args.num_epochs_gen):
		new_lr = args.learning_rate_gen * (args.decay_rate ** epoch)
		sess.run(tf.assign(gan.lr_gen, new_lr))
		
		for batch in xrange(args.num_batches_gen):
			start = time.time()
			gen_train_loss, gen_summary, state_gen, _ = sess.run([
				gan.cost, 
				gan.merged,
				gan.final_state_gen,
				gan.gen_train_op])

			train_writer.add_summary(gen_summary)
			end   = time.time()

			print '{}/{} (epoch {}), gen_train_loss = {:.3f}, time/batch = {:.3f}' \
				.format(epoch * args.num_batches_gen + batch,
					args.num_epochs_gen * args.num_batches_gen,
					epoch, gen_train_loss, end - start)
			
			
def train_discriminator(sess, gan, args):
	'''Train the discriminator via classical approach'''
	logging.debug('Training discriminator...')
	
	batcher  = DiscriminatorBatcher(args.real_input_file, 
									args.fake_input_file, 
									args.data_dir, args.vocab_file,
									args.batch_size, args.seq_length)

	logging.debug('Vocabulary...')
	with open(os.path.join(args.save_dir_GAN, args.vocab_file), 'w') as f:
		cPickle.dump((batcher.chars, batcher.vocab), f)

	for epoch in xrange(args.num_epochs_dis):
		new_lr = args.learning_rate_dis * (args.decay_rate ** epoch)
		sess.run(tf.assign(gan.lr_dis, new_lr))
		batcher.reset_batch_pointer()
		state = gan.initial_state_dis.eval()

		assert batcher.num_batches > args.num_batches_dis
		for batch in xrange(args.num_batches_dis):
			start = time.time()
			x, y  = batcher.next_batch()
			feed  = {gan.input_data: x, gan.targets: y,
					 gan.initial_state_dis: state}
			train_loss, state, _ = sess.run([gan.cost,
											gan.final_state_dis,
											gan.dis_train_op], 
											feed)
			end   = time.time()
			
			print '{}/{} (epoch {}), dis_train_loss = {:.3f}, time/batch = {:.3f}' \
				.format(epoch * args.num_batches_dis + batch,
					args.num_epochs_dis * args.num_batches_dis,
					epoch, train_loss, end - start)

			
def generate_samples(sess, gan, args):
	'''Generate samples.'''
	with open(os.path.join(args.save_dir_GAN, args.vocab_file)) as f:
		chars, vocab = cPickle.load(f)

	data_file = os.path.join(args.data_dir, args.fake_input_file)
	
	for _ in xrange(args.num_example_batches):
		indices = sess.run(gan.sample_op)
		int_to_char = lambda x: chars[x]
		mapfunc = np.vectorize(int_to_char)
		samples = mapfunc(indices.T)	
		with open(data_file, 'a+') as f:
			for line in samples:
				print line
				print>>f, ''.join(line) 
	print('Generated samples to %s' % data_file)    


def reset_reviews(data_dir, file_name):
	'''Clear the file containing the generated reviews.
	Args:
		data_dir:  Directory to store generated reviews.
		file_name:  Name of file containing generated reviews.
	'''
	open(os.path.join(data_dir, file_name), 'w').close()


def adversarial_training(sess, gan, gan_dis, writer, saver, args, weights_load='random'):
	'''Adversarial Training'''
	
	if weights_load == 'random':
		print('Initializing GAN parameters randomly.')
	elif weights_load == 'gan':
		print('Initializing GAN parameters from %s' % args.save_dir_GAN)		
		ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		raise Exception('Invalid initialization for GAN: %s' % weights_load)

	train_generator(sess, gan, args, writer)
	generate_samples(sess, gan, args)

	for epoch in xrange(args.num_epochs_GAN):
		train_discriminator(sess, gan_dis, args)
		train_generator(sess, gan, args, writer)
		reset_reviews(args.data_dir, args.fake_input_file)
		generate_samples(sess, gan, args)

		if epoch % args.save_every == 0:
			checkpoint_path = os.path.join(args.save_dir_GAN, 'model.ckpt')
			saver.save(sess, checkpoint_path)
			print 'GAN model saved to {}'.format(checkpoint_path)



if __name__=='__main__':
	args = parse_args()
	# TODO: Check batcher is operating as expected.
	batcher  = GANBatcher(args.fake_input_file, args.vocab_file, 
						  args.data_dir, args.batch_size, args.seq_length)

	with open(os.path.join(args.save_dir_GAN, 'config.pkl'), 'w') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.save_dir_GAN, args.vocab_file), 'w') as f:
		cPickle.dump((batcher.chars, batcher.vocab), f)

	# global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

	with tf.variable_scope('gan') as scope:
		gan_gen = GAN(args, train_method='train_gen')
		scope.reuse_variables()
		gan_dis = GAN(args, train_method='train_dis')

	tvars = tf.trainable_variables()
	saver = tf.train.Saver(tvars)
	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
	with tf.Session(config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)) as sess:
		tf.set_random_seed(1)
		writer = tf.train.SummaryWriter(args.log_dir, sess.graph)

		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		
		adversarial_training(sess, gan_gen, gan_dis, writer, saver, args)

		# Components of adversarial training.
		# reset_reviews(args.data_dir, args.fake_input_file)
		# generate_samples(sess, gan_gen, args)
		# train_generator(sess, gan_gen, args, writer)
		# train_discriminator(sess, gan_dis, args, writer)