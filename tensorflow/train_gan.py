import tensorflow as tf
import numpy as np
import logging
from tensorflow.models.rnn import *
from argparse import ArgumentParser
from batcher_gan import DiscriminatorBatcher, GANBatcher
from gan_categorical import GAN
from discriminator import Discriminator
from generator import Generator
import time
import os
import cPickle

logger = logging.getLogger()
logger.setLevel(logging.ERROR)


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
	parser.add_argument('--save_dir_dis', type=str, default='models_GAN/discriminator',
		help='directory to store checkpointed discriminator models')
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
	parser.add_argument('-n', type=int, default=10,
		help='number of characters to sample')
	parser.add_argument('--prime', type=str, default=' ',
		help='prime text')
	parser.add_argument('--num_epochs_GAN', type=int, default=25,
		help='number of epochs of GAN')
	parser.add_argument('--num_epochs_gen', type=int, default=1,
		help='number of epochs to train generator')
	parser.add_argument('--num_epochs_dis', type=int, default=1,
		help='number of epochs to train discriminator')
	parser.add_argument('--save_every', type=int, default=500,
		help='save frequency')
	parser.add_argument('--grad_clip', type=float, default=5.,
		help='clip gradients at this value')
	parser.add_argument('--learning_rate_gen', type=float, default=0.05,
		help='learning rate')
	parser.add_argument('--learning_rate_dis', type=float, default=0.0002,
		help='learning rate for discriminator')
	parser.add_argument('--decay_rate', type=float, default=0.97,
		help='decay rate for rmsprop')
	parser.add_argument('--keep_prob', type=float, default=0.5,
		help='keep probability for dropout')
	parser.add_argument('--vocab_size', type=float, default=5,
		help='size of the vocabulary (characters)')
	return parser.parse_args()


def train_generator(gan, args, sess, train_writer, weights_load = 'random'):
	'''Train Generator via GAN.'''
	logging.debug('Training generator...')

	batcher  = GANBatcher(args.fake_input_file, args.vocab_file, 
						  args.data_dir, args.batch_size, 
						  args.seq_length)

	logging.debug('Vocabulary...')
	# TODO:  Why do this each time? Unnecessary
	with open(os.path.join(args.save_dir_GAN, 'config.pkl'), 'w') as f:
		cPickle.dump(args, f)
	with open(os.path.join(args.save_dir_GAN, 'simple_vocab.pkl'), 'w') as f:
		cPickle.dump((batcher.chars, batcher.vocab), f)

	# Save all GAN variables to gan_saver
	gan_vars = [v for v in tf.all_variables() if not 
					(v.name.startswith('classic/'))]
	gan_saver = tf.train.Saver(gan_vars)


	if weights_load is 'random':
		print('Random GAN parameters')

	elif weights_load is 'gan':
		print('Initial load of GAN parameters from %s' % args.save_dir_GAN)		
		ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
		if ckpt and ckpt.model_checkpoint_path:
			gan_saver.restore(sess, ckpt.model_checkpoint_path)

	elif weights_load is 'discriminator':
		print('Update GAN parameters from Discriminator of %s' % args.save_dir_dis)		
		# TODO:  Trainable vs. All Variables?
		dis_vars = [v for v in tf.trainable_variables() if 
					v.name.startswith('discriminator/')]
		dis_saver = tf.train.Saver(dis_vars)
		ckpt = tf.train.get_checkpoint_state(args.save_dir_dis)
		if ckpt and ckpt.model_checkpoint_path:
			dis_saver.restore(sess, ckpt.model_checkpoint_path)

	else:
		raise Exception('Invalid weight initialization for GAN: %s' % weights_load)

	for epoch in xrange(args.num_epochs_gen):
		new_lr = args.learning_rate_gen * (args.decay_rate ** epoch)
		sess.run(tf.assign(gan.lr_gen, new_lr))
		batcher.reset_batch_pointer()
		state_gen = gan.initial_state_gen.eval()
		state_dis = gan.initial_state_dis.eval()

		# for batch in xrange(50):
		for batch in xrange(batcher.num_batches):
			start = time.time()
			feed  = {gan.initial_state_gen: state_gen, 
					gan.initial_state_dis: state_dis}
	
			gen_train_loss, gen_summary, state_gen, state_dis, _ = sess.run([
				gan.gen_cost, 
				gan.merged,
				gan.final_state_gen,
				gan.final_state_dis, 
				gan.gen_train_op], feed)

			train_writer.add_summary(gen_summary, batch)
			end   = time.time()

			print '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}' \
				.format(epoch * batcher.num_batches + batch,
					args.num_epochs_gen * batcher.num_batches,
					epoch, gen_train_loss, end - start)
			
			if (epoch * batcher.num_batches + batch) % args.save_every == 0:
				checkpoint_path = os.path.join(args.save_dir_GAN, 'model.ckpt')
				gan_saver.save(sess, checkpoint_path, global_step = epoch * batcher.num_batches + batch)
				print 'GAN model saved to {}'.format(checkpoint_path)



def train_discriminator(discriminator, args, sess):
	'''Train the discriminator via classical approach'''
	logging.debug('Training discriminator...')
	
	batcher  = DiscriminatorBatcher(args.real_input_file, 
									args.fake_input_file, 
									args.data_dir, args.vocab_file,
									args.batch_size, args.seq_length)

	logging.debug('Vocabulary...')
	with open(os.path.join(args.save_dir_GAN, 'simple_vocab.pkl'), 'w') as f:
		cPickle.dump((batcher.chars, batcher.vocab), f)

	logging.debug('Loading GAN parameters to Discriminator...')
	dis_vars = [v for v in tf.trainable_variables() if v.name.startswith('classic/')]
	dis_dict = {}
	for v in dis_vars:
		# Key:    op.name in GAN Checkpoint file
		# Value:  Local generator Variable 
		dis_dict[v.op.name.replace('classic/','discriminator/')] = v
	dis_saver = tf.train.Saver(dis_dict)

	ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
	if ckpt and ckpt.model_checkpoint_path:
		dis_saver.restore(sess, ckpt.model_checkpoint_path)
	
	for epoch in xrange(args.num_epochs_dis):
		# Anneal learning rate
		new_lr = args.learning_rate_dis * (args.decay_rate ** epoch)
		sess.run(tf.assign(discriminator.lr, new_lr))
		batcher.reset_batch_pointer()
		state = discriminator.initial_state.eval()

		for batch in xrange(200):
		# for batch in xrange(batcher.num_batches):
			start = time.time()
			x, y  = batcher.next_batch()

			feed  = {discriminator.input_data: x, 
					 discriminator.targets: y, 
					 discriminator.initial_state: state}
			train_loss, state, _ = sess.run([discriminator.cost,
											discriminator.final_state,
											discriminator.train_op], 
											feed)
			end   = time.time()
			
			print '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}' \
				.format(epoch * batcher.num_batches + batch,
					args.num_epochs_dis * batcher.num_batches,
					epoch, train_loss, end - start)
			
			if (epoch * batcher.num_batches + batch) % args.save_every == 0:
				checkpoint_path = os.path.join(args.save_dir_dis, 'model.ckpt')
				dis_saver.save(sess, checkpoint_path, global_step = epoch * batcher.num_batches + batch)
				print 'Discriminator model saved to {}'.format(checkpoint_path)

			
# def generate_samples(generator, args, sess, num_samples=500, weights_load = 'random'):
# 	'''Generate samples from the current version of the GAN

# 	Args:
# 		generator: 
# 		args:
# 		sess:
# 		num_samples:
# 		weights_load:  
# 	'''
# 	samples = []
# 	with open(os.path.join(args.save_dir_GAN, 'config.pkl')) as f:
# 		saved_args = cPickle.load(f)
# 	with open(os.path.join(args.save_dir_GAN, args.vocab_file)) as f:
# 		chars, vocab = cPickle.load(f)
	
# 	if weights_load == 'random':
# 		logging.debug('Loading random parameters to Generator...')
# 	elif weights_load == 'GAN':
# 		logging.debug('Loading GAN parameters to Generator...')
# 		gen_vars = [v for v in tf.all_variables() if v.name.startswith('sampler/')]
# 		gen_dict = {}
# 		for v in gen_vars:
# 			# Key:    op.name in GAN Checkpoint file
# 			# Value:  Local generator Variable 
# 			gen_dict[v.op.name.replace('sampler/','')] = v
# 		gen_saver = tf.train.Saver(gen_dict)
# 		ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
# 		if ckpt and ckpt.model_checkpoint_path:
# 			gen_saver.restore(sess, ckpt.model_checkpoint_path)
# 	else:
# 		raise ValueError('Cannot restore parameters from: {}'.format(weights_load))

# 	assert num_samples / args.batch_size > 0, 'Generating only %d samples for'\
# 					' a batch_size of %d.' % (num_samples, saved_args.batch_size)
# 	num_batches = num_samples / saved_args.batch_size

# 	return generator.generate_samples(sess, num_batches, saved_args, chars, 
# 												   vocab, args.n)


def generate_samples(sess, gan, args):
	'''Generate samples.'''
	with open(os.path.join(args.save_dir_GAN, args.vocab_file)) as f:
		chars, vocab = cPickle.load(f)

	data_file = os.path.join(args.data_dir, args.fake_input_file)
	indices = sess.run(gan.sample_op)
	int_to_char = lambda x: chars[x]
	mapfunc = np.vectorize(int_to_char)
	samples = mapfunc(indices.T)
	print('Generating samples to %s' % data_file)    
	with open(data_file, 'a+') as f:
		for line in samples:
			print line
			print>>f, ''.join(line) 

	
def reset_reviews(data_dir, file_name):
	'''Clear the file containing the generated reviews.
	Args:
		data_dir:  Directory to store generated reviews.
		file_name:  Name of file containing generated reviews.
	'''
	open(os.path.join(data_dir, file_name), 'w').close()


def adversarial_training(gan, discriminator, generator, train_writer, args, sess):
	'''Adversarial Training'''
	train_generator(gan, args, sess, train_writer, weights_load = 'random')
	generate_samples(generator, args, sess, 200, init = True)

	for epoch in xrange(args.num_epochs_GAN):
		train_discriminator(discriminator, args, sess)
		train_generator(gan, args, sess, train_writer, weights_load = 'discriminator')
		reset_reviews(args.data_dir, args.fake_input_file)
		generate_samples(generator, args, sess, 200, weights_load='GAN')


if __name__=='__main__':
	args = parse_args()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)) as sess:
		tf.set_random_seed(1)
		logging.debug('Creating models...')
		gan = GAN(args)
		# with tf.variable_scope('classic'):
		# 	discriminator = Discriminator(args, is_training = True)

		logging.debug('TensorBoard...')
		train_writer = tf.train.SummaryWriter(args.log_dir, sess.graph)

		# logging.debug('Initializing variables in graph...')
		init_op = tf.initialize_all_variables()
		sess.run(init_op)

		# reset_reviews(args.data_dir, args.fake_input_file)
		# adversarial_training(gan, discriminator, generator, train_writer, args, sess)
		generate_samples(sess, gan, args)
		# train_generator(gan, args, sess, train_writer, weights_load = 'random')
		
		# train_discriminator(discriminator, args, sess)
