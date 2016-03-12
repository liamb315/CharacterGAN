import numpy as np
import cPickle as pickle
import theano
import sys
import csv
import logging
import random
from dataset import *
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *
from argparse import ArgumentParser
theano.config.on_unused_input = 'ignore'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def parse_args():
	argparser = ArgumentParser()
	argparser.add_argument("reviews")
	argparser.add_argument("--log", default="loss/generator_loss_current.txt")
	return argparser.parse_args()


class WindowedBatcher(object):

	def __init__(self, sequences, encodings, batch_size=100, sequence_length=50):
		self.sequences = sequences

		self.pre_vector_sizes = [c.seq[0].shape[0] for c in self.sequences]
		self.pre_vector_size = sum(self.pre_vector_sizes)

		self.encodings = encodings
		self.vocab_sizes = [c.index for c in self.encodings]
		self.vocab_size = sum(self.vocab_sizes)
		self.batch_index = 0
		self.batches = []
		self.batch_size = batch_size
		self.sequence_length = sequence_length + 1
		self.length = len(self.sequences[0])

		self.batch_index = 0
		self.X = np.zeros((self.length, self.pre_vector_size))
		self.X = np.hstack([c.seq for c in self.sequences])

		N, D = self.X.shape
		assert N > self.batch_size * self.sequence_length, "File has to be at least %u characters" % (self.batch_size * self.sequence_length)

		self.X = self.X[:N - N % (self.batch_size * self.sequence_length)]
		self.N, self.D = self.X.shape
		self.X = self.X.reshape((self.N / self.sequence_length, self.sequence_length, self.D))

		self.N, self.S, self.D = self.X.shape

		self.num_sequences = self.N / self.sequence_length
		self.num_batches = self.N / self.batch_size
		self.batch_cache = {}

	def next_batch(self):
		idx = (self.batch_index * self.batch_size)
		if self.batch_index >= self.num_batches:
			self.batch_index = 0
			idx = 0

		if self.batch_index in self.batch_cache:
			batch = self.batch_cache[self.batch_index]
			self.batch_index += 1
			return batch

		X = self.X[idx:idx + self.batch_size]
		y = np.zeros((X.shape[0], self.sequence_length, self.vocab_size))
		for i in xrange(self.batch_size):
			for c in xrange(self.sequence_length):
				seq_splits = np.split(X[i, c], np.cumsum(self.pre_vector_sizes))
				vec = np.concatenate([e.convert_representation(split) for
									  e, split in zip(self.encodings, seq_splits)])
				y[i, c] = vec

		X = y[:, :-1, :]
		y = y[:, 1:, :self.vocab_sizes[0]]

		X = np.swapaxes(X, 0, 1)
		y = np.swapaxes(y, 0, 1)
		self.batch_index += 1
		return X, y


def generate_number_samples(num_reviews):
	'''Generate a batch of samples from the current version of the generator'''
	pred_seq = generator_sample.predict(np.tile(np.eye(100)[0], (num_reviews, 1)))
	return pred_seq


def generate_text_samples(num_reviews):
	'''Generate fake reviews using the current generator'''
	pred_seq = generate_number_samples(num_reviews).argmax(axis=2).T
	num_seq  = [NumberSequence(pred_seq[i]).decode(text_encoding) for i in xrange(num_reviews)]
	return_str = [''.join(n.seq) for n in num_seq]
	return return_str


def generate_training_set(gan_versions=100, reviews_per_gan=3000, train_iter=100, step_size=100):
	'''Generate a reviews classically  Note:  Reviews may contain non-unicode characters'''
	with open('data/fake_beer_reviews_0.1_30000.txt', 'wb') as f:
		for i in xrange(gan_versions):	
			logging.debug('Generating reviews...')		
			reviews = generate_text_samples(reviews_per_gan)
			
			logging.debug('Appending reviews to file...')
			for review in reviews:
				print >> f, review

			logging.debug('Training generator...')
			train_generator(train_iter, step_size)


if __name__ == '__main__':
	args = parse_args()
	
	logging.debug('Reading file...')
	with open(args.reviews, 'r') as f:
		reviews = [r[3:] for r in f.read().strip().split('\n')]
		reviews = [r.replace('\x05',  '') for r in reviews]
		reviews = [r.replace('<STR>', '') for r in reviews]

	logging.debug('Retrieving text encoding...')
	with open('data/charnet-encoding.pkl', 'rb') as fp:
		text_encoding = pickle.load(fp)

	# Create reviews and targets
	logging.debug('Converting to one-hot...')
	review_sequences = [CharacterSequence.from_string(r) for r in reviews]
	num_sequences    = [c.encode(text_encoding) for c in review_sequences]
	final_sequences  = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in num_sequences]))

	# Batcher and generator
	batcher          = WindowedBatcher([final_sequences], [text_encoding], sequence_length=200, batch_size=100)
	generator        = Sequence(Vector(len(text_encoding), batch_size=100)) >> Repeat(LSTM(1024, stateful=True), 2) >> Softmax(len(text_encoding))
	generator_sample = Generate(Vector(len(text_encoding)) >> Repeat(LSTM(1024), 2) >> Softmax(len(text_encoding)), 500)
	
	# Tie the weights
	generator_sample = generator_sample.tie(generator)

	# logging.debug('Loading prior model...')
	# with open('models/generative/generative-model-current.pkl', 'rb') as fp:
	# 	generator.set_state(pickle.load(fp))
	
	logging.debug('Compiling graph...')
	rmsprop = RMSProp(generator, CrossEntropy(), clip_gradients=500)
	#rmsprop = RMSProp(generator, CrossEntropy())

	def train_generator(iterations, step_size):
		with open(args.log, 'w') as f:
			for _ in xrange(iterations):
				X, y = batcher.next_batch()
				# grads = rmsprop.gradient(X, y)
				# if grads:
				# 	for g in grads:
				# 		print np.linalg.norm(np.asarray(g))
				loss = rmsprop.train(X, y, step_size)
				print >> f, 'Loss[%u]: %f' % (_, loss)
				print 'Loss[%u]: %f' % (_, loss)
				f.flush()

		with open('models/generative/generative-model-current.pkl', 'wb') as g:
			pickle.dump(generator.get_state(), g)

