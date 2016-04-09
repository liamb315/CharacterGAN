import theano
theano.config.on_unused_input = 'ignore'
import numpy as np
import cPickle as pickle
import theano
import sys
import csv
import logging
import random
import Tkinter
from dataset import *
from dataset.sequence import *
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *
from deepx import backend as T
from argparse import ArgumentParser
from utils import *
import string

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def text_to_num(text):
	'''Convert text to number representation'''
	char_seq   = CharacterSequence.from_string(text)
	num_seq    = char_seq.encode(text_encoding_D)
	num_seq_np = num_seq.seq.astype(np.int32)
	X          = np.eye(len(text_encoding_D))[num_seq_np]
	return X

def predict(model, encoding, text, preprocess=True):
	'''Return prediction array at each time-step of input text'''
	if preprocess:
		text = text.replace('<STR>','')
		text = text.replace('<EOS>','')
	char_seq   = CharacterSequence.from_string(text)
	num_seq    = char_seq.encode(encoding)
	num_seq_np = num_seq.seq.astype(np.int32)
	X          = np.eye(len(encoding))[num_seq_np]
	return model.predict(X)


###############
# Experiment 1
###############
def noise_test(num_reviews, data_dir = 'data/fake_beer_reviews.txt', fractional_noise = 0.2, distribution='uniform'):
	'''Test performance of the discriminator with noise added to one-hot vectors'''

	reviews     = load_reviews(data_dir)
	last_review = np.random.randint(num_reviews, len(reviews))
	reviews     = reviews[last_review - num_reviews : last_review]
	reviews     = [r.replace('<STR>','').replace('<END>','').replace('<','').replace('>','') for r in reviews]

	for i, review in enumerate(reviews):
		print 'Review #%i'%(i)
		print review, '\n'
		num_seq = text_to_num(review)
		shape    = num_seq.shape
		print '  Unperturbed_0: ', discriminator_0.predict(num_seq)[-1]

		if distribution is 'constant':
			noise = fractional_noise * np.ones(shape)
			blurred = num_seq + noise	
		elif distribution is 'uniform':
			noise = np.random.uniform(0.0, fractional_noise, shape)
			blurred = num_seq + noise
		elif distribution is 'dirichlet':
			blurred = [np.random.dirichlet(num_seq[j,0,:] + fractional_noise) for j in xrange(len(num_seq))]
			blurred = np.asarray(blurred)
			blurred = blurred.reshape(shape)
		print '  Perturbed_0:   ', discriminator_0.predict(blurred)[-1], '\n'

		print '  Unperturbed_1: ', discriminator_1.predict(num_seq)[-1]
		print '  Perturbed_1:   ', discriminator_1.predict(blurred)[-1], '\n'

###############
# Experiment 2
###############

class DiscriminatorEvaluation(object):
	def __init__(self, models):
		self.models = models


	def load_sequences(self, num_sequences=100):
		sequences = {}
		sequences['real']       = load_reviews('data/real_beer_reviews.txt')[:num_sequences]
		sequences['fake']       = load_reviews('data/fake_beer_reviews.txt')[:num_sequences]
		if num_sequences <= 79:  #79 and above not in encoding
			sequences['repeat'] = [char*200 for char in string.printable[:num_sequences]]
		else:
			sequences['repeat'] = [char*200 for char in string.printable[:79]]
		sequences['random_char'] = load_reviews('data/curriculum/random_reviews_1.discriminator_0')[:num_sequences] 


	def manual_evaluation(self, sequence_dict, model_list):
		'''Review predictions on specific sequences'''
		pass

	def batch_evaluation(self, model):
		'''Return predictions on specific sequences'''
		pass





def discriminator_evaluation(models, encoding, num_sequences=5):
	# Sequence 0: Real reviews
	real_reviews = load_reviews('data/real_beer_reviews.txt')[:num_sequences]

	# Sequence 1: Fake reviews (Original)
	fake_reviews = load_reviews('data/fake_beer_reviews.txt')[:num_sequences]

	# Sequence 2: Repeating characters
	repeating_chars = [char*200 for char in string.printable[:79]] #79 and above not in encoding

	# Sequence 3: Repeating words 
	repeating_words = []

	# Sequence 4: Random characters
	random_chars = load_reviews('data/curriculum/random_reviews_1.0.txt')[:num_sequences]

	# Sequence 5: Random words
	random_words = []

	print 'Real'
	for review in real_reviews:
		print review
		for model in models:
			print predict(model, encoding, review)[-1]
		print '\n'

	print 'Fake'
	for review in fake_reviews:
		print review
		
		for model in models:
			print predict(model, encoding, review)[-1]
		print '\n'

	print 'Repeating'
	for review in repeating_chars:
		print review
		
		for model in models:
			print predict(model, encoding, review)[-1]
		print '\n'

	print 'Random'
	for review in random_chars:
		print review
		
		for model in models:
			print predict(model, encoding, review)[-1]
		print '\n'


def discriminator_histograms(models, encoding, num_sequences = 10):
	real_reviews = load_reviews('data/real_beer_reviews.txt')[:num_sequences]
	random_chars = load_reviews('data/curriculum/random_reviews_1.0.txt')[:num_sequences]

	results = {}

	for name in models.keys():
		print name
		results[name] = []

	print results

	for review in random_chars:
		print review
		for name, model in models.iteritems():

			print predict(model, encoding, review)[-1][0][0]
			results[name].append(predict(model, encoding, review)[-1][0][0]) 

	return results





if __name__ == '__main__':
	logging.debug('Loading encoding...')
	with open('data/charnet-encoding.pkl', 'rb') as fp:
		text_encoding_D = pickle.load(fp)
		text_encoding_D.include_stop_token  = False
		text_encoding_D.include_start_token = False

	discriminator_0 = Sequence(Vector(len(text_encoding_D))) >> (Repeat(LSTM(1024), 2) >> Softmax(2))
	discriminator_1 = Sequence(Vector(len(text_encoding_D))) >> (Repeat(LSTM(1024), 2) >> Softmax(2))
	discriminator_2 = Sequence(Vector(len(text_encoding_D))) >> Repeat(LSTM(1024) >> Dropout(0.5), 2) >> Softmax(2)
	discriminator_3 = Sequence(Vector(len(text_encoding_D))) >> (Repeat(LSTM(1024), 2) >> Softmax(2))
	discriminator_4 = Sequence(Vector(len(text_encoding_D))) >> Repeat(LSTM(1024) >> Dropout(0.5), 2) >> Softmax(2)

	logging.debug('Loading discriminators...')
	with open('models/discriminative/discriminative-model-0.0.0.pkl', 'rb') as fp:
		state = pickle.load(fp)
		state = (state[0][0], (state[0][1], state[1]))
		discriminator_0.set_state(state)

	with open('models/discriminative/discriminative-model-0.3.1.pkl', 'rb') as fp:
		discriminator_1.set_state(pickle.load(fp))

	with open('models/discriminative/discriminative-dropout-model-0.0.2.pkl', 'rb') as fp:
		discriminator_2.set_state(pickle.load(fp))

	with open('models/discriminative/discriminative-adversarial-model-0.0.0.pkl', 'rb') as fp:
		state = pickle.load(fp)
		state = (state[0][0], (state[0][1], state[1]))
		discriminator_3.set_state(state)

	with open('models/discriminative/discriminative-adversarial-dropout-model-0.0.0.pkl', 'rb') as fp:
		discriminator_4.set_state(pickle.load(fp))

	models = {
		'original': discriminator_0, 
		'mix': discriminator_1, 
		'dropout': discriminator_2,
		'adversarial': discriminator_3, 
		'adversarial_dropout': discriminator_4}

	# discriminator_evaluation(models, text_encoding_D, 5)

