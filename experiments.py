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

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def predict(text):
    '''Return prediction array at each time-step of input text'''
    char_seq   = CharacterSequence.from_string(text)
    num_seq    = char_seq.encode(text_encoding_D)
    num_seq_np = num_seq.seq.astype(np.int32)
    X          = np.eye(len(text_encoding_D))[num_seq_np]
    return discriminator.predict(X)

def text_to_num(text):
	'''Convert text to number representation'''
	char_seq   = CharacterSequence.from_string(text)
	num_seq    = char_seq.encode(text_encoding_D)
	num_seq_np = num_seq.seq.astype(np.int32)
	X          = np.eye(len(text_encoding_D))[num_seq_np]
	return X


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



if __name__ == '__main__':
	logging.debug('Loading encoding...')
	with open('data/charnet-encoding.pkl', 'rb') as fp:
		text_encoding_D = pickle.load(fp)
		text_encoding_D.include_stop_token  = False
		text_encoding_D.include_start_token = False

	discriminator_0 = Sequence(Vector(len(text_encoding_D))) >> (Repeat(LSTM(1024), 2) >> Softmax(2))
	discriminator_1 = Sequence(Vector(len(text_encoding_D))) >> (Repeat(LSTM(1024), 2) >> Softmax(2))

	logging.debug('Loading discriminators...')
	with open('models/discriminative/discriminative-model-0.0.renamed.pkl', 'rb') as fp:
		state = pickle.load(fp)
		state = (state[0][0], (state[0][1], state[1]))
		discriminator_0.set_state(state)

	with open('models/discriminative/discriminative-model-1.2.pkl', 'rb') as fp:
		state = pickle.load(fp)
		state = (state[0][0], (state[0][1], state[1]))
		discriminator_1.set_state(state)		

	noise_test(5)





