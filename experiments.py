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

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


###############
# Experiment 1
###############

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


def noise_test(num_reviews, fractional_noise = 0.2, distribution='uniform'):
	'''Test performance of the discriminator with noise added to one-hot vectors'''

	reviews     = load_reviews('data/fake_beer_reviews.txt')[:num_reviews]
	# reviews_seq = [text_to_num(r) for r in reviews]
	# shape       = reviews.shape

	for i, review in enumerate(reviews):
		print 'Review #%i'%(i)
		print review, '\n'
		num_seq = text_to_num(review)
		shape    = num_seq.shape

		print '  Unperturbed: ', discriminator.predict(num_seq)[-1]

		if distribution is 'constant':
			noise = fractional_noise * np.ones(shape)
		elif distribution is 'uniform':
			noise = np.random.uniform(0.0, fractional_noise, shape)

		blurred = num_seq + noise		

		print '  Perturbed:   ', discriminator.predict(blurred)[-1], '\n'





if __name__ == '__main__':

	logging.debug('Loading encoding...')
	with open('data/charnet-encoding.pkl', 'rb') as fp:
        text_encoding_D = pickle.load(fp)
        text_encoding_D.include_stop_token  = False
        text_encoding_D.include_start_token = False

    logging.debug('Compiling dscriminator...')
	discriminator = Sequence(Vector(len(text_encoding_D))) >> (Repeat(LSTM(1024), 2) >> Softmax(2))

	noise_test(5000, 0.2, uniform)





