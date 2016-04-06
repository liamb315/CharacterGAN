import numpy as np
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

import cPickle as pickle
import theano
import sys
import csv
import logging
import random
from dataset import *
from batcher import *
from argparse import ArgumentParser
theano.config.on_unused_input = 'ignore'

logging.basicConfig(level=logging.DEBUG)

def parse_args():
	argparser = ArgumentParser()
	argparser.add_argument("real_file")
	argparser.add_argument("fake_file")
	argparser.add_argument("--log", default="loss/discriminative/discriminative-adversarial-loss-0.0.0.txt")
	return argparser.parse_args()




if __name__ == "__main__":
	args = parse_args()
	
	logging.debug("Reading file...")
	with open(args.real_file, 'r') as fp:
		real_reviews = [r[3:] for r in fp.read().strip().split('\n')]
		real_reviews = [r.replace('\x05',  '') for r in real_reviews] 
		real_reviews = [r.replace('<STR>', '') for r in real_reviews]
	with open(args.fake_file, 'r') as fp:
		fake_reviews = [r[3:] for r in fp.read().strip().split('\n')]
		fake_reviews = [r.replace('\x05',  '') for r in fake_reviews]
		fake_reviews = [r.replace('<STR>', '') for r in fake_reviews]

	# Load and shuffle reviews
	real_targets, fake_targets = [],  []
	for _ in xrange(len(real_reviews)):
		real_targets.append([0, 1])
	for _ in xrange(len(fake_reviews)):
		fake_targets.append([1, 0])

	all_reviews = zip(real_reviews, real_targets) + zip(fake_reviews, fake_targets)

	random.seed(1)
	random.shuffle(all_reviews)

	reviews, targets = zip(*all_reviews[:150000])

	logging.debug('Retrieving text encoding...')
	with open('data/charnet-encoding.pkl', 'rb') as fp:
		text_encoding = pickle.load(fp)
	text_encoding.include_stop_token  = False
	text_encoding.include_start_token = False

	logging.debug("Converting to one-hot...")
	review_sequences = [CharacterSequence.from_string(review.replace('<STR>', '\x00').replace('<EOS>', '\x01').replace('>', '').replace('<', '').replace('"','')) for review in reviews]
	
	num_sequences = [c.encode(text_encoding) for c in review_sequences]
	target_sequences = [NumberSequence([target]).replicate(len(r)) for target, r in zip(targets, num_sequences)]
	final_seq = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in num_sequences]))
	final_target = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in target_sequences]))

	# Construct the batcher
	batcher = WindowedBatcher([final_seq], [text_encoding], final_target, sequence_length=200, batch_size=100)
	
	logging.debug("Compiling discriminator...")
	
	# Failing here on refactor branch
	#  TypeError: unsupported operand type(s) for >>: 'Sequence' and 'Chain'
	discriminator = Sequence(Vector(len(text_encoding), batch_size=100)) >> Repeat(LSTM(1024, stateful=True), 2) >> Softmax(2)
	
	with open('models/discriminative/discriminative-model-0.0.renamed.pkl', 'rb') as fp:
	# with open('models/discriminative/discriminative-model-1.0.pkl', 'rb') as fp:
		discriminator.set_state(pickle.load(fp))

