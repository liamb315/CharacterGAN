import numpy as np
import cPickle as pickle
import theano
import sys
import csv
import logging
import random
from dataset import *
from batcher import *
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *
from argparse import ArgumentParser
theano.config.on_unused_input = 'ignore'

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("reviews")
    argparser.add_argument("--log", default="loss/generator_loss_current.txt")
    return argparser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	
	logging.debug('Reading file...')
	with open(args.reviews, 'r') as f:
		reviews = [r[3:] for r in f.read().strip().split('\n')]
		reviews = [r.replace('\x05',  '') for r in reviews]
		reviews = [r.replace('<STR>', '') for r in reviews]

	# Create reviews and targets


	with open('data/charnet-encoding.pkl', 'rb') as fp:
        text_encoding = pickle.load(fp)

    # Construct the batcher
    batcher = WindowedBatcher([final_seq], [text_encoding], final_target, sequence_length=200, batch_size=100)

    logging.debug("Compiling discriminator...")
	generator = Generate(Vector(len(text_encoding)) >> Repeat(LSTM(1024), 2) >> Softmax(len(text_encoding)), args.sequence_length)

	# Optimization procedure
    rmsprop = RMSProp(discriminator, CrossEntropy(), clip_gradients=5)

	# def train_generator():
	# 	with open(args.log, 'w') as f:
	# 		