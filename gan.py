import theano
theano.config.on_unused_input = 'ignore'
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

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--sequence_length', default=100)
    argparser.add_argument('--batch_size', default=100)
    argparser.add_argument('--log', default='log.txt')
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Retrieve the text encoding
    with open('data/charnet-encoding.pkl', 'rb') as fp:
        text_encoding = pickle.load(fp)

    logging.debug("Converting to one-hot...")

    # Define the GAN model
    logging.debug('Compiling model...')
    discriminator_i = (Sequence(Vector(len(text_encoding))) >> Repeat(LSTM(1024), 2) >> Softmax(2)).freeze()
    with open('models/discriminative-model.pkl', 'rb') as fp:
        state = pickle.load(fp)
        discriminator_i.set_state(state)
    discriminator = discriminator_i.left.right >> discriminator_i.right
    
    generator = Generate(Vector(len(text_encoding)) >> Repeat(LSTM(1024), 2) >> Softmax(len(text_encoding)), args.sequence_length)
    with open('models/generative-model-original.pkl', 'rb') as fp:
        generator.set_state(pickle.load(fp))

    # Generator outputs to discriminator
    gan = generator >> discriminator
    
    # Optimization 
    rmsprop = RMSProp(gan, ConvexSequentialLoss(CrossEntropy(), 0.5))

    # Train the generative adversarial model
    def iterate(iterations, step_size):
        with open(args.log, 'w') as fp:
            for _ in xrange(iterations):
                index = text_encoding.encode('<STR>')
                batch = np.tile(text_encoding.convert_representation([index]), (args.batch_size, 1))
                y = np.tile([0, 1], (args.sequence_length, args.batch_size, 1))
                loss = rmsprop.train(batch, y, step_size)
                print >> fp,  "Loss[%u]: %f" % (_, loss)
                print "Loss[%u]: %f" % (_, loss)
                fp.flush()
        
        #with open('models/current-gan-model.pkl', 'wb') as fp:
        #    pickle.dump(gan.get_state(), fp)

    def generate_sample():
        pred_seq = generator.predict(np.eye(100)[None,0])
        print NumberSequence(pred_seq.argmax(axis=2).ravel()).decode(text_encoding)