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

    with open('data/charnet-encoding.pkl', 'rb') as fp:
        text_encoding = pickle.load(fp)

    logging.debug("Converting to one-hot...")

    # Define the model
    logging.debug("Compiling model")
    discriminator_i = (Sequence(Vector(len(text_encoding))) >> Repeat(LSTM(1024), 2) >> Softmax(2)).freeze()
    with open('models/current-model.pkl', 'rb') as fp:
        state = pickle.load(fp)
        discriminator_i.set_state(state)
    discriminator = discriminator_i.left.right >> discriminator_i.right
    generator = Generate(Vector(len(text_encoding)) >> Repeat(LSTM(1024), 2) >> Softmax(len(text_encoding)), args.sequence_length)
    with open('models/generative-model-original.pkl', 'rb') as fp:
        generator.set_state(pickle.load(fp))

    gan = generator >> discriminator
    
    # Training
    rmsprop = RMSProp(gan, ConvexSequentialLoss(CrossEntropy(), 0.5))

    # Training loss
    train_loss = []
    def iterate(iterations, step_size):
        with open(args.log, 'w') as fp:
            for _ in xrange(iterations):
                batch = np.tile(text_encoding.convert_representation([text_encoding.encode('<STR>')]), (args.batch_size, 1))
                y = np.tile([0, 1], (args.sequence_length, args.batch_size, 1))
                loss = rmsprop.train(batch, y, step_size)
                print >> fp,  "Loss[%u]: %f" % (_, loss)
                print "Loss[%u]: %f" % (_, loss)
                fp.flush()
                train_loss.append(loss)
        #with open('models/current-model-2.pkl', 'wb') as fp:
        #    pickle.dump(discriminator.get_state(), fp)
          

    # Train accuracy
    def train_accuracy(num_batches=100):
        '''Calculate the training accuracy over number of batches. Calculates
           accuracy based on prediction at final time-step'''
        errors = 0
        total  = 0

        for _ in xrange(num_batches):
            X, y = batcher.next_batch()
            pred = discriminator.predict(X)

            # Retrieve last label
            last_y = y[-1, :, :]
            last_p = pred[-1, :, :]

            errors += np.count_nonzero(last_y.argmax(axis=1) - last_p.argmax(axis=1))
            total  += batcher.batch_size

        return 1.0 - float(errors)/float(total)



