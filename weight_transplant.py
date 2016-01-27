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

    argparser.add_argument("real_file")
    argparser.add_argument("fake_file")
    argparser.add_argument("--log", default="loss/current_loss.txt")

    return argparser.parse_args()

def generate(length, temperature):
    results = charrnn.generate(
        np.eye(len(encoding))[encoding.encode("i")],
        length,
        temperature).argmax(axis=1)
    return NumberSequence(results).decode(encoding)


def create_data_batcher(reviews, targets, encoding, sequence_length=200, batch_size=100):
    '''Create a batcher for a set of reviews and targets given a text encoding'''
    logging.debug('Converting to one-hot...')
    review_seq   = [CharacterSequence.from_string(review) for review in reviews]
    
    num_seq      = [c.encode(encoding) for c in review_seq]
    target_seq   = [NumberSequence([target]).replicate(len(r)) for target, r in zip(targets, num_seq)]

    final_seq    = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in num_seq]))
    final_target = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in target_seq]))

    batcher = WindowedBatcher([final_seq], [encoding], final_target, sequence_length, batch_size)
    return batcher
 

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

    # Partition data
    reviews, targets = zip(*all_reviews[:100000])
    test_reviews, test_targets = zip(*all_reviews[100000:])

    with open('data/charnet-encoding.pkl', 'rb') as fp:
        text_encoding = pickle.load(fp)


    logging.debug("Converting to one-hot...")
    review_sequences = [CharacterSequence.from_string(review) for review in reviews]
    
    num_sequences = [c.encode(text_encoding) for c in review_sequences]
    target_sequences = [NumberSequence([target]).replicate(len(r)) for target, r in zip(targets, num_sequences)]
    final_seq = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in num_sequences]))
    final_target = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in target_sequences]))

    # Construct the batcher
    batcher = WindowedBatcher([final_seq], [text_encoding], final_target, sequence_length=200, batch_size=100)
    

    # batcher = create_data_batcher(reviews, targets, text_encoding)    
    # test_batcher = create_data_batcher(test_reviews, test_targets, text_encoding, sequence_length=200, batch_size=100)


    # Define the model
    logging.debug("Compiling model")
    discriminator = Sequence(Vector(len(text_encoding))) >> Repeat(LSTM(1024), 2) >> Softmax(2)
    with open('models/current-model.pkl', 'rb') as fp:
        discriminator.set_state(pickle.load(fp))
    
    # Training
    rmsprop = RMSProp(discriminator, ConvexSequentialLoss(CrossEntropy(), 0.5), clip_gradients=5)

    # Training loss
    train_loss = []
    def iterate(iterations, step_size):
        with open(args.log, 'w') as fp:
            for _ in xrange(iterations):
                X, y = batcher.next_batch()
                loss = rmsprop.train(X,y,step_size)
                print >> fp,  "Loss[%u]: %f" % (_, loss)
                print "Loss[%u]: %f" % (_, loss)
                fp.flush()
                train_loss.append(loss)
        with open('models/current-model-2.pkl', 'wb') as fp:
            pickle.dump(discriminator.get_state(), fp)
          

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

   



