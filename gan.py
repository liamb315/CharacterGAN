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
from batcher import *
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *
from deepx import backend as T
from argparse import ArgumentParser

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--sequence_length', default=200)
    argparser.add_argument('--batch_size', default=30)
    argparser.add_argument('--log', default='loss/gan_log_current.txt')
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.debug('Retrieving text encoding...')
    with open('data/charnet-encoding.pkl', 'rb') as fp:
        text_encoding_G = pickle.load(fp)

    with open('data/charnet-encoding.pkl', 'rb') as fp:
        text_encoding_D = pickle.load(fp)
        text_encoding_D.include_stop_token  = False
        text_encoding_D.include_start_token = False

    logging.debug('Compiling discriminator...')
    discriminator = Sequence(Vector(len(text_encoding_D))) >> (LSTM(1024) >> Softmax(2))

    logging.debug('Compiling generator...')
    generator = Generate(Vector(len(text_encoding_G)) >> Repeat(LSTM(1024), 2) >> Softmax(len(text_encoding_G)), args.sequence_length)

    logging.debug('Compiling human-readable generator...')
    gennet = Sequence(Vector(len(text_encoding_G))) >> Repeat(LSTM(1024), 2) >> Softmax(len(text_encoding_G))
    generator = generator.tie(gennet)

    assert gennet.get_parameters() == generator.get_parameters()

    logging.debug('Compiling GAN...')
    gan = gennet >> discriminator.right

    # Optimization for the generator (G)
    rmsprop_G = RMSProp(gan.left >> Freeze(gan.right), CrossEntropy())

    # Optimization for the discrimator (D)
    # gan.right.frozen = False
    # assert len(discriminator.get_parameters()) > 0
    rmsprop_D = RMSProp(discriminator, CrossEntropy())


    ##########
    # Stage I
    ##########
    # Load parameters after chaining operations due to known issue in DeepX
    with open('models/generative/generative-model-0.1.renamed.pkl', 'rb') as fp:
        generator.set_state(pickle.load(fp))

    # with open('models/generative/generative-model-0.1.renamed.pkl', 'rb') as fp:
        # generator3.set_state(pickle.load(fp))

    with open('models/discriminative/discriminative-model-0.0.renamed.pkl', 'rb') as fp:
        state = pickle.load(fp)
        state = (state[0][0], (state[0][1], state[1]))
        discriminator.set_state(state)

    # with open('models/gan/   ', 'rb') as fp:
    #     gan.set_state(pickle.load(fp))

    def load_reviews(file_dir):
        '''Loads list of reviews from file_dir'''
        with open(file_dir, 'rb') as f:
            reviews = [r[3:] for r in f.read().strip().split('\n')]
            reviews = [r.replace('\x05',  '') for r in reviews]
            reviews = [r.replace('<STR>', '') for r in reviews]
        return reviews

    def generate_sample(num_reviews):
        '''Generate a sample from the current version of the generator'''
        pred_seq = generator.predict(np.tile(np.eye(100)[0], (num_reviews, 1)))
        return pred_seq

    def generate_fake_reviews(num_reviews):
        '''Generate fake reviews using the current generator'''
        pred_seq = generate_sample(num_reviews).argmax(axis=2).T
        num_seq  = [NumberSequence(pred_seq[i]).decode(text_encoding_D) for i in xrange(num_reviews)]
        return_str = [''.join(n.seq) for n in num_seq]
        return return_str

    def predict(text):
        '''Return prediction array at each time-step of input text'''
        char_seq   = CharacterSequence.from_string(text)
        num_seq    = char_seq.encode(text_encoding_D)
        num_seq_np = num_seq.seq.astype(np.int32)
        X          = np.eye(len(text_encoding_D))[num_seq_np]
        return discriminator.predict(X)

    def classification_accuracy(reviews, labels):
        '''Classification accuracy based on prediction at final time-step'''
        correct = 0.0
        for review, label in zip(reviews, labels):
            pred   = predict(review)[-1][0]
            print pred, label, pred.argmax() == label.argmax()
            if pred.argmax() == label.argmax():
                correct += 1
        return correct/len(reviews)


    ###########
    # Stage II
    ###########

    def train_generator(iterations, step_size, stop_criteria=0.001):
        '''Train the generative model (G) via a GAN framework'''

        avg_loss = []
        # rmsprop_G.reset_parameters()
        with open(args.log, 'a+') as fp:
            for i in xrange(iterations):
                batch = generate_sample(args.batch_size)
                starts = np.tile(np.eye(len(text_encoding_D))[0], (1, batch.shape[1], 1))
                batch = np.concatenate([starts, batch])[:-1]
                y = np.tile([0, 1], (args.sequence_length, args.batch_size, 1))
                loss = rmsprop_G.train(batch, y, step_size)
                rmsprop_G.model.reset_states()

                if i == 0:
                    avg_loss.append(loss)
                avg_loss.append(loss * 0.05 + avg_loss[-1] * 0.95)

                print >> fp,  "Generator Loss[%u]: %f (%f)" % (i, loss, avg_loss[-1])
                print "Generator Loss[%u]: %f (%f)" % (i, loss, avg_loss[-1])
                fp.flush()

                # if i > 10:
                #     avg_loss_delta = avg_loss[-2]/avg_loss[-1] - 1
                #     if avg_loss_delta < stop_criteria:
                #         return

    def train_discriminator(iterations, step_size, real_reviews, stop_criteria=0.001):
        '''Train the discriminator (D) on real and fake reviews'''
        random.seed(1)

        num_reviews = len(real_reviews)

        fake_reviews = generate_sample(num_reviews)

        # Load and shuffle reviews
        logging.debug("Converting to one-hot...")
        batches = []
        targets = []
        for i in xrange(len(real_reviews)):
            batches.append(np.eye(len(text_encoding_D))[None, CharacterSequence.from_string(real_reviews[i][:args.sequence_length]).encode(text_encoding_D).seq.ravel()])
            assert batches[-1].shape == (1, args.sequence_length, len(text_encoding_D)), batches[-1].shape
            targets.append(np.tile([0, 1], (1, args.sequence_length, 1)))
        for i in xrange(len(real_reviews)):
            batches.append(fake_reviews[None, :, i])
            assert batches[-1].shape == (1, args.sequence_length, len(text_encoding_D)), batches[-1].shape
            targets.append(np.tile([1, 0], (1, args.sequence_length, 1)))
        batches = np.concatenate(batches).swapaxes(0, 1)
        targets = np.concatenate(targets).swapaxes(0, 1)
        assert batches.shape == (args.sequence_length, num_reviews * 2, len(text_encoding_D)), batches.shape
        assert targets.shape == (args.sequence_length, num_reviews * 2, 2), targets.shape

        avg_loss = []
        # rmsprop_D.reset_parameters()
        with open(args.log, 'a+') as fp:
            for i in xrange(iterations):
                idx = np.random.permutation(xrange(batches.shape[1]))[:args.batch_size]
                X, y = batches[:, idx], targets[:, idx]
                loss = rmsprop_D.train(X, y, step_size)
                if i == 0:
                    avg_loss.append(loss)
                avg_loss.append(loss * 0.05 + avg_loss[-1] * 0.95)

                print >> fp,  "Discriminator Loss[%u]: %f (%f)" % (i, loss, avg_loss[-1])
                print "Discriminator Loss[%u]: %f (%f)" % (i, loss, avg_loss[-1])
                fp.flush()

    def alternating_gan(num_epoch, dis_iter, gen_iter, dis_lr=1, gen_lr=1, num_reviews = 100, seq_length=args.sequence_length):
        '''Alternating GAN procedure for jointly training the generator (G)
        and the discriminator (D)'''

        logging.debug('Loading real reviews...')
        real_reviews_all = load_reviews('data/real_beer_reviews.txt')

        logging.debug('Generating fake reviews...')

        fake_reviews = generate_fake_reviews(num_reviews)

        with open(args.log, 'w') as fp:
            print >> fp, 'Alternating GAN for ',num_epoch,' epochs.'

        for i in xrange(num_epoch):
            logging.debug('Training discriminator...')
            last_review  = np.random.randint(num_reviews, len(real_reviews_all))
            real_reviews = real_reviews_all[last_review : last_review + num_reviews]
            train_discriminator(dis_iter, dis_lr, real_reviews)

            logging.debug('Training generator...')
            train_generator(gen_iter, gen_lr)

            logging.debug('Generating new fake reviews...')
            fake_reviews = generate_fake_reviews(num_reviews)

            # fake_labels = []
            # for _ in xrange(len(fake_reviews)):
            #     fake_labels.append([1,0])
            # fake_labels = np.asarray(fake_labels)
            # print classification_accuracy(fake_reviews, fake_labels)

            with open('data/gan/gan_reviews_'+str(i)+'.txt', 'wb') as f:
                for review in fake_reviews[:10]:
                    print review
                    print >> f, review

            # logging.debug('Saving models...')
            # with open('models/gan/gan-model-epoch'+str(i)+'.pkl', 'wb') as f:
                # pickle.dump(gan.get_state(), f)

            # with open('models/generative/generative-model-epoch-'+str(i)+'.pkl', 'wb') as f:
                # pickle.dump(generator.get_state(), f)

            # with open('models/discriminative/discriminative-model-epoch-'+str(i)+'.pkl', 'wb') as f:
                # pickle.dump(discriminator.get_state(), f)

