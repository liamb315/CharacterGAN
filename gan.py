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
    discriminator = Sequence(Vector(len(text_encoding_D))) >> (Repeat(LSTM(1024), 2) >> Softmax(2))
    
    logging.debug('Compiling generator...')
    generator = Generate(Vector(len(text_encoding_G)) >> Repeat(LSTM(1024), 2) >> Softmax(len(text_encoding_G)), args.sequence_length)
    
    logging.debug('Compiling GAN...')
    gan = generator >> discriminator.right

    # # Optimization for the generator (G)
    # gan.right.frozen = True
    # assert gan.get_parameters() == gan.left.get_parameters()
    # rmsprop_G = RMSProp(gan, ConvexSequentialLoss(CrossEntropy(), 0.5))

    # # Optimization for the discrimator (D)
    # gan.right.frozen = False
    # assert len(discriminator.get_parameters()) > 0
    # rmsprop_D = RMSProp(discriminator, CrossEntropy(), clip_gradients=5)


    ##########
    # Stage I 
    ##########
    # Load parameters after chaining operations due to known issue in DeepX
    with open('models/generative-model-0.0.pkl', 'rb') as fp:
        generator.set_state(pickle.load(fp))

    with open('models/discriminative-model-0.2.pkl', 'rb') as fp:
        state = pickle.load(fp)
        state = (state[0][0], (state[0][1], state[1]))
        discriminator.set_state(state)
    

    def generate_sample():
        '''Generate a sample from the current version of the generator'''
        pred_seq = generator.predict(np.eye(100)[None,0])
        num_seq  = NumberSequence(pred_seq.argmax(axis=2).ravel()).decode(text_encoding_G)
        return_str = ''.join(num_seq.seq)
        return_str = return_str.replace('<STR>', '').replace('<EOS>', '')
        return return_str


    def generate_fake_reviews(num_reviews):
        '''Generate fake reviews using the current generator'''
        fake_reviews = []
        
        for _ in xrange(num_reviews):
            review = generate_sample()
            fake_reviews.append(review)
        
        fake_reviews = [r.replace('\x05',  '') for r in fake_reviews]
        fake_reviews = [r.replace('<STR>', '') for r in fake_reviews]
        fake_reviews = [r.replace('<EOS>', '') for r in fake_reviews]
        return fake_reviews


    def predict(text):
        '''Return prediction array at each time-step of input text'''
        char_seq   = CharacterSequence.from_string(text)
        num_seq    = char_seq.encode(text_encoding_D)
        num_seq_np = num_seq.seq.astype(np.int32)
        X          = np.eye(len(text_encoding_D))[num_seq_np]
        return discriminator.predict(X)


    ###########
    # Stage II 
    ###########

    def train_generator(iterations, step_size, stop_criteria=0.001):
        '''Train the generative model (G) via a GAN framework'''  
        # Optimization for the generator (G)
        gan.right.frozen = True
        assert gan.get_parameters() == gan.left.get_parameters()
        rmsprop_G = RMSProp(gan, ConvexSequentialLoss(CrossEntropy(), 0.5))
        rmsprop_G.reset_parameters()

        avg_loss = []
        with open(args.log, 'a+') as fp:
            for i in xrange(iterations):
                index = text_encoding_G.encode('<STR>')
                batch = np.tile(text_encoding_G.convert_representation([index]), (args.batch_size, 1))
                y = np.tile([0, 1], (args.sequence_length, args.batch_size, 1))
                loss = rmsprop_G.train(batch, y, step_size)
                if i == 0:
                    avg_loss.append(loss)
                avg_loss.append(loss * 0.05 + avg_loss[-1] * 0.95)
                
                print >> fp,  "Generator Loss[%u]: %f (%f)" % (i, loss, avg_loss[-1])
                print "Generator Loss[%u]: %f (%f)" % (i, loss, avg_loss[-1])
                fp.flush()
                # if i > 5:
                #     avg_loss_delta = avg_loss[-2]/avg_loss[-1] - 1
                #     if avg_loss_delta < stop_criteria:
                #         return 
        
        
    def train_discriminator(iterations, step_size, real_reviews, fake_reviews, stop_criteria=0.001):
        '''Train the discriminator (D) on real and fake reviews'''
        random.seed(1)
        
        # Optimization for the discrimator (D)
        gan.right.frozen = False
        assert len(discriminator.get_parameters()) > 0
        rmsprop_D = RMSProp(discriminator, CrossEntropy(), clip_gradients=5)
        rmsprop_D.reset_parameters()

        # Load and shuffle reviews
        real_targets, fake_targets = [],  []
        for _ in xrange(len(real_reviews)):
            real_targets.append([0, 1])
        for _ in xrange(len(fake_reviews)):
            fake_targets.append([1, 0])

        all_reviews = zip(real_reviews, real_targets) + zip(fake_reviews, fake_targets)
        random.shuffle(all_reviews)
        
        reviews, targets = zip(*all_reviews[:1000]) #TEMP:  Just testing

        logging.debug("Converting to one-hot...")
        review_sequences = [CharacterSequence.from_string(review) for review in reviews]
        num_sequences = [c.encode(text_encoding_D) for c in review_sequences]
        target_sequences = [NumberSequence([target]).replicate(len(r)) for target, r in zip(targets, num_sequences)]
        final_seq = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in num_sequences]))
        final_target = NumberSequence(np.concatenate([c.seq.astype(np.int32) for c in target_sequences]))

        # Construct the batcher
        batcher = WindowedBatcher([final_seq], [text_encoding_D], final_target, sequence_length=200, batch_size=100)
        
        avg_loss = []
        with open(args.log, 'a+') as fp:
            for i in xrange(iterations):
                X, y = batcher.next_batch()
                loss = rmsprop_D.train(X, y, step_size)
                if i == 0:
                    avg_loss.append(loss)
                avg_loss.append(loss * 0.05 + avg_loss[-1] * 0.95)

                print >> fp,  "Discriminator Loss[%u]: %f (%f)" % (i, loss, avg_loss[-1])
                print "Discriminator Loss[%u]: %f (%f)" % (i, loss, avg_loss[-1])
                fp.flush()
                
                # if i > 5:
                #     avg_loss_delta = avg_loss[-2]/avg_loss[-1] - 1
                #     if avg_loss_delta < stop_criteria:
                #         return 


    def alternating_gan(num_iter):
        '''Alternating GAN procedure for jointly training the generator (G) 
        and the discriminator (D)'''

        logging.debug('Loading real reviews...')
        with open('data/real_beer_reviews.txt', 'r') as f:
            real_reviews = [r[3:] for r in f.read().strip().split('\n')]
            real_reviews = [r.replace('\x05',  '') for r in real_reviews] 
            real_reviews = [r.replace('<STR>', '') for r in real_reviews]

        real_reviews = real_reviews[0:1000] #TEMP:  Just testing

        with open(args.log, 'w') as fp:
            print >> fp, 'Alternating GAN for ',num_iter,' iterations.'

        for i in xrange(num_iter):
            logging.debug('Training generator...')
            train_generator(50, 1) 
            
            logging.debug('Generating new fake reviews...')
            fake_reviews = generate_fake_reviews(1000)
            with open('data/gan_reviews_'+str(i)+'.txt', 'w') as f:
                for review in fake_reviews[0:10]:
                    print review
                    print >> f, review

            logging.debug('Training discriminator...')
            train_discriminator(25, 10, real_reviews, fake_reviews)

            with open('models/gan-model-current.pkl', 'wb') as f:
                pickle.dump(gan.get_state(), f)




