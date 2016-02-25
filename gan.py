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

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--sequence_length', default=200)
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
    
    logging.debug('Compiling human-readable generator...')
    generator2 = Sequence(Vector(len(text_encoding_G), batch_size=1)) >> Repeat(LSTM(1024, stateful=True), 2) >> Softmax(len(text_encoding_G))

    logging.debug('Compiling GAN...')
    gan = generator >> discriminator.right

    # Optimization for the generator (G)
    # gan.right.frozen = True
    # assert gan.get_parameters() == gan.left.get_parameters()
    rmsprop_G = RMSProp(gan.left >> Freeze(gan.right), ConvexSequentialLoss(CrossEntropy(), 0.5))

    # Optimization for the discrimator (D)
    # gan.right.frozen = False
    # assert len(discriminator.get_parameters()) > 0
    rmsprop_D = RMSProp(discriminator, CrossEntropy(), clip_gradients=5)


    ##########
    # Stage I 
    ##########
    # Load parameters after chaining operations due to known issue in DeepX
    with open('models/generative/generative-model-0.1.renamed.pkl', 'rb') as fp:
        generator.set_state(pickle.load(fp))

    with open('models/generative/generative-model-0.1.renamed.pkl', 'rb') as fp:
        generator2.set_state(pickle.load(fp))

    with open('models/discriminative/discriminative-model-0.0.renamed.pkl', 'rb') as fp:
        state = pickle.load(fp)
        state = (state[0][0], (state[0][1], state[1]))
        discriminator.set_state(state)
    
    # with open('models/gan/   ', 'rb') as fp:
    #     gan.set_state(pickle.load(fp))


    def generate_sample():
        '''Generate a sample from the current version of the generator'''
        pred_seq = generator.predict(np.eye(100)[None,0])
        num_seq  = NumberSequence(pred_seq.argmax(axis=2).ravel()).decode(text_encoding_G)
        return_str = ''.join(num_seq.seq)
        return_str = return_str.replace('<STR>', '').replace('<EOS>', '')
        return return_str


    def generate_sample_2(length):
        '''Generate a sample from the current version of the generator'''
        characters = [np.array([0])]
        generator2.reset_states()
        for i in xrange(length):
            output = generator2.predict(np.eye(len(text_encoding_G))[None, characters[-1]])
            sample = np.random.choice(xrange(len(text_encoding_G)), p=output[0, 0])
            characters.append(np.array([sample]))
        characters =  np.array(characters).ravel()
        return ''.join([text_encoding_G.decode(c) for c in characters[1:]])


    def generate_fake_reviews(num_reviews, len_reviews):
        '''Generate fake reviews using the current generator'''
        fake_reviews = []
        
        for _ in xrange(num_reviews):
            review = generate_sample_2(len_reviews)
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
                
                # if i > 10:
                #     avg_loss_delta = avg_loss[-2]/avg_loss[-1] - 1
                #     if avg_loss_delta < stop_criteria:
                #         return 
        
        
    def train_discriminator(iterations, step_size, real_reviews, fake_reviews, stop_criteria=0.001):
        '''Train the discriminator (D) on real and fake reviews'''
        random.seed(1)
        
        # Load and shuffle reviews
        real_targets, fake_targets = [],  []
        for _ in xrange(len(real_reviews)):
            real_targets.append([0, 1])
        for _ in xrange(len(fake_reviews)):
            fake_targets.append([1, 0])

        all_reviews = zip(real_reviews, real_targets) + zip(fake_reviews, fake_targets)
        random.shuffle(all_reviews)
        
        reviews, targets = zip(*all_reviews) 

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
                
                # if i > 10:
                #     avg_loss_delta = avg_loss[-2]/avg_loss[-1] - 1
                #     if avg_loss_delta < stop_criteria:
                #         return 


    def alternating_gan(num_epoch, dis_iter, gen_iter, dis_lr=1, gen_lr=1, num_reviews = 25, seq_length=args.sequence_length):
        '''Alternating GAN procedure for jointly training the generator (G) 
        and the discriminator (D)'''

        logging.debug('Loading real reviews...')
        with open('data/real_beer_reviews.txt', 'r') as f:
            real_reviews_all = [r[3:] for r in f.read().strip().split('\n')]
            real_reviews_all = [r.replace('\x05',  '') for r in real_reviews_all] 
            real_reviews_all = [r.replace('<STR>', '') for r in real_reviews_all]
        
        logging.debug('Generating fake reviews...')
        fake_reviews = generate_fake_reviews(num_reviews, seq_length) 

        with open(args.log, 'w') as fp:
            print >> fp, 'Alternating GAN for ',num_epoch,' epochs.'

        for i in xrange(num_epoch):
            logging.debug('Training discriminator...')
            last_review  = np.random.randint(num_reviews, len(real_reviews_all)) 
            real_reviews = real_reviews_all[last_review-num_reviews : last_review] 
            train_discriminator(dis_iter, dis_lr, real_reviews, fake_reviews)

            logging.debug('Training generator...')
            train_generator(gen_iter, gen_lr) 
            
            logging.debug('Generating new fake reviews...')
            generator2.set_state(generator.get_state())

            fake_reviews = generate_fake_reviews(num_reviews, seq_length) 

            # fake_labels = []
            # for _ in xrange(len(fake_reviews)):
            #     fake_labels.append([1,0])
            # fake_labels = np.asarray(fake_labels)
            # print classification_accuracy(fake_reviews, fake_labels)

            with open('data/gan/gan_reviews_'+str(i)+'.txt', 'wb') as f:
                for review in fake_reviews[:10]:
                    print review
                    print >> f, review

            logging.debug('Saving models...')
            with open('models/gan/gan-model-epoch'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(gan.get_state(), f)

            with open('models/generative/generative-model-epoch-'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(generator.get_state(), f)

            with open('models/discriminative/discriminative-model-epoch-'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(discriminator.get_state(), f)



