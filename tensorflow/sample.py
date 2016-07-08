import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import time
import os
import cPickle
from batcher import Batcher
from generator import Generator
from gan import GAN

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_dir_gen', type=str, default='models_generator',
                       help='model directory to store checkpointed models')
    parser.add_argument('--save_dir_GAN', type=str, default='models_GAN',
                       help='model directory to store checkpointed models')
    parser.add_argument('--data_dir_gen', type=str, default='data',
                        help='data directory containing reviews')
    parser.add_argument('--data_dir_GAN', type=str, default='data',
                        help='data directory containing reviews')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    return parser.parse_args()
    

def sample_generator(args, num_samples = 10):
    with open(os.path.join(args.save_dir_GAN, 'config.pkl')) as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir_GAN, 'real_beer_vocab.pkl')) as f:
        chars, vocab = cPickle.load(f)
    generator = Generator(saved_args, is_training = False, batch = True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # for i in range(num_samples):
            #     print 'Review',i,':', generator.generate(sess, chars, vocab, args.n, args.prime), '\n'

            return generator.generate_batch(sess, saved_args, chars, vocab)

def sample_GAN(args, num_samples = 10):
    with open(os.path.join(args.save_dir_GAN, 'config.pkl')) as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir_GAN, 'simple_vocab.pkl')) as f:
        chars, vocab = cPickle.load(f)
    gan = GAN(saved_args, is_training = False)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir_GAN)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return gan.generate_samples(sess, saved_args, chars, vocab, args.n)

if __name__ == '__main__':
    args = parse_args()
    with tf.device('/gpu:3'):
        # sample_GAN(args)
        reviews = sample_GAN(args)
        for review in reviews:
            print ''.join(review)


