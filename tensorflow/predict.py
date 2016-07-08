import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import time
import os
import cPickle
from batcher import DiscriminatorBatcher
from discriminator import Discriminator

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('text', 
        help='string of text to predict')
    parser.add_argument('--save_dir', type=str, default='models_discriminator',
                       help='model directory to store checkpointed models')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory containing reviews')
    return parser.parse_args()

def predict(args):
    with open(os.path.join(args.save_dir, 'config.pkl')) as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'combined_vocab.pkl')) as f:
        _, vocab = cPickle.load(f)
    model = Discriminator(saved_args, is_training = False)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt  = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return model.predict(sess, args.text, vocab)

if __name__=='__main__':
    args = parse_args()
    with tf.device('/gpu:3'):
        probs = predict(args)
    
    for char, prob in zip(args.text, probs):
        print char, prob