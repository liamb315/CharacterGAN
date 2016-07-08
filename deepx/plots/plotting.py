import matplotlib.pyplot as plt
import numpy as np
import re
from argparse import ArgumentParser


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('log_file', default='../loss/gan_log_current.txt',
        help='The training loss log file for the GAN training')
    argparser.add_argument('predict_file', default='../data/sequences/predictions.txt',
        help='Predictions')
    return argparser.parse_args()


def plot_gan(args):
    '''Plot adversarial training'''
    with open(args.log_file, 'r') as f:
        next(f)
        train_loss = []
        labels     = []
        prev_label  = None
        # cur_label = 1

        for i, line in enumerate(f):
            train_loss.append(float(re.search('\((.*?)\)', line).group(1)))
            label = re.search('^\w+', line).group()
            

            if i == 0 or label != prev_label:
                labels.append(label)

            # if label == 'Generator':
            #     if cur_label % 5 == 0:
            #         labels.append(cur_label)
            #     else:
            #         labels.append('')
            #     cur_label += 1
                
            else:
                labels.append('')
            prev_label = label

    # fig = plt.figure()
    # ax  = fig.add_subplot(111)
    # x_min, x_max = ax.get_xlim()
    # ticks_scaled = [(tick - x_min)/(x_max - x_min) for tick in ax.get_xticks()]
    # ax.xaxis.set_major_locator(eval(locator))
    # plt.plot((x1, x2), (0.0, 1.0), 'k-')    
    plt.plot(train_loss)
    plt.xticks(np.arange(len(labels)), labels, rotation=75)
    plt.title('GAN Adversarial Training Loss')
    plt.show()



def discriminator_prediction(args):
    '''Plot the discriminator prediction over characters'''
    prob, labels = [], []

    with open(args.predict_file, 'rb') as f:
        labels = [r[0] for r in f.read().strip().split('\n')]
    with open(args.predict_file, 'rb') as f:
        prob   = [float(r[3:]) for r in f.read().strip().split('\n')]

    plt.plot(prob)
    plt.title('Probability of Real as Function of Text')
    plt.xticks(np.arange(len(labels)), labels) 
    plt.show()



if __name__=='__main__':
    args = parse_args()

    # plot_gan(args)
    # discriminator_prediction(args)