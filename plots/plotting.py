import matplotlib.pyplot as plt
import numpy as np
import re
from argparse import ArgumentParser

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--log_file', default='gan_log_current.txt',
        help='The training loss log file for the GAN training')
    return argparser.parse_args()



if __name__=='__main__':
    args = parse_args()

    with open(args.log_file, 'r') as f:
        next(f)
        train_loss = []
        labels     = []
        for i, line in enumerate(f):
            train_loss.append(float(re.search('\((.*?)\)', line).group(1)))
            if i % 50 == 0:
                labels.append(re.search('^\w+', line).group())
            else:
                labels.append('')


    plt.plot(train_loss)
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.title('GAN Adversarial Training Loss')
    plt.show()


