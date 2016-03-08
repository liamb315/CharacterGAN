import matplotlib.pyplot as plt
import numpy as np
import re
from argparse import ArgumentParser

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('log_file', default='gan_log_current.txt',
        help='The training loss log file for the GAN training')
    return argparser.parse_args()



if __name__=='__main__':
    args = parse_args()

    with open(args.log_file, 'r') as f:
        next(f)
        train_loss = []
        labels     = []
        prev_label  = None

        for i, line in enumerate(f):
            train_loss.append(float(re.search('\((.*?)\)', line).group(1)))
            label = re.search('^\w+', line).group()

            if i == 0 or label != prev_label:
                labels.append(label)
                
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


