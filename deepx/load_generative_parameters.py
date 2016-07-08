from deepx.nn import *
from deepx.rnn import *
import cPickle as pickle

def convert_params(params):
    new_params = {}
    for param, value in params.items():
        new_params["%s-0" % param] = value.tolist()
    return new_params

if __name__ == "__main__":
    with open('data/charnet-top_2-1024-2.pkl', 'rb') as fp:
        generative_params = pickle.load(fp)
    lstm1 = convert_params(generative_params['lstm']['input_layer']['parameters'])
    lstm2 = convert_params(generative_params['lstm']['layers'][0]['parameters'])
    softmax = generative_params['output']['parameters']
    
    new_state = (({}, (lstm1, lstm2)), softmax)
    with open('data/generative-model-original.pkl', 'wb') as fp:
        pickle.dump(new_state, fp)

