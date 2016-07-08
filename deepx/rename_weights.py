from argparse import ArgumentParser
import cPickle as pickle

def convert_params(params):
   if isinstance(params, tuple):
       return (convert_params(params[0]), convert_params(params[1]))
   new_params = {}
   for param, value in params.items():
       if param[-1].isdigit():
           new_params[param[:-2]] = value
       else:
           new_params["%s" % param] = value
   return new_params

if __name__ == "__main__":
   argparser = ArgumentParser()
   argparser.add_argument('weights')
   argparser.add_argument('out')

   args = argparser.parse_args()

   with open(args.weights) as fp:
       weights = pickle.load(fp)
   with open(args.out, 'w') as fp:
       pickle.dump(convert_params(weights), fp)