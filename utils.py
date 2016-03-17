


def load_reviews(file_dir, min_sequence_length=200):
	'''Loads list of reviews from file_dir'''
	with open(file_dir, 'rb') as f:
		reviews = [r[3:] for r in f.read().strip().split('\n')]
		reviews = [r.replace('\x05',  '') for r in reviews]
		reviews = [r.replace('<STR>', '') for r in reviews]
	reviews = [r for r in reviews if len(r) >= min_sequence_length]
	return reviews


def write_predictions_to_file(text, file_dir='data/sequences/predictions.txt'):
	'''Write predictions of real probabiltiy of sequence to file'''
	text = text.replace('<STR>','').replace('<EOS>','')

	prob = predict(text)[:, 0, 1].tolist()

	with open(file_dir, 'w') as f:
		for i in xrange(len(text)):
			print >> f, '%s, %f' % (text[i], prob[i])

