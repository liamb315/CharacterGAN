


def load_reviews(file_dir, min_sequence_length=200):
	'''Loads list of reviews from file_dir'''
	with open(file_dir, 'rb') as f:
		reviews = [r[3:] for r in f.read().strip().split('\n')]
		reviews = [r.replace('\x05',  '') for r in reviews]
		reviews = [r.replace('<STR>', '') for r in reviews]
	reviews = [r for r in reviews if len(r) >= min_sequence_length]
	return reviews