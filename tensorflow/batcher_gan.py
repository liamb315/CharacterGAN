import numpy as np
import os
import collections
import cPickle
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class DiscriminatorBatcher(object):
	def __init__(self, real_file, fake_file, data_dir, vocab_file, batch_size, seq_length):
		self.batch_size = batch_size
		self.seq_length = seq_length

		real_file    = os.path.join(data_dir, real_file)
		fake_file    = os.path.join(data_dir, fake_file)
		real_tensor  = os.path.join(data_dir, 'real_data.npy')
		fake_tensor  = os.path.join(data_dir, 'fake_data.npy')
		vocab_fiel   = os.path.join(data_dir, vocab_file)

		# if not (os.path.exists(vocab_file) and os.path.exists(real_tensor) and os.path.exists(fake_tensor)):
		# 	self.preprocess(real_file, fake_file, vocab_file, real_tensor, fake_tensor)
		# else:
		# 	self.load_preprocessed(vocab_file, real_tensor, fake_tensor)

		self.preprocess(real_file, fake_file, vocab_file, real_tensor, fake_tensor)

		self.create_batches()
		self.reset_batch_pointer()

	def preprocess(self, real_file, fake_file, vocab_file, tensor_file_real, tensor_file_fake):
		logging.debug('Preprocessing...')
		with open(real_file, 'r') as f:
			data_real = f.read()
		with open(fake_file, 'r') as f:
			data_fake = f.read()		
		data = data_real + data_fake
		counter = collections.Counter(data)
		count_pairs = sorted(counter.items(), key=lambda x: -x[1])
		self.chars, _ = list(zip(*count_pairs))
		self.vocab_size = len(self.chars)
		self.vocab      = dict(zip(self.chars, range(len(self.chars))))
		with open(vocab_file, 'w') as f:
			cPickle.dump(self.chars, f)

		def build_tensor(tensor_file, data_str):
			tensor = np.array(map(self.vocab.get, data_str))
			np.save(tensor_file, tensor)
			return tensor

		self.tensor_real = build_tensor(tensor_file_real, data_real)
		self.tensor_fake = build_tensor(tensor_file_fake, data_fake)
		np.save(tensor_file_real, self.tensor_real)
		np.save(tensor_file_fake, self.tensor_fake)

	# def load_preprocessed(self, vocab_file, tensor_file_real, tensor_file_fake):
	# 	logging.debug('Loading preprocessed files...')
	# 	with open(vocab_file, 'r') as f:
	# 		self.chars = cPickle.load(f)
	# 	self.vocab_size  = len(self.chars)
	# 	self.vocab       = dict(zip(self.chars, range(len(self.chars))))
	# 	self.tensor_real = np.load(tensor_file_real)
	# 	self.tensor_fake = np.load(tensor_file_fake)	

	def create_batches(self):
		logging.debug('Creating batches...')
		
		# Real batches
		num_batches      = self.tensor_real.size / (self.batch_size / 2 * self.seq_length) 
		self.tensor_real = self.tensor_real[:num_batches * self.batch_size / 2 * self.seq_length]
		x_data_real      = self.tensor_real
		y_data_real      = np.ones((len(x_data_real), 1))
		x_batches_real   = np.split(x_data_real.reshape(self.batch_size / 2, -1), num_batches, 1)	
		y_batches_real   = np.split(y_data_real.reshape(self.batch_size / 2, -1), num_batches, 1)		
		batches_real     = [np.hstack([x, y]) for x, y in zip(x_batches_real, y_batches_real)]

		# Fake batches
		num_batches      = self.tensor_fake.size / (self.batch_size / 2 * self.seq_length)
		self.tensor_fake = self.tensor_fake[:num_batches * self.batch_size / 2 * self.seq_length]
		x_data_fake      = self.tensor_fake
		y_data_fake      = np.zeros((len(x_data_fake), 1))
		x_batches_fake   = np.split(x_data_fake.reshape(self.batch_size / 2, -1), num_batches, 1)	
		y_batches_fake   = np.split(y_data_fake.reshape(self.batch_size / 2, -1), num_batches, 1)		
		batches_fake     = [np.hstack([x, y]) for x, y in zip(x_batches_fake, y_batches_fake)]

		# Combine batches 
		batches          = [np.vstack((real, fake)) for real, fake in zip(batches_real, batches_fake)]
		for arr in batches:
			np.random.shuffle(arr)
		self.x_batches   = [arr[:, :self.seq_length] for arr in batches]
		self.y_batches   = [arr[:, self.seq_length:] for arr in batches]
		self.num_batches = len(batches)
	
	def next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1 
		return x, y

	def reset_batch_pointer(self):
		self.pointer = 0


class GANBatcher(object):
	def __init__(self, input_file, vocab_file, data_dir, batch_size, seq_length):
		self.batch_size = batch_size
		self.seq_length = seq_length

		input_file  = os.path.join(data_dir, input_file)		
		vocab_file  = os.path.join(data_dir, vocab_file)
		tensor_file = os.path.join(data_dir, 'simple_data.npy')

		if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
			self.preprocess(input_file, vocab_file, tensor_file)
		else:
			self.load_preprocessed(vocab_file, tensor_file)

		self.create_batches()
		self.reset_batch_pointer()

	def preprocess(self, input_file, vocab_file, tensor_file):
		logging.debug('Reading text file...')
		with open(input_file, 'r') as f:
			data = f.read()
		counter         = collections.Counter(data)
		count_pairs     = sorted(counter.items(), key=lambda x: -x[1])
		self.chars, _   = list(zip(*count_pairs))
		self.vocab_size = len(self.chars)
		self.vocab      = dict(zip(self.chars, range(len(self.chars))))
		with open(vocab_file, 'w') as f:
			cPickle.dump(self.chars, f)
		self.tensor     = np.array(map(self.vocab.get, data))
		np.save(tensor_file, self.tensor)

	def load_preprocessed(self, vocab_file, tensor_file):
		logging.debug('Loading preprocessed files...')
		with open(vocab_file, 'r') as f:
			self.chars = cPickle.load(f)
		self.vocab_size  = len(self.chars)
		self.vocab       = dict(zip(self.chars, range(len(self.chars))))
		self.tensor      = np.load(tensor_file)

	def create_batches(self):
		logging.debug('Creating batches...')
		self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)
		self.tensor      = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
		x_data           = self.tensor
		y_data           = np.copy(self.tensor)	
		y_data[:-1]      = x_data[1:] # Labels are simply the next char
		y_data[-1]       = x_data[0]
		self.x_batches   = np.split(x_data.reshape(self.batch_size, -1), self.num_batches, 1)
		self.y_batches   = np.split(y_data.reshape(self.batch_size, -1), self.num_batches, 1)

	def next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1 
		return x, y

	def reset_batch_pointer(self):
		self.pointer = 0