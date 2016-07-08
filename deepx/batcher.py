import numpy as np

class WindowedBatcher(object):

    def __init__(self, sequences, encodings, target, batch_size=100, sequence_length=50):
        self.sequences = sequences

        self.pre_vector_sizes = [c.seq[0].shape[0] for c in self.sequences] + [target.seq[0].shape[0]]
        self.pre_vector_size = sum(self.pre_vector_sizes)
        self.target_size = target.seq[0].shape[0]

        self.encodings = encodings
        self.vocab_sizes = [c.index for c in self.encodings] + [self.target_size]
        self.vocab_size = sum(self.vocab_sizes)
        self.batch_index = 0
        self.batches = []
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.length = len(self.sequences[0])

        self.batch_index = 0
        self.X = np.zeros((self.length, self.pre_vector_size), dtype=np.int32)
        self.X = np.hstack([c.seq for c in self.sequences] + [target.seq])

        N, D = self.X.shape
        assert N > self.batch_size * self.sequence_length, "File has to be at least %u characters" % (self.batch_size * self.sequence_length)

        self.X = self.X[:N - N % (self.batch_size * self.sequence_length)]
        self.N, self.D = self.X.shape
        self.X = self.X.reshape((self.N / self.sequence_length, self.sequence_length, self.D))

        self.N, self.S, self.D = self.X.shape

        self.num_sequences = self.N / self.sequence_length
        self.num_batches = self.N / self.batch_size
        self.batch_cache = {}

    def next_batch(self):
        idx = (self.batch_index * self.batch_size)
        if self.batch_index >= self.num_batches:
            self.batch_index = 0
            idx = 0

        if self.batch_index in self.batch_cache:
            batch = self.batch_cache[self.batch_index]
            self.batch_index += 1
            return batch

        X = self.X[idx:idx + self.batch_size]
        y = np.zeros((X.shape[0], self.sequence_length, self.vocab_size))
        for i in xrange(self.batch_size):
            for c in xrange(self.sequence_length):
                seq_splits = np.split(X[i, c], np.cumsum(self.pre_vector_sizes))
                vec = np.concatenate([e.convert_representation(split) for
                                      e, split in zip(self.encodings, seq_splits)] + [X[i, c, -self.target_size:]])
                y[i, c] = vec

        X = y[:, :, :-self.target_size]
        y = y[:, :, -self.target_size:]

        X = np.swapaxes(X, 0, 1)
        y = np.swapaxes(y, 0, 1)
        # self.batch_cache[self.batch_index] = X, y
        self.batch_index += 1
        return X, y
