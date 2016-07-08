import tensorflow as tf
import numpy as np
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops.nn import rnn
from tensorflow.python.ops.nn import seq2seq


class Generator(object):
    def __init__(self, args, is_training=True, batch=True):
        self.args = args

        if not is_training:    
            args.seq_length = 1

        if not batch:
            args.batch_size = 1

        if args.model == 'rnn':
            cell = rnn_cell.BasicRNNCell(args.rnn_size)
        elif args.model == 'gru':
            cell = rnn_cell.GRUCell(args.rnn_size)
        elif args.model == 'lstm':
            cell = rnn_cell.BasicLSTMCell(args.rnn_size)
        else:
            raise Exception("model type not supported: {}".format(args.model))

        self.cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data     = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets        = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state  = self.cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnn'):
            softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
            
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
                inputs    = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs    = [tf.squeeze(i, [1]) for i in inputs]

        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, 
            self.cell, loop_function=None if is_training else loop, scope='rnn')
        output      = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs  = tf.nn.softmax(self.logits)
        loss        = seq2seq.sequence_loss_by_example([self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([args.batch_size * args.seq_length])],
            args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr       = tf.Variable(0.0, trainable = False)
        tvars         = tf.trainable_variables()
        grads, _      = tf.clip_by_global_norm(tf.gradients(self.cost, tvars, aggregation_method=2), args.grad_clip)
        optimizer     = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def generate(self, sess, chars, vocab, seq_length = 200, initial=' '):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in initial[:-1]:
            x       = np.zeros((1,1))
            x[0,0]  = vocab[char]
            feed    = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        sequence = initial
        char = initial[-1]
        for n in xrange(seq_length):
            x = np.zeros((1,1))
            x[0,0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p  = probs[0]
            sample = int(np.random.choice(len(p), p=p))
            pred = chars[sample]
            sequence += pred
            char = pred
        return sequence

    def generate_probabilities(self, sess, chars, vocab, seq_length = 200, initial = ' '):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in initial[:-1]:
            x       = np.zeros((1,1))
            x[0,0]  = vocab[char]
            feed    = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        probability_sequence = []
        char = initial[-1]
        for n in xrange(seq_length):
            x = np.zeros((1,1))
            x[0,0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p  = probs[0]
            probability_sequence.append(p)
            sample = int(np.random.choice(len(p), p=p))
            pred = chars[sample]
            char = pred
        return probability_sequence

    def sample_logits(self, logits, temperature=1.0):
        ''' This function is like sample_with_temperature except it can handle
         batch input a of [batch_size x logits]  this function takes logits 
         input, and produces a specific number from the array. 

        args: 
            logits:  2d array [batch_size x logits]
            temperature: hyperparameter to control variance of sampling
        
        returns:
            sequence_matrix: 1d array [batch_size] of selected numbers 
                             from distribution
        '''
        exponent_raised = tf.exp(tf.div(logits, temperature)) 
        matrix_X = tf.div(exponent_raised, tf.reduce_sum(exponent_raised, 
            reduction_indices = 1, keep_dims = True)) 
        matrix_U = tf.random_uniform(logits.get_shape(), minval = 0, maxval = 1)
        # You want dimension = 1 because you are argmaxing across rows.
        final_number = tf.argmax(tf.sub(matrix_X, matrix_U), dimension = 1) 
        return final_number

    def sample_probs(self, probs):
        ''' This function takes probs input, [batch_size x probs] and 
            produces a specific number from the array. 

        args: 
            probs:  2d array [batch_size x probs]
            temperature: hyperparameter to control variance of sampling
        
        returns:
            sequence_matrix: 1d array [batch_size] of selected numbers 
                             from distribution
        '''
        matrix_U = tf.random_uniform(probs.get_shape(), minval = 0, maxval = 1)
        # You want dimension = 1 because you are argmaxing across rows.
        final_number = tf.argmax(tf.sub(probs, matrix_U), dimension = 1) 
        return final_number


    def generate_batch(self, sess, args, chars, vocab, seq_length = 200, initial = ' '):
        ''' Generate a batch of reviews entirely within TensorFlow'''

        state = self.cell.zero_state(args.batch_size, tf.float32).eval()

        sequence_matrix = []
        for i in xrange(args.batch_size):
            sequence_matrix.append([])
        char_arr = args.batch_size * [initial]
        
        # TF implementation (doesn't quite work):
        # probs_tf  = tf.placeholder(tf.float32, [args.batch_size, args.vocab_size])
        # sample_op = self.sample_probs(probs_tf)

        for n in xrange(seq_length):
            x = np.zeros((args.batch_size, 1))
            for i, char in enumerate(char_arr):
                x[i,0] = vocab[char]    
            feed = {self.input_data: x, self.initial_state: state} 
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            # TF implementation (doesn't quite work):
            # probs_feed = {probs_tf: probs}
            # [sample_indexes] = sess.run([sample_op], probs_feed)
            
            # Numpy implementation:
            sample_indexes = [int(np.random.choice(len(p), p=p)) for p in probs]
            print len(sample_indexes)
            char_arr = [chars[i] for i in sample_indexes]
            for i, char in enumerate(char_arr):
                sequence_matrix[i].append(char)
        return sequence_matrix