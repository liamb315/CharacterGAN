import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops.nn import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn import seq2seq 
from tensorflow.contrib.distributions import Categorical
# TODO: Eliminate depencence on seq2seq


def variable_summaries(var, name):
    '''Attach a lot of summaries to a Tensor.'''
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


class GAN(object):
    def __init__(self, args):
        if args.model == 'rnn':
            cell_gen = rnn_cell.BasicRNNCell(args.rnn_size)
            cell_dis = rnn_cell.BasicRNNCell(args.rnn_size)
        elif args.model == 'gru':
            cell_gen = rnn_cell.GRUCell(args.rnn_size)
            cell_dis = rnn_cell.GRUCell(args.rnn_size)
        elif args.model == 'lstm':
            cell_gen = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)
            cell_dis = rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=False)
        else:
            raise NotImplementedError('Model type not supported: {}'
                                      .format(args.model))

        # TODO: Better initialization.
        indices = []
        batch_indices = tf.fill([args.batch_size], 0)
        
        # Targets for Generator are 1            
        self.targets = tf.ones([args.batch_size, args.seq_length], dtype=tf.int32)
        
        # Generator Portion of GAN.
        with tf.variable_scope('generator'):
            outputs_gen, logit_sequence = [], []
            cell_gen = rnn_cell.MultiRNNCell([cell_gen] * args.num_layers)
            self.initial_state_gen = cell_gen.zero_state(args.batch_size, tf.float32)
            state_gen = self.initial_state_gen

            with tf.variable_scope('rnn'):
                softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
                softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
                embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
                inp = tf.nn.embedding_lookup(embedding, batch_indices)
        
                for i in xrange(args.seq_length):
                    indices.append(batch_indices)
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                  
                    # RNN.
                    output_gen, state_gen = cell_gen(inp, state_gen)
                    logits_gen = tf.nn.xw_plus_b(output_gen, softmax_w, softmax_b)
                    
                    # Sampling.
                    sample_op = tf.stop_gradient(Categorical(
                                                logits_gen).sample(n=1))
                    batch_indices = tf.squeeze(sample_op)
                    inp = tf.nn.embedding_lookup(embedding, batch_indices)                
                    
                    # Use Only Logit Sampled.
                    one_hot = tf.stop_gradient(tf.one_hot(batch_indices,
                                                          depth = args.vocab_size,
                                                          dtype = tf.float32))
                    logit_gen = one_hot * logits_gen
                    logit_sequence.append(logit_gen)
                    outputs_gen.append(output_gen)

            self.final_state_gen = state_gen
            
            # Sampled indices
            self.sample_op = tf.pack(indices)
        
        # Discriminator Portion of GAN. 
        with tf.variable_scope('discriminator'):
            cell_dis = rnn_cell.MultiRNNCell([cell_dis] * args.num_layers)
            self.initial_state_dis = cell_dis.zero_state(args.batch_size, tf.float32)

            with tf.variable_scope('rnn'):
                softmax_w_dis = tf.get_variable('softmax_w', [args.rnn_size, 2])
                softmax_b_dis = tf.get_variable('softmax_b', [2])
                embedding_dis = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])            

                # Input sequence to Discriminator.
                inputs_dis = []
                for logit in logit_sequence:
                    inputs_dis.append(tf.matmul(logit, embedding_dis))

                # RNN.
                assert len(inputs_dis) == len(outputs_gen)
                outputs_dis, self.final_state_dis = seq2seq.rnn_decoder(inputs_dis,
                    self.initial_state_dis, cell_dis, loop_function=None)

                # Predictions.
                probs, logits = [], []
                for output_dis in outputs_dis:
                    logit = tf.nn.xw_plus_b(output_dis, softmax_w_dis, softmax_b_dis)
                    prob = tf.nn.softmax(logit)
                    logits.append(logit)
                    probs.append(prob)
                    
        with tf.name_scope('train'):
            gen_loss = seq2seq.sequence_loss_by_example(logits, 
                tf.unpack(tf.transpose(self.targets)), 
                tf.unpack(tf.transpose(tf.ones_like(self.targets, dtype=tf.float32))))

            self.gen_cost = tf.reduce_sum(gen_loss) / args.batch_size
            tf.scalar_summary('training loss', self.gen_cost)
            self.lr_gen = tf.Variable(0.0, trainable = False)
            tvars = tf.trainable_variables()
            gen_vars = [v for v in tvars if v.name.startswith("generator")]
            gen_grads            = tf.gradients(self.gen_cost, gen_vars)
            gen_grads_clipped, _ = tf.clip_by_global_norm(gen_grads, args.grad_clip)
            gen_optimizer        = tf.train.AdamOptimizer(self.lr_gen)
            self.gen_train_op    = gen_optimizer.apply_gradients(zip(gen_grads_clipped, gen_vars))

        with tf.name_scope('summary'):
            with tf.name_scope('weight_summary'):
                for v in tvars:
                    variable_summaries(v, v.op.name)
            with tf.name_scope('grad_summary'):
                all_grads = tf.gradients(self.gen_cost, tvars)
                for var, grad in zip(tvars, all_grads):
                    variable_summaries(grad, 'grad/' + var.op.name)

        self.merged = tf.merge_all_summaries()