import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops.nn import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn import seq2seq 
from tensorflow.contrib.distributions import Categorical


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
    def __init__(self, args, global_step_tensor, train_method):
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

        if train_method == 'train_gen':
            indices = []
            index = tf.fill([args.batch_size], 0)
            
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
                    input_gen = tf.nn.embedding_lookup(embedding, index)
            
                    for i in xrange(args.seq_length):
                        indices.append(index)
                        if i > 0:
                            tf.get_variable_scope().reuse_variables()
                      
                        output_gen, state_gen = cell_gen(input_gen, state_gen)
                        logits_gen = tf.nn.xw_plus_b(output_gen, softmax_w, softmax_b)
                        log_probs_gen = tf.nn.log_softmax(logits_gen)

                        # Sampling.
                        index = tf.contrib.bayesflow.stochastic_graph.DistributionTensor(
                            tf.contrib.distributions.Categorical, logits = log_probs_gen)
                        one_hot = tf.one_hot(index, args.vocab_size, dtype = tf.float32)
                        
                        logit_sequence.append(one_hot)
                        outputs_gen.append(output_gen)

                self.final_state_gen = state_gen
            
                # Sampled indices
                self.sample_op = tf.pack(indices)
          
        elif train_method == 'train_dis':
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length]) # Target replication
      
        else:
            raise Exception('train method not supported: {}'.format(train_method))

        # Discriminator Portion of GAN. 
        with tf.variable_scope('discriminator'):
            cell_dis = rnn_cell.MultiRNNCell([cell_dis] * args.num_layers)
            self.initial_state_dis = cell_dis.zero_state(args.batch_size, tf.float32)

            with tf.variable_scope('rnn'):
                softmax_w_dis = tf.get_variable('softmax_w', [args.rnn_size, 2])
                softmax_b_dis = tf.get_variable('softmax_b', [2])
                embedding_dis = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])            

                if train_method == 'train_gen':
                    # Input sequence to Discriminator.
                    inputs_dis = []
                    for logit in logit_sequence:
                        inputs_dis.append(tf.matmul(logit, embedding_dis))
                    assert len(inputs_dis) == len(outputs_gen)

                elif train_method == 'train_dis':
                    # Input sequence to Discriminator.
                    inputs = tf.split(1, 
                        args.seq_length, 
                        tf.nn.embedding_lookup(embedding_dis, self.input_data))
                    inputs_dis = [tf.squeeze(i, [1]) for i in inputs]

                else:
                    raise Exception('train method not supported: {}'.format(train_method))

                # RNN.
                outputs_dis, self.final_state_dis = seq2seq.rnn_decoder(inputs_dis,
                    self.initial_state_dis, cell_dis, loop_function=None)

                # Predictions.
                predictions, logits_dis = [], []
                for output_dis in outputs_dis:
                    logit_dis = tf.nn.xw_plus_b(output_dis, softmax_w_dis, softmax_b_dis)
                    prob_dis = tf.nn.softmax(logit_dis)
                    logits_dis.append(logit_dis)
                    predictions.append(prob_dis)
     
        with tf.name_scope('train'):
            tvars = tf.trainable_variables()

            if train_method == 'train_gen':         
                rewards = []
                for pred in predictions:
                    rewards.append(pred[:, 1])
                
                # Exponential baseline.
                ema = tf.train.ExponentialMovingAverage(decay = args.baseline_decay)
                rewards_tf = tf.pack(rewards)
                reduced_reward = tf.reduce_mean(rewards_tf)
                maintain_avg_op = ema.apply([reduced_reward])
                baseline = ema.average(reduced_reward)
                
                # Advantage.
                loss = []
                for reward in rewards:
                    advantage = reward - baseline
                    loss.append(-advantage)

                # Optimizer 
                self.lr_gen = tf.Variable(0.0, trainable = False)
                final_loss = tf.contrib.bayesflow.stochastic_graph.surrogate_loss(loss)
                self.gen_vars = [v for v in tvars if v.name.startswith('gan/generator')]
                gen_optimizer = tf.train.AdamOptimizer(self.lr_gen)
                min_op = gen_optimizer.minimize(final_loss, var_list = self.gen_vars)
                
                # Group operations.
                self.gen_train_op = tf.group(min_op, maintain_avg_op)
                self.baseline = baseline
                self.advantage = advantage
                self.gen_cost = final_loss

            elif train_method == 'train_dis':
                loss = seq2seq.sequence_loss_by_example(logits_dis, 
                    tf.unpack(tf.transpose(self.targets)), 
                    tf.unpack(tf.transpose(tf.ones_like(self.targets, dtype = tf.float32))))
                self.cost = tf.reduce_sum(loss) / args.batch_size
                tf.scalar_summary('gen training loss', self.cost)
                
                # Optimizer
                self.lr_dis = tf.Variable(0.0, trainable = False)
                self.dis_vars = [v for v in tvars if v.name.startswith("gan/discriminator")]
                self.dis_grads       = tf.gradients(self.cost, self.dis_vars)
                dis_grads_clipped, _ = tf.clip_by_global_norm(self.dis_grads, args.grad_clip)
                dis_optimizer        = tf.train.AdamOptimizer(self.lr_dis)
                self.dis_train_op = dis_optimizer.apply_gradients(zip(dis_grads_clipped, self.dis_vars))
                
        
            else:
                raise Exception('train method not supported: {}'.format(train_method))

        self.merged = tf.merge_all_summaries()            
              