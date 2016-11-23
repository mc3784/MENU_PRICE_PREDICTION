import tensorflow as tf
import numpy as np
from sys import exit

class Recurrent_CBOW(object):
    """
    A CBOF for text classification.
    Uses an embedding layer, followed by a hidden layer, and output layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, n_hidden, dropout_keep_prob,l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout (which you need to implement!!!!)

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        #mask = tf.not_equal(self.input_x, 0, name=None)
#        print self.input_x
        #tf.boolean_mask(self.input_x , mask, name='boolean_mask')
  

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #self.dropout_keep_prob  = tf.placeholder(tf.float32, name="dropout_keep_prob") 
        self.dropout_keep_prob = tf.constant(0.5,name="dropout_keep_prob") 
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
          lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            E = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="E")
            self.embedded_chars = tf.nn.embedding_lookup(E, self.input_x)

            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print("embedded_chars: {}".format(self.embedded_chars.get_shape()))

        with tf.name_scope('dropout'):
            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.name_scope('LSTM'):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        with tf.name_scope("output"):
            output = tf.reshape(tf.concat(1, outputs), [-1, size])
            softmax_w = tf.get_variable(
                "softmax_w", [size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
            self.logits = tf.matmul(output, softmax_w) + softmax_b
            loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(input_.targets, [-1])],
                [tf.ones([batch_size * num_steps], dtype=data_type())])
            self.loss=  tf.reduce_sum(loss) / batch_size
            self._final_state = state   

            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr) 
                

        # Accuracy
        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    @property
    def input(self):
        return self.input_x

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def loss(self):
        return self.loss

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

