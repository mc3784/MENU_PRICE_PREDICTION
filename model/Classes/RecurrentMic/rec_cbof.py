import tensorflow as tf
import numpy as np
from sys import exit




class LSTM_CBOW(object):
    """
    A CBOF for text classification.
    Uses an embedding layer, followed by a hidden layer, and output layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, max_grad_norm, batch_size, dropout_keep_prob,
      embedding_size, n_hidden, n_layers, l2_reg_lambda=0.0):

        self.is_training = tf.placeholder(tf.bool, name="is_training")
        #l2_loss = tf.constant(0.0)
        # Placeholders for input, output and dropout (which you need to implement!!!!)

        #self.dropout_keep_prob  = tf.placeholder(tf.float32, name="dropout_keep_prob") 
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.dropout_keep_prob = dropout_keep_prob

        self.input_y = tf.placeholder(tf.int32, [batch_size, num_classes], name="input_y")

        # Keeping track of l2 regularization loss (optional)
        #l2_loss = tf.constant(0.0)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
        if self.is_training is not None:
          lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_cell, output_keep_prob=self.dropout_keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # Embedding layer
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable(
          "embedding", [vocab_size, embedding_size], dtype=tf.float32, trainable=True)
            embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)

            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        print("inputx: {}".format(self.input_x.get_shape()))

        print("embedded_chars: {}".format(embedded_chars.get_shape()))
        #self.embedded_chars = tf.reshape(self.embedded_chars, [1, batch_size, embedding_size])
        #print("embedded_chars2: {}".format(self.embedded_chars.get_shape()))

        #with tf.name_scope('dropout'):
        if self.is_training is not None and self.dropout_keep_prob <1:
            embedded_chars = tf.nn.dropout(embedded_chars, self.dropout_keep_prob, seed =1)

        outputs = []
        state = self._initial_state
        print("initialstate: {}".format(len(state)))
        print("initialstate: {}".format(state[0]))
        #Do it sentence by sentence in batch 
        with tf.variable_scope('RNN'):
            for ind in range(sequence_length):
                if ind > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(embedded_chars[:,ind, :], state)
                outputs.append(cell_output)

        print("output1: {}".format(len(outputs))) 
        print("output1: {}".format(outputs[0].get_shape()))         
        print("All variables")
        for v in tf.all_variables():
            print(v.name)

        outputs = tf.add_n(outputs)/sequence_length
        #print("outputred: {}".format(outputs.get_shape()))  
        #with tf.name_scope("output"):
        output = tf.reshape(tf.concat(1, outputs), [-1, n_hidden])
        print("output2: {}".format(output.get_shape()))   

        #output = outputs[0]
        softmax_w = tf.get_variable(
            "softmax_w", [n_hidden, num_classes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=tf.float32)
        self.logits = tf.matmul(output, softmax_w) + softmax_b
     
        print("logits: {}".format(self.logits.get_shape()))
        print("y: {}".format(tf.reshape(self.input_y, [-1]).get_shape()))
        print("w: {}".format(softmax_w.get_shape()))
        print("b: {}".format(softmax_b.get_shape()))


        loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y)
        print("loss: {}".format(loss))
        #loss = tf.nn.seq2seq.sequence_loss_by_example(
        #    [self.logits],
        #    [tf.reshape(self.input_y, [-1])],
        #    [tf.ones([batch_size*num_classes], dtype=tf.float32 )]
        #    )
        self.loss=  tf.reduce_mean(loss)
        print("loss: {}".format(loss))
        self._final_state = state 

        # Accuracy
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        print("acc: {}".format(self.accuracy))


        if self.is_training is None:
          return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr) 

        self._new_em= tf.placeholder(
            tf.float32, shape=[vocab_size, embedding_size], name="new_embedding")
        self._em_update = tf.assign(self.embedding, self._new_em) 

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
 
    def assign_em(self, session, em_value):
        session.run(self._em_update, feed_dict={self._new_em: em_value})               

    @property
    def input(self):
        return self.input_x

    @property
    def initial_state(self):
        return self._initial_state

    #@property
    #def loss(self):
    #    return self.loss

    #@property
    #def accuracy(self):
    #    return self.accuracy

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
