import tensorflow as tf
import numpy as np
from sys import exit
from tensorflow.models.embedding import gen_word2vec as word2vec



class TextCBOF(object):
    """
    A CBOF for text classification.
    Uses an embedding layer, followed by a hidden layer, and output layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,embedding_type,
      embedding_size, n_hidden, dropout_keep_prob,l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout (which you need to implement!!!!)

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        #mask = tf.not_equal(self.input_x, 0, name=None)
#        print self.input_x
        #tf.boolean_mask(self.input_x , mask, name='boolean_mask')
  
        self.embedding_type = embedding_type

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #self.dropout_keep_prob  = tf.placeholder(tf.float32, name="dropout_keep_prob") 
        self.dropout_keep_prob = tf.constant(0.5,name="dropout_keep_prob") 
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if self.embedding_type:

                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="E",
                    trainable = False)
                #self.embedded_chars = tf.nn.embedding_lookup(E, self.input_x)
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            else:
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="E")
                #self.embedded_chars = tf.nn.embedding_lookup(E, self.input_x)
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)            

        with tf.name_scope('multilayer'):
            self.embedded_mean=tf.reduce_mean(self.embedded_chars ,1)

            #Store weights and bias
            W = tf.Variable(tf.truncated_normal([embedding_size,n_hidden], stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([n_hidden]), name="b")
            
            # Layer Layer
            self.layer_out = tf.nn.xw_plus_b( self.embedded_mean, W, b)
            self.layer_out = tf.nn.relu(self.layer_out,name="tanh")

        #Set the seed to 1 to ensure same results at each run
        with tf.name_scope("dropout"):
            self.h_dropout = tf.nn.dropout(self.layer_out , self.dropout_keep_prob, seed =1)
                            

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[n_hidden, num_classes],
                #initializer=tf.contrib.layers.variance_scaling_initializer())
                initializer=tf.contrib.layers.xavier_initializer())
                #variance_scaling_initializer()
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b( self.h_dropout, W, b, name="scores")
            print("self.scores: {}".format(self.scores.get_shape()))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))

            print(self.input_y.get_shape())
            print(self.predictions.get_shape())
            print(self.input_y) 
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.predicted_labels = self.predictions
            self.true_labels =  tf.argmax(self.input_y, 1)

