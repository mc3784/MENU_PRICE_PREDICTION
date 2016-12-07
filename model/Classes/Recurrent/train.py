#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from rec_cbof import LSTM_CBOW
from tensorflow.contrib import learn
from sys import exit
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Max. gradient allowed (default: 5.0)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Prob of drop out")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#tf.flags.DEFINE_boolean("use_fp16", False,
#                  "Train using 16-bit floats instead of 32bit floats")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

createFile = False
useBigram = False
splitPercentage_1 = 0.15
splitPercentage_2 = 0.3
timestamp = str(int(time.time()))
output_file = 'results.txt.' +timestamp

# Files Header 
with open(output_file, 'a') as out:
    out.write("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        out.write("{}={}".format(attr.upper(), value))
        out.write("\n")
    out.write("step,train_loss,train_acc,test_loss,test_acc"+ '\n')
loss_list=[]
earlyStopping = True
notImproving = 0
maxNotImprovingTimes = 4


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels()
print("Total number of samples: {}".format(len(x_text))) 
numberTestSamples_1 = int(splitPercentage_1*int(len(x_text)))
numberTestSamples_2 = int(splitPercentage_2*int(len(x_text)))
#print("Number of test samples: {}".format(numberTestSamples)) 

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
print("max_document_length:")
print(max_document_length) 
#max_document_length = 70
print("max_document_length: {} ".format(max_document_length)) 

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

x = np.array(x_text)
y = np.array(y)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))

x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]


# Split train/test/dev set
x_dev, x_test, x_train = x_shuffled[:numberTestSamples_1], x_shuffled[numberTestSamples_1:numberTestSamples_2], x_shuffled[numberTestSamples_2:]
y_dev, y_test, y_train = y_shuffled[:numberTestSamples_1], y_shuffled[numberTestSamples_1:numberTestSamples_2], y_shuffled[numberTestSamples_2:]

print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))

#print(x_train.tolist())
#exit() 
vocabulary = data_helpers.create_vocabulary(x_train.tolist(),max_document_length)




#vocabulary_file='vocabulary.txt.'+timestamp
#with open(vocabulary_file, 'w') as thefile:
#    for item in vocabulary:
#        thefile.write("%s\n" % item)


x_train = data_helpers.substitute_oov(x_train,vocabulary,max_document_length)
x_dev = data_helpers.substitute_oov(x_dev,vocabulary,max_document_length)



if useBigram:
    train_bigrammize = data_helpers.find_bigrams(x_train)
    bigram_voc = data_helpers.create_bigram_voc(train_bigrammize)
    train_bigrammize = data_helpers.substitute_bgr_oov(train_bigrammize,bigram_voc,max_document_length)
    x_train = [x_train[i]+" "+train_bigrammize[i] for i in range(len(x_train))]

    dev_bigrammize = data_helpers.find_bigrams(x_dev)
    dev_bigrammize = data_helpers.substitute_bgr_oov(dev_bigrammize,bigram_voc,max_document_length)
    x_dev = [x_dev[i]+" "+dev_bigrammize[i] for i in range(len(x_dev))]

#print [min(50,len(x.split(" "))) for x in x_text]
#exit()
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_dev = np.array(list(vocab_processor.transform(x_dev)))


#print x_train[0]

#exit()


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cbof = LSTM_CBOW(
            sequence_length=x_train.shape[1],
            num_classes=10,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            n_hidden=x_train.shape[1],
            max_grad_norm = FLAGS.max_grad_norm,
            n_layers = 1,
            #num_filters=FLAGS.num_filters,
            batch_size = FLAGS.batch_size,
            #use_fp16 = FLAGS.use_fp16,
            #dropout_keep_prob = FLAGS.dropout_keep_prob,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            #max_len_doc = max_document_length
            )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cbof.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
        print("OPT")
        print(train_op)
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cbof.loss)
        acc_summary = tf.scalar_summary("accuracy", cbof.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)


        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """

            fetches = {
                  "loss": cbof.loss,
                  "final_state": cbof.final_state,
              }
            start_time = time.time()
            state = sess.run(cbof.initial_state)
            #if eval_op is not None:

            feed_dict = {
                          cbof.input_x: x_batch,
                          cbof.input_y: y_batch,
                          cbof.is_training: True,
                          cbof.dropout_keep_prob: 1.}
            for i, (c, h) in enumerate(cbof.initial_state):
                  feed_dict[c] = state[i].c
                  feed_dict[h] = state[i].h


            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op,  cbof.loss, cbof.accuracy], 
                feed_dict, fetches)


            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #save value for plot

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                with open(output_file, 'a') as out:
                    out.write("{},{:g},{:g}".format(step, loss, accuracy) + ',')
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            global notImproving
            feed_dict = {
              cbof.input_x: x_batch,
              cbof.input_y: y_batch,
              cbof.is_training: False ,
              cbof.dropout_keep_prob: 1.,
            }
            fetches = {
                  "loss": cbof.loss,
                  "final_state": cbof.final_state,
              }
            state = sess.run(cbof.initial_state)


            for i, (c, h) in enumerate(cbof.initial_state):
                  feed_dict[c] = state[i].c
                  feed_dict[h] = state[i].h

            step, summaries, loss, accuracy = sess.run(
                [ global_step, dev_summary_op,  cbof.loss, cbof.accuracy], 
                feed_dict, fetches)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
	    #Save value for plot:
            with open(output_file, 'a') as out:
                out.write("{:g},{:g}".format(loss, accuracy) + '\n')

            #Early stopping condition
            if len(loss_list) > 0 and loss > loss_list[-1]:
               notImproving+=1 
               print("NOT IMPROVING FROM PREVIOUS STEP")
            else:
               notImproving = 0
            if earlyStopping and notImproving > maxNotImprovingTimes:
               print(loss_list)
               sess.close()
               exit()
            loss_list.append(loss) 

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)

            print("Original X len %i" %len(x_batch))
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation: notImproving: {}".format(notImproving))
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
                #print(loss_list)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
