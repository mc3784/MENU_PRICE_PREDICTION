#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers2 as data_helpers
from rec_cbof2 import LSTM_CBOW
from tensorflow.contrib import learn
from sys import exit
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Max. gradient allowed (default: 5.0)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Prob of drop out")
tf.flags.DEFINE_float("learning_rate", 1.0, "Learning Rate")
tf.flags.DEFINE_float("lr_decay", 0.5, "Learning Rate Decay")
tf.flags.DEFINE_float("init_scale", 0.1, "Initial Scale")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_steps", 1, "Num step Size (default: 1)")
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
splitPercentage_1 = 0.1
splitPercentage_2 = 0.2
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

def run_step(x_db, y_db, model, session, eval_op= None ):
    """
    A single training step
    """
    start_time = time.time()
    state = session.run(model.initial_state)
    costs = 0.0
    iters = 0
    fetches = {
          "loss": model.loss,
          "accuracy": model.accuracy,
          "final_state": model.final_state,
      }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.num_batches_per_epoch):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
              feed_dict[c] = state[i].c
              feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)

        loss = vals["loss"]
        state = vals["final_state"]
        accuracy = vals["accuracy"]

        costs += loss

        current_step = tf.train.global_step(session, sv.global_step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss, accuracy))
    #save value for plot

        if current_step % FLAGS.evaluate_every == 0:
            with open(output_file, 'a') as out:
                out.write("{},{:g},{:g}".format(current_step, loss, accuracy) + ',')
    #train_summary_writer.add_summary(summaries, step)

with tf.Graph().as_default():

    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                FLAGS.init_scale)
    with tf.name_scope("Train"):
        x_train_i, y_train_i, nbe =data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs, FLAGS.num_steps)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            cbof_train = LSTM_CBOW(
            input_x = x_train_i,
            input_y = y_train_i,
            num_batches_per_epoch= nbe,
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
            dropout_keep_prob = FLAGS.dropout_keep_prob,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            #max_len_doc = max_document_length
            )
        tf.scalar_summary("Training_Loss", cbof_train.loss)
        tf.scalar_summary("Training_Accuracy", cbof_train.accuracy)
        tf.scalar_summary("Learning_rate", cbof_train.lr)

    with tf.name_scope("Valid"):
        x_val_i, y_val_i, nbe_val =data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1, FLAGS.num_steps)
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            cbof_val= LSTM_CBOW(
            input_x = x_val_i,
            input_y = y_val_i,
            num_batches_per_epoch= nbe,
            sequence_length=x_train.shape[1],
            num_classes=10,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            n_hidden=x_train.shape[1],
            max_grad_norm = FLAGS.max_grad_norm,
            n_layers = 1,
            #num_filters=FLAGS.num_filters,
            batch_size = len(y_dev),
            #use_fp16 = FLAGS.use_fp16,
            dropout_keep_prob = 1.,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            #max_len_doc = max_document_length
            )
        tf.scalar_summary("loss_dev", cbof_val.loss)
        tf.scalar_summary("accuracy_dev", cbof_val.loss)

    # Write vocabulary
    #vocab_processor.save(os.path.join(out_dir, "vocab"))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    print("Writing to {}\n".format(out_dir))
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    sv = tf.train.Supervisor(logdir=train_summary_dir)

    with sv.managed_session() as sess:
        for i in FLAGS.num_epochs:
            lr_decay = FLAGS.lr_decay ** max(i- 4, 0.0)
            cbof_train.assign_lr(sess, FLAGS.learning_rate * lr_decay)

            run_step(x_batch, y_batch, cbof_train, sess, eval_op=cbof_train.train_op)
            current_step = tf.train.global_step(sess, sv.global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation: notImproving: {}".format(notImproving))
                run_step(x_dev, y_dev, cbof_val, sess)
            print("")
                #print(loss_list)
            if current_step % FLAGS.checkpoint_every == 0:
                path = sv.saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

