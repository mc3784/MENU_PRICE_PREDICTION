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
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Max. gradient allowed (default: 5.0)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Prob of drop out")
tf.flags.DEFINE_float("learning_rate", 1.0, "Learning Rate")
tf.flags.DEFINE_float("lr_decay", 0.5, "Learning Rate Decay")
tf.flags.DEFINE_float("init_scale", 0.1, "Initial Scale")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.app.flags.DEFINE_string("data_dir", "../../../data/", "Data directory")
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
x_text, y = data_helpers.load_data_and_labels(FLAGS.data_dir)
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
x_test = data_helpers.substitute_oov(x_test,vocabulary,max_document_length)


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
x_test = np.array(list(vocab_processor.transform(x_test)))

#print x_train[0]

#exit()


# Training
# ==================================================

with tf.Graph().as_default():

    #init_op = tf.initialize_all_variables()
    #se1 = tf.InteractiveSession()
    #se1.run([init_op])
    #se1.close()
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                FLAGS.init_scale)
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            cbof_train = LSTM_CBOW(
            sequence_length=x_train.shape[1],
            num_classes=1,
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
        #tf.scalar_summary("Training_Loss", cbof_train.loss)
        #tf.scalar_summary("Training_Accuracy", cbof_train.accuracy)
        #tf.scalar_summary("Learning_rate", cbof_train.lr)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            cbof_val= LSTM_CBOW(
            sequence_length=x_train.shape[1],
            num_classes=1,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            n_hidden=x_train.shape[1],
            max_grad_norm = FLAGS.max_grad_norm,
            n_layers = 1,
            #num_filters=FLAGS.num_filters,
            batch_size = FLAGS.batch_size,
            #use_fp16 = FLAGS.use_fp16,
            dropout_keep_prob = 1.,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            #max_len_doc = max_document_length
            )
        #tf.scalar_summary("loss_dev", cbof_val.loss)
        #tf.scalar_summary("accuracy_dev", cbof_val.loss)

    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            cbof_test= LSTM_CBOW(
            sequence_length=x_train.shape[1],
            num_classes=1,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            n_hidden=x_train.shape[1],
            max_grad_norm = FLAGS.max_grad_norm,
            n_layers = 1,
            #num_filters=FLAGS.num_filters,
            batch_size = FLAGS.batch_size,
            #use_fp16 = FLAGS.use_fp16,
            dropout_keep_prob = 1.,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            #max_len_doc = max_document_length
            )


    def train_step(x_db, y_db, model, session, eval_op= None ):
        """
        A single training step
        """
        if len(x_db)==FLAGS.batch_size:
            start_time = time.time()
            state = session.run(model.initial_state)

            fetches = {
                  "loss": model.loss,
                  "accuracy": model.accuracy,
                  "final_state": model.final_state,
              }

            if eval_op is not None:
                fetches["eval_op"] = eval_op

            feed_dict = {
                          model.input_x: x_db,
                          model.input_y: y_db,
                          model.is_training: True
                        }

            for i, (c, h) in enumerate(model.initial_state):
                  feed_dict[c] = state[i].c
                  feed_dict[h] = state[i].h

            vals = session.run(fetches, feed_dict)

            loss = vals["loss"]
            state = vals["final_state"]
            accuracy = vals["accuracy"]


            #_, step, summaries, loss, accuracy = sess.run(
            #    [train_op, global_step, train_summary_op,  cbof.loss, cbof.accuracy], 
            #    feed_dict, fetches)

            current_step = tf.train.global_step(session, sv.global_step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: Train step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss, accuracy))
            #save value for plot


            if current_step % FLAGS.evaluate_every == 0:
                with open(output_file, 'a') as out:
                    out.write("{},{:g},{:g}".format(current_step, loss, accuracy) + ',')
        #train_summary_writer.add_summary(summaries, step)

    def dev_step(x_tot, y_tot, model, model_2, session, writer=None):
        """
        Evaluates model on a dev set
        """
        global notImproving
        start_time = time.time()
        state = session.run(model.initial_state)

        loss = 0.0
        accuracy = 0.0
        fetches = {
              "loss": model.loss,
              "accuracy": model.accuracy,
              "final_state": model.final_state,

          }

        count= 0
        ba_dev = data_helpers.batch_iter(list(zip(x_tot, y_tot)), FLAGS.batch_size, 1)
        print("Dev split created")
        for batch in ba_dev:

            x_batch, y_batch = zip(*batch)
            if len(x_batch)==FLAGS.batch_size:
                count= count+1
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.is_training: False
                }
                for i, (c, h) in enumerate(model.initial_state):
                      feed_dict[c] = state[i].c
                      feed_dict[h] = state[i].h

                vals = session.run(fetches, feed_dict)
                loss = loss + vals["loss"]
                state = vals["final_state"]
                accuracy = accuracy+ vals["accuracy"]
                #print(loss, accuracy)

        loss = loss*1./count
        accuracy = accuracy*1./count

        #step, summaries, loss, accuracy = sess.run(
        #    [ global_step, dev_summary_op,  cbof.loss, cbof.accuracy], 
        #    feed_dict, fetches)
        current_step = tf.train.global_step(session, sv.global_step)
        print(current_step)
        time_str = datetime.datetime.now().isoformat()
        print("Evaluation Results")
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss, accuracy))
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
           test_evaluation(x_test,y_test, model_2, session)
           sess.close()
           exit()
        loss_list.append(loss) 

    def test_evaluation(x_test, y_test,model,session):
        """
        Evaluates model on a test set
        """
        state = session.run(model.initial_state)

        loss = 0.0
        accuracy = 0.0
        fetches = {
              "loss": model.loss,
              "accuracy": model.accuracy,
              "final_state": model.final_state,

          }
        count= 0
        ba_test = data_helpers.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1)
        print("Dev split created")
        for batch in ba_test:
            x_batch, y_batch = zip(*batch)
            if len(x_batch)==FLAGS.batch_size:
                count= count+1
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.is_training: False
                }
                for i, (c, h) in enumerate(model.initial_state):
                      feed_dict[c] = state[i].c
                      feed_dict[h] = state[i].h

                vals = session.run(fetches, feed_dict)
                loss = loss + vals["loss"]
                state = vals["final_state"]
                accuracy = accuracy+ vals["accuracy"]
                #print(loss, accuracy)

        loss = loss*1./count
        accuracy = accuracy*1./count

        print("test loss: {}".format(loss))
        print("test accuracy: {}".format(accuracy))
        with open(output_file, 'a') as out:
            out.write("\nEvaluation on test set of size {}\n Loss, Accuracy\n".format(len(y_test)))
            out.write("{:g},{:g}".format(loss, accuracy) + '\n')
    # Generate batches
    # Initialize all variables

    # Output directory for models and summaries


    # Write vocabulary
    #vocab_processor.save(os.path.join(out_dir, "vocab"))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    print("Writing to {}\n".format(out_dir))
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    sv = tf.train.Supervisor(logdir=checkpoint_prefix)

    batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    with sv.managed_session() as sess:
        i=1
        c=0
        for batch in batches:
            c = c+len(batch)
            if c==len(x_train): 
                c+0
                i=i+1

            x_batch, y_batch = zip(*batch)
            lr_decay = FLAGS.lr_decay ** max(i - FLAGS.num_epochs, 0.0)
            cbof_train.assign_lr(sess, FLAGS.learning_rate * lr_decay)

            train_step(x_batch, y_batch, cbof_train, sess, eval_op=cbof_train.train_op)
            current_step = tf.train.global_step(sess, sv.global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation: notImproving: {}".format(notImproving))
                dev_step(x_dev, y_dev, cbof_val, cbof_test,sess)
            print("")
                #print(loss_list)
            if current_step % FLAGS.checkpoint_every == 0:
                sv.saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint ")

