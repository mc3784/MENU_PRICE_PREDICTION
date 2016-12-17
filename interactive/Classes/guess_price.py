import tensorflow as tf
from text_cbof import TextCBOF
from tensorflow.contrib import learn
import numpy as np
import sys 

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Prob of drop out")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.app.flags.DEFINE_string("data_dir", "../../../data/", "Data directory")
FLAGS = tf.flags.FLAGS

def create_model(session,checkpoint_file):
    model = TextCBOF(
        sequence_length=185,
        num_classes=1,
    	#batch_size = FLAGS.batch_size,
        vocab_size=7614,#len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        n_hidden=64,
        #num_filters=FLAGS.num_filters,
        dropout_keep_prob = FLAGS.dropout_keep_prob,
        l2_reg_lambda=0.0
        )

    model.saver.restore(session, checkpoint_file)
    return model
#model_path="model/Classes/10_classes/runs/1481925866/"
model_path="/scratch/mc3784/Continuous/MLP/run-9314686/runs/1481686692"
vocab_processor = learn.preprocessing.VocabularyProcessor(185)
vocab = vocab_processor.restore(model_path+"/vocab")
#text=['Two Eggs with apple wood smoked bacon']
#text = np.array(list(vocab.transform(text)))
#print text

sys.stdout.write("> ")
sys.stdout.flush()
sentence = sys.stdin.readline()
with tf.Session() as sess:
	checkpoint_file = tf.train.latest_checkpoint(model_path+"/checkpoints")
	model = create_model(sess,checkpoint_file)
	while sentence:
		print checkpoint_file
		text = [str(sentence)]# ['Two Eggs with apple wood smoked bacon']
		print sentence
		text = np.array(list(vocab.transform(text)))
		print text
		x_batch = text
		feed_dict = {
		    model.input_x: text
		}
		predictions = sess.run([model.predictions],feed_dict)
		print predictions
		print "> ",
		sys.stdout.flush()
		sentence = sys.stdin.readline()
