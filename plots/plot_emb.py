
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from sys import exit
from sklearn.manifold import TSNE
from tensorflow.contrib import learn
from sys import exit

from collections import Counter
import collections


def mostCommondWords(words):
    numberOfWords=100
    count = [('oov', -1)]
    count.extend(collections.Counter(words).most_common(numberOfWords - 1))
    frequentWords = set()
    for element in count:
        frequentWords.add(element[0]) 
    return frequentWords

x_text1 = list(open("/home/mc3784/NLP/MENU_PRICE_PREDICTION/data/menuItemsNYC.csv", "r").readlines())
x_text2 = list(open("/home/mc3784/NLP/MENU_PRICE_PREDICTION/data/menuItemsSF.csv", "r").readlines())
x_text = x_text1+x_text2
x_text = [s.strip() for s in x_text]

frequentWords = mostCommondWords(" ".join(x_text).split())




vocab_processor = learn.preprocessing.VocabularyProcessor(200)
vocab = vocab_processor.restore("/scratch/mc3784/MENU/run-9311213/runs/1481645517/vocab")
#/scratch/mc3784/Continuous/MLP/run-9317041/runs/1481732337/
vocabMap = vocab.vocabulary_._mapping

modelEmbedding = None
checkpoint_file = tf.train.latest_checkpoint("/scratch/mc3784/MENU/run-9311213/runs/1481645517/checkpoints")


print "{}.meta".format(checkpoint_file)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    for v in tf.all_variables():
#embedding/E:0
        print v.name
        print v
    	if v.name == "embedding/E:0":
    		embedding = v
    #Normalizing the plot is less clear:
    embedding = tf.nn.l2_normalize(embedding, dim=1)
    modelEmbedding = sess.run(embedding)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
embed2d = tsne.fit_transform(modelEmbedding)
n=[(k,v) for k,v in  vocabMap.items()]
fig, ax = plt.subplots()
for i, txt in enumerate(n):
        if txt[0] in frequentWords:
      	   ax.scatter(embed2d[txt[1],0], embed2d[txt[1],1])
	   ax.annotate(txt[0], (embed2d[txt[1],0],embed2d[txt[1],1]))
plt.savefig("word2vecNormalize")
