import numpy as np
import re
import itertools
from collections import Counter
import glob
import collections
import pickle
from sys import exit
import csv
from sys import exit

from nltk.corpus import stopwords



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"<br />", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = [word for word in string if word not in stopwords.words('english')]
    return string.strip().lower()


def load_data_and_labels(data_dir):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    with open(data_dir+'/menuItemsNYC.csv', 'rb') as f:
        next(f, None)
        reader = csv.reader(f)
        fileList1 = list(reader)
    with open(data_dir+'/menuItemsSF.csv', 'rb') as f:
        next(f, None)
        reader = csv.reader(f)
        fileList2 = list(reader)
    fileList = fileList1 + fileList2


    
    x = []
    y = []
    for l in fileList:
        prices = l[4].split("$")[1:]
        try:
            if len(prices)==1:
                if prices[0]<>"":
                    price = float(prices[0].strip(" - "))
                else:
                    price = 0
            else:
                price = float(prices[1])/2. if prices[0].strip(" - ")=="" else (float(prices[0].strip(" - "))+float(prices[1]))/2
            # if price<=5:
            #     priceClass = [1,0,0,0,0,0,0,0,0,0]
            # elif price<=10:
            #     priceClass = [0,1,0,0,0,0,0,0,0,0]
            # elif price<=15:
            #     priceClass = [0,0,1,0,0,0,0,0,0,0]
            # elif price<=20:
            #     priceClass = [0,0,0,1,0,0,0,0,0,0]
            # elif price<=25:
            #     priceClass = [0,0,0,0,1,0,0,0,0,0]
            # elif price<=30:
            #     priceClass = [0,0,0,0,0,1,0,0,0,0]
            # elif price<=40:
            #     priceClass = [0,0,0,0,0,0,1,0,0,0]
            # elif price<=50:
            #     priceClass = [0,0,0,0,0,0,0,1,0,0]
            # elif price<=70:
            #     priceClass = [0,0,0,0,0,0,0,0,1,0]
            # elif price>=70:
            #     priceClass = [0,0,0,0,0,0,0,0,0,1]
        except Exception as e:
            print e
            print l
            print len(prices)
            exit()
        if price >5000:
            continue
        x.append(str(l[3])+" "+str(l[5]))
        y.append([price])
    return [x, y]


def create_vocabulary(words,maxlength):
    all_text=[]
    print "all_text: "+str(len(words)) 
    for sample in words:
        sentence = []
        for word in sample.split(" ")[:maxlength]:
            sentence.append(word)  
        all_text.extend(sentence)
    print len(all_text) 
    vocabulary_size=10000
    count = [('oov', -1)]
    count.extend(collections.Counter(all_text).most_common(vocabulary_size - 1))
    vocabulary = set()
    for element in count:
        vocabulary.add(element[0]) 
    return vocabulary

def substitute_oov(text,vocabulary,maxlength):
    process_text = []

    for sample in text:
        #i = i+1
        processed_sample = []
        for word in sample.split(" ")[:maxlength]:
            if word in vocabulary:
                processed_sample.append(word)
            else:
                processed_sample.append("oov")
        processed_sample = " ".join(processed_sample)        
        process_text.append(processed_sample)
    return process_text

def substitute_oov_test(text,vocabulary,maxlength):
    process_text = []

    for sample in text:
        #i = i+1
        processed_sample = []
        for word in sample.split(" ")[:maxlength]:
            #if word in vocabulary:
            if vocabulary.get(word)!=0:
                processed_sample.append(word)
            else:
                processed_sample.append("oov")
        processed_sample = " ".join(processed_sample)        
        process_text.append(processed_sample)
    return process_text

def substitute_bgr_oov(text,vocabulary,maxlength):
    process_text = []

    for sample in text:
        #i = i+1
        processed_sample = []
        for word in sample[:maxlength]:
            if word in vocabulary:
                processed_sample.append(word)
            else:
                processed_sample.append("bgr-oov")
        processed_sample = " ".join(processed_sample)        
        process_text.append(processed_sample)
    return process_text



#def replace_all(text, vocabulary):
#    process_text = []
#    for t in text:
#        print t
#        for i in vocabulary:
#            t = t.replace(i,'oov')
#        print t
#        exit()
#        process_text.append(t)
#    return process_text



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def ptb_producer(raw_data, batch_size, num_steps, name=None):
      with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
          epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
        y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
        return x, y


def find_bigrams(text):
    bigrammize_text = []

    for sentence in text: 
        #niter = niter+1
        words = sentence.split()
        bigram = zip(words, words[1:])
        bigram = [t[0]+"-"+ t[1] for t in bigram] 
        #bigrammize_sentence = words + bigram
        bigrammize_text.append(bigram)
        #if niter%100==0:
        #    print niter
    return bigrammize_text

def create_bigram_voc(bigrams):
    bigram_voca_size=5000
    count = [('oobv', -1)]
    Allbigrams=[]
    for b in bigrams:
        Allbigrams.extend(b)
    count.extend(collections.Counter(Allbigrams).most_common(bigram_voca_size - 1))
    vocabulary = set()
    for element in count:
        vocabulary.add(element[0]) 
    return vocabulary


def do_ngrams(text,n):
    ngrams_text = []
    for item in text:
        words = item.split(" ")
        words = find_ngrams(words, n)
        words=["-".join(word) for word in words]
        ngrams_text.append(" ".join(words[1:]))
    return ngrams_text


def create_ngram_voc(ngrams, size):
    list_vocabulary = [line.split(" ") for line in ngrams]
    list_vocabulary = [item for sublist in list_vocabulary for item in sublist]
    list_vocabulary= [ite for ite, it in Counter(list_vocabulary).most_common(size)]
    list_vocabulary = set(list_vocabulary)
    return list_vocabulary


def build_noov(ngram, vocabulary):
    aux_text = []
    for element in ngram:
        actual_word = []
        for word in element.split(" "):
            if word in vocabulary:
                actual_word.append(word)
            else:
                actual_word.append("noov")
        actual_word = " ".join(actual_word)
        aux_text.append(actual_word)
    return aux_text


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


