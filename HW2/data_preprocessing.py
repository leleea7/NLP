import numpy as np
import xml.etree.ElementTree as et
import math

### load_glove_embeddings ###
# This function generates the embedding matrix, the word dictionary and the reversed word dictionary
#
### Parameters ###
# DATA_DIR: the directory that contains the glove embeddings
### Return values ###
# embeddings: a numpy matrix
# word2id: word dictionary (each lemma has a unique id)
# id2word: reversed word dictionary

def load_glove_embeddings(DATA_DIR):
    word2id = {}
    embedding = {}
    index = 0
    f = open(DATA_DIR + 'glove.6B.100d.txt', 'r', encoding='utf8')
    for line in f.readlines():
        line = line.split()
        key, values = line[0], line[1 : ]
        word2id[key] = index
        index += 1
        embedding[key] = np.array(values, dtype=np.float32)
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.zeros((len(embedding), len(embedding[id2word[0]])), dtype=np.float32)
    for key in embedding:
        embeddings[word2id[key]] = embedding[key]
    return embeddings, word2id, id2word


### load_training_set ###
# This function generates the train set, the sense dictionary, the reversed sense dictionary, and the senses dictionary
#
### Parameters ###
# DATA_DIR: the directory that contains the glove embeddings
# word2id: the word dictionary
### Return values ###
# training_data: a list of sentences, each sentence is a list that contains pairs (id(lemma), sense(lemma))
# sense2id: sense dictionary (each sense has a unique id)
# id2sense: reversed sense dictionary

def load_training_set(DATA_DIR, word2id):
    tree = et.parse(DATA_DIR + 'semcor.data.xml')
    corpus = tree.getroot()
    sense2id = {}
    index = 0
    training_data = []
    senses = {}
    f = open(DATA_DIR + 'semcor.gold.key.bnids.txt', 'r', encoding='utf8')
    for text in corpus:
        for sentence in text:
            s = []
            for node in sentence:
                lemma = node.attrib['lemma'].lower()
                if 'id' in node.attrib:
                    sense = f.readline().split()[1]
                    if sense not in sense2id:
                        sense2id[sense] = index
                        index += 1
                    if lemma in senses:
                        senses[lemma].add(sense)
                    else:
                        senses[lemma] = set()
                        senses[lemma].add(sense)
                    if lemma in word2id:
                        s.append((word2id[lemma], sense2id[sense]))
                    else:
                        s.append((word2id['unk'], sense2id[sense]))
                else:
                    if lemma not in sense2id:
                        sense2id[lemma] = index
                        index += 1
                    if lemma in word2id:
                        s.append((word2id[lemma], sense2id[lemma]))
                    else:
                        s.append((word2id['unk'], sense2id[lemma]))
            training_data.append(s)
    id2sense = {v: k for k, v in sense2id.items()}
    return training_data, sense2id, id2sense, senses

### load_development_set ###
# This function generates the train set, the sense dictionary, the reversed sense dictionary, and the senses dictionary
#
### Parameters ###
# DATA_DIR: the directory that contains the glove embeddings
# word2id: the word dictionary
### Return values ###
# dataset: a dictionary that groups each Semeval/Senseval dataset (each dataset is represented as a pair(list of lemma ids, list of senses as strings))

def load_development_set(DATA_DIR, word2id):
    tree = et.parse(DATA_DIR + 'ALL.data.xml')
    corpus = tree.getroot()
    dataset = {}        #it contains pairs (encoded sequence, senses)
    f = open(DATA_DIR + 'ALL.gold.key.bnids.txt', 'r', encoding='utf8')
    for text in corpus:
        for sentence in text:
            dataset_name = sentence.attrib['id']
            dataset_name = dataset_name[: dataset_name.index('.')]
            if dataset_name not in dataset:
                dataset[dataset_name] = []
            s = []
            k = []
            for node in sentence:
                lemma = node.attrib['lemma'].lower()
                if 'id' in node.attrib:
                    sense = f.readline().split()[1]
                    k.append(sense)
                    if lemma in word2id:
                        s.append(word2id[lemma])
                    else:
                        s.append(word2id['unk'])
                else:
                    k.append('')
                    if lemma in word2id:
                        s.append(word2id[lemma])
                    else:
                        s.append(word2id['unk'])
            dataset[dataset_name].append((s, k))
    return dataset

### adapt_to_batch_training ###
# This function generates the preprocessed train set
#
### Parameters ###
# train: the train set
# window_size: the window size (integer)
### Return values ###
# batch_dataset: the list of batch splitted according to the window_size
# label_dataset: the list of labels splitted according to the window_size
# sequence_dataset: the list of sequence lengths

def adapt_to_batch_training(train, window_size):
    batch_dataset = []
    label_dataset = []
    sequence_dataset = []
    for sentence in train:
        batches, labels, seq_len = pad_sequence(sentence, window_size)
        for batch in batches:
            batch_dataset.append(batch)
        for label in labels:
            label_dataset.append(label)
        for sequence_length in seq_len:
            sequence_dataset.append(sequence_length)
    return batch_dataset, label_dataset, sequence_dataset

### generate_batch ###
# This function generates three sub lists of the train set
#
### Parameters ###
# batch_size: an integer
# step: an integer
# batch: the list of batches
# label: the list of labels
# sequence: the list of sequence lengths
### Return values ###
# btc: a sub list of batch
# lbl: a sub list of label
# seq: a sub list of sequence

def generate_batch(batch_size, step, batch, label, sequence):
    btc = np.array(batch[step * batch_size : (step + 1) * batch_size])
    lbl = np.array(label[step * batch_size : (step + 1) * batch_size])
    seq = np.array(sequence[step * batch_size : (step + 1) * batch_size])
    return btc, lbl, seq

### pad_sequence ###
# This function applies the padding to the given sentence
#
### Parameters ###
# sentence: a list of pairs(lemma ids, sense ids)
# window_size: the window size (integer)
### Return values ###
# batch: the padded list of lemma ids
# label: the padded list of sense ids
# sequence_length: the padded list of sequence lengths

def pad_sequence(sentence, window_size):
    n = len(sentence)
    sequence_length = []
    rows = math.ceil(len(sentence) / window_size)
    batch = np.zeros(shape=(rows, window_size), dtype=np.int32)
    label = np.zeros(shape=(rows, window_size), dtype=np.int32)
    for i in range(rows):
        btc = [pair[0] for pair in sentence[i * window_size: (i + 1) * window_size]]
        lbl = [pair[1] for pair in sentence[i * window_size: (i + 1) * window_size]]
        if n >= window_size:
            sequence_length.append(window_size)
        else:
            sequence_length.append(n)
        n -= window_size
        for j in range(len(btc)):
            batch[i][j] = btc[j]
            label[i][j] = lbl[j]
    return batch, label, np.array(sequence_length)





