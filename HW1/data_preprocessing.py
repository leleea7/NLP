import collections
import random
import numpy as np
import pickle
import re

### preprocess_text ###
# This function generates the dataset
#
### Parameters ###
# line: text line of a file in the dataset folder
# stopwords: a set of stopwords
### Return values ###
# clean_sentences: a list of sentences (each sentence is represented as a list of words)
# EXAMPLE:
# >>> line = 'Hello guys!!!. I am Emanuele. THIS is the first NLP homework'
# >>> preprocess_text(line)
# [['hello', 'guys'], ['emanuele'], ['first', 'nlp', 'homework']]
def preprocess_text(line, stopwords):
    line = re.sub(r'[^a-z0-9.\' ]', ' ', line.lower())
    sentences = line.strip().replace('\n', '').split('.')
    clean_sentences = []
    for sentence in sentences:
        if sentence:
            split = sentence.strip().split()
            tmp_split = []
            for word in split:
                if word not in stopwords:
                    word = word.split('\'')
                    for splitted_word in word:
                        splitted_word = splitted_word.strip()
                        splitted_word = ''.join(char for char in splitted_word if char.isalnum())
                        if splitted_word and not splitted_word.isdigit() and splitted_word not in stopwords:
                            tmp_split.append(splitted_word)
            if tmp_split:
                clean_sentences.append(tmp_split)
    return clean_sentences

def shuffle_sentences(data):
    random.shuffle(data)

### adapt_to_batch_training ###
# This function returns a dataset that contains pairs (batch_input, batch_label)
def adapt_to_batch_training(data, window_size):
    train = []
    for sentence in data:
        for i in range(len(sentence)):
            for j in range(i - window_size, i + window_size + 1):
                if j < 0 or j >= len(sentence) or i == j:
                    continue
                train.append((sentence[i], sentence[j]))
    return train

### generate_batch ###
# This function generates the train data and label batch from the dataset.
#
### Parameters ###
# batch_size: the number of train_data,label pairs to produce per batch
# curr_batch: the current batch number.
# window_size: the size of the context
# data: the dataset
### Return values ###
# train_data: train data for current batch
# labels: labels for current batch
def generate_batch(batch_size, curr_batch, window_size, data):

    ###FILL HERE###
    train_data = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    p = 0
    for i in range(curr_batch * batch_size, (curr_batch + 1) * batch_size):
        train_data[p] = data[i][0]
        labels[p] = data[i][1]
        p += 1

    return train_data, labels

### build_dataset ###
# This function is responsible of generating the dataset and dictionaries.
# While constructing the dictionary take into account the unseen words by
# retaining the rare (less frequent) words of the dataset from the dictionary
# and assigning to them a special token in the dictionary: UNK. This
# will train the model to handle the unseen words.
### Parameters ###
# sentences: a list of sentences (each sentence is a list)
# vocab_size:  the size of vocabulary
#
### Return values ###
# data: list of codes (integers from 0 to vocabulary_size-1).
#       This is the original text but words are replaced by their codes
# dictionary: map of words(strings) to their codes(integers)
# reverse_dictionary: maps codes(integers) to words(strings)
#def build_dataset(words, vocab_size):
def build_dataset(sentences, vocab_size):
    dictionary = dict()

    ###FILL HERE###

    #1 - insert words into the dictionary
    dictionary['UNK'] = 0
    index = 1
    for word, _ in collections.Counter([word for sentence in sentences for word in sentence]).most_common(vocab_size - 1):
        dictionary[word] = index
        index += 1

    #2 - insert indexes into data
    data = []
    for sentence in sentences:
        lst = []
        for word in sentence:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
            lst.append(index)
        data.append(lst)

    #3 - create the reversed dictionary
    reversed_dictionary = {value: key for key, value in dictionary.items()}
    print('Dictionary size', len(dictionary))

    return data, dictionary, reversed_dictionary

###
# Save embedding vectors in a suitable representation for the domain identification task
###
def save_vectors(vectors):

    ###FILL HERE###
    pickle.dump(vectors, open('vectors.pkl', 'wb'))


# Reads through the analogy question file.
#    Returns:
#      questions: a [n, 4] numpy array containing the analogy question's
#                 word ids.
#      questions_skipped: questions skipped due to unknown words.
#
def read_analogies(file, dictionary):
    questions = []
    questions_skipped = 0
    with open(file, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                continue
            words = line.strip().lower().split(" ")
            ids = [dictionary.get(str(w.strip())) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)
