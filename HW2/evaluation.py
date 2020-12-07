import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from data_preprocessing import load_development_set
import warnings
warnings.filterwarnings('ignore')

### pad_sequence ###
# This function applies the padding to the given sentence of the development set
#
### Parameters ###
# sentence: a list of lemma ids
# window_size: the window size (integer)
### Return values ###
# padded_sentence: the padded list of lemma ids
# sequence_length: the padded list of sequence lengths

def pad_sequence_dev(sentence, window_size):
    n = len(sentence)
    rows = math.ceil(len(sentence) / window_size)
    padded_sentence = np.zeros(shape=(rows, window_size), dtype=np.int32)
    sequence_length = []
    for i in range(rows):
        btc = [num for num in sentence[i * window_size: (i + 1) * window_size]]
        for j in range(len(btc)):
            padded_sentence[i][j] = btc[j]
        if n >= window_size:
            sequence_length.append(window_size)
        else:
            sequence_length.append(n)
        n -= window_size
    return padded_sentence, np.array(sequence_length)

### get_predictions ###
# This function prints the f1 scores for each semeval/senseval dataset

def get_predictions(run_metadata, data_dir, word2id, window_size, session, word_ids, sequence_lengths, scores, id2word, senses, sense2id, id2sense):
    ### PRINT F1 SCORES ###

    development_set = load_development_set(data_dir, word2id)
    print('--- F1 SCORES ---')
    for dataset in development_set:
        y_pred = []
        y_true = []
        for sentence in development_set[dataset]:
            sentence_to_predict, sentence_len = pad_sequence_dev(sentence[0], window_size)
            senses_predicted = session.run(scores,
                                           feed_dict={word_ids: sentence_to_predict,
                                                      sequence_lengths: sentence_len},
                                           run_metadata=run_metadata)
            l = []
            k = 0
            to_predict = len([(word) for word in sentence[1] if word])
            # for each sentence
            for i in range(len(senses_predicted[0])):
                # for each word
                for j in range(len(senses_predicted[0][i])):
                    if to_predict == 0:
                        k += 1
                        continue
                    if k < len(sentence[1]) and not sentence[1][k]:
                        k += 1
                        continue
                    word = id2word[sentence_to_predict[i][j]]
                    index = None
                    max_score = float('-inf')
                    if word in senses:
                        for synset in senses[word]:
                            if senses_predicted[0][i][j][sense2id[synset]] > max_score:
                                max_score = senses_predicted[0][i][j][sense2id[synset]]
                                index = sense2id[synset]
                        l.append(id2sense[index])
                    else:
                        for key in sense2id:
                            if key.startswith('bn:'):
                                if senses_predicted[0][i][j][sense2id[key]] > max_score:
                                    max_score = senses_predicted[0][i][j][sense2id[key]]
                                    index = sense2id[key]
                        l.append(id2sense[index])
                    to_predict -= 1
                    k += 1
            # print(l)
            l = [(word) for word in l if word]
            m = [(word) for word in sentence[1] if word]
            y_pred = y_pred + l
            y_true = y_true + m
        #print(len(y_true) == len(y_pred))
        print(dataset + ':\t' + str(print_f1_score(y_true, y_pred)))

### print_f1_score ###
# This function prints the f1 scores for each semeval/senseval dataset
#
### Parameters ###
# y_true: a list of senses of the dataset
# y_pred: a list of predicted senses
def print_f1_score(y_true, y_pred):
    le = LabelEncoder().fit(y_true + y_pred)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    return f1_score(y_true, y_pred, average='macro')
