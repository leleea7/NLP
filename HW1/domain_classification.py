import pickle
import numpy as np
import os
import tqdm
from data_preprocessing import preprocess_text
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def read_data(directory):
    dataset = np.array([])
    stopwords = set([w.rstrip('\r\n') for w in open('stopwords.txt')])
    for domain in os.listdir(directory):
        print(domain)
        files = os.listdir(os.path.join(directory, domain))
        for i in tqdm.tqdm(range(len(files))):
            if files[i].endswith(".txt"):
                with open(os.path.join(directory, domain, files[i]), encoding='utf8') as file:
                    data = []
                    for line in file.readlines():
                        split = preprocess_text(line, stopwords)  # split is a list of sentences
                        for sentence in split:
                            if sentence:
                                data += [sentence]
                    val = embeddings_mean(embedding, emb, data, domain)
                    if val is not None:
                        if dataset.size == 0:
                            dataset = np.append(dataset, val)
                        else:
                            dataset = np.vstack((dataset, val))

    return dataset

def embeddings_mean(embedding, emb, data, domain=None):
    word_count = 0
    s = 0
    for sentence in data:
        for word in sentence:
            if word in emb:
                s += embedding[emb[word]]
            else:
                s += embedding[emb['UNK']]
            word_count += 1
    if word_count > 0:
        if domain:
            return np.append(s / float(word_count), domain)
        else:
            return s / float(word_count)
    return None

def read_test_data(directory):
    dataset = np.array([])
    stopwords = set([w.rstrip('\r\n') for w in open('stopwords.txt')])
    files = os.listdir(directory)
    for i in tqdm.tqdm(range(len(files))):
        if files[i].endswith(".txt"):
            with open(os.path.join(directory, files[i]), encoding='utf8') as file:
                data = []
                for line in file.readlines():
                    split = preprocess_text(line, stopwords)  # split is a list of sentences
                    for sentence in split:
                        if sentence:
                            data += [sentence]
                val = embeddings_mean(embedding, emb, data)
                if val is not None:
                    if dataset.size == 0:
                        dataset = np.append(dataset, val)
                    else:
                        dataset = np.vstack((dataset, val))
    return dataset

def save_confusion_matrix():
    df_cm = pd.DataFrame(array,
                         index=[domain for domain in os.listdir(TRAIN_DIR)],
                         columns=[domain for domain in os.listdir(TRAIN_DIR)])
    plt.figure(figsize=(20, 15))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('conf_mat.png')

embedding = pickle.load(open('vectors.pkl', 'rb'))
emb = {}
row = 0
for word in open('tmp/metadata.tsv', 'r'):
    emb[word.replace('\n','')] = row
    row += 1

TRAIN_DIR = "dataset/DATA/TRAIN"
VALID_DIR = "dataset/DATA/DEV"
TEST_DIR = "dataset/DATA/TEST"
TMP_DIR = "tmp/"

if os.path.exists(TMP_DIR + 'train.pkl'):
    # if the train set has already been stored in train.pkl then load it
    train = pickle.load(open(TMP_DIR + 'train.pkl', 'rb'))
else:
    # otherwise read the train set and store it in train.pkl
    train = read_data(TRAIN_DIR)
    pickle.dump(train, open(TMP_DIR + 'train.pkl', 'wb'))

if os.path.exists(TMP_DIR + 'dev.pkl'):
    # if the validation set has already been stored in dev.pkl then load it
    validation = pickle.load(open(TMP_DIR + 'dev.pkl', 'rb'))
else:
    # otherwise read the validation set and store it in dev.pkl
    validation = read_data(VALID_DIR)
    pickle.dump(validation, open(TMP_DIR + 'dev.pkl', 'wb'))

X_train = train[:, :-1].astype(np.float32)
Y_train = train[:, train.shape[1] - 1]
del train

X_validation = validation[:, :-1].astype(np.float32)
Y_validation = validation[:, validation.shape[1] - 1]
del validation

model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
array = confusion_matrix(Y_validation, predictions)
print(classification_report(Y_validation, predictions))
print('Accuracy: %.2f' % (accuracy_score(Y_validation, predictions) * 100) + '%')

save_confusion_matrix()

if os.path.exists(TMP_DIR + 'test.pkl'):
    # if the test dataset has already been stored in train.pkl then load it
    test = pickle.load(open(TMP_DIR + 'test.pkl', 'rb'))
else:
    # otherwise read the test dataset and store it in train.pkl
    test = read_test_data(TEST_DIR)
    pickle.dump(train, open(TMP_DIR + 'test.pkl', 'wb'))

test_predictions = model.predict(test)

with open('test_answers.tsv', 'w') as f:
    for i in range(len(test_predictions)):
        f.write('test_' + str(i) + '\t' + test_predictions[i] + '\n')