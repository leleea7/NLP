import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def evaluate(gold, prediction, index2role):
    TP = FP = TN = FN = 0
    assert(len(gold) == len(prediction))
    for i in range(len(gold)):
        true = index2role[gold[i]]
        pred = index2role[prediction[i]]
        if true == pred:
            if pred != '_':
                TP += 1
            else:
                TN += 1
        else:
            if pred != '_':
                FP += 1
            else:
                FN += 1
    precision = precision_score(TP, FP)
    recall = recall_score(TP, FN)
    f1 = f1_score(precision, recall)
    accuracy = accuracy_score(TP, TN, FP, FN)
    return precision, recall, f1, accuracy

def precision_score(true_positive, false_positive):
    try:
        return true_positive / (true_positive + false_positive)
    except:
        return 0

def recall_score(true_positive, false_negative):
    try:
        return true_positive / (true_positive + false_negative)
    except:
        return 0

def f1_score(precision, recall):
    try:
        return 2 * (precision * recall) / (precision + recall)
    except:
        return 0

def accuracy_score(true_positive, true_negative, false_positive, false_negative):
    try:
        return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    except:
        return 0

def draw_confusion_matrix(gold, prediction, index2role):
    assert(len(gold) == len(prediction))
    conf_mat = np.zeros(shape=(len(index2role), len(index2role)), dtype=np.int32)
    for i in range(len(gold)):
        row = gold[i]
        column = prediction[i]
        conf_mat[row][column] += 1
    conf_mat = pd.DataFrame(conf_mat,
                            index=[index2role[index] for index in index2role],
                            columns=[index2role[index] for index in index2role])
    plt.figure(figsize=(30, 25))
    sb.heatmap(conf_mat, annot=True)
    plt.savefig('confusion_matrix.png')
