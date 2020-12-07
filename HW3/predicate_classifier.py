import tensorflow as tf
import os
import tqdm
import numpy as np
import data_preprocessing as dp
from evaluation import evaluate

def generate_batches(dataset, batch_size):
    batch_dataset = []
    pos_dataset = []
    label_dataset = []
    sequence_dataset = []
    for i in range(0, len(dataset), batch_size):
        batches, pos_tags, labels, seq_len = pad_sequence(dataset[i: i + batch_size])
        batch_dataset.append(batches)
        pos_dataset.append(pos_tags)
        label_dataset.append(labels)
        sequence_dataset.append(seq_len)
    return batch_dataset, pos_dataset, label_dataset, sequence_dataset

def pad_sequence(sentences, labels=True):
    sequence_length = [len(sentence) for sentence in sentences]
    n = max(sequence_length)
    batch = np.zeros(shape=(len(sentences), n), dtype=np.int32)
    pos_tag = np.zeros(shape=(len(sentences), n), dtype=np.int32)
    if labels:
        label = np.zeros(shape=(len(sentences), n), dtype=np.int32)
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            batch[i][j] = sentences[i][j][0]
            pos_tag[i][j] = sentences[i][j][1]
            if labels:
                label[i][j] = sentences[i][j][2]
    if labels:
        return batch, pos_tag, label, np.array(sequence_length)
    return batch, pos_tag, np.array(sequence_length)

def generate_coefficients(labels_matrix, sequence_lenght, predicate_dict, null_coefficient=0.2):
    coefficients = np.array([1 for _ in range(sum(sequence_lenght))], dtype=np.float32)
    i = 0
    k = 0
    for row in labels_matrix:
        labels = 0
        for index in row:
            if labels >= sequence_lenght[k]:
                break
            if predicate_dict['_'] == index:
                coefficients[i] = null_coefficient
            i += 1
            labels += 1
        k += 1
    return coefficients

def get_sentences(file_path):
    f = open(file_path, 'r', encoding='utf8')
    sentences = []
    l = []
    for line in f.readlines():
        if line == '\n':
            sentences.append(l)
            l = []
        else:
            l.append(line.replace('\n', '').split('\t'))    #aggiunto
    f.close()
    return sentences

def load_dataset(sentences, word2id, pos_dict=None, predicate_dict=None, test=False):
    index_pos = 0
    index_predicate = 0
    if pos_dict:
        pos2id = pos_dict
    else:
        pos2id = {}
    if predicate_dict:
        predicate2id = predicate_dict
    else:
        predicate2id = {}
        predicate2id['UNK'] = index_predicate
        index_predicate += 1
    dataset = []
    for sentence in sentences:
        s = []
        for word in sentence:
            lemma = word[2]
            pos = word[4]
            if not test:
                predicate = word[13]
            if not pos_dict:
                if pos not in pos2id:
                    pos2id[pos] = index_pos
                    index_pos += 1
            if not test:
                if not predicate_dict:
                    if predicate not in predicate2id:
                        predicate2id[predicate] = index_predicate
                        index_predicate += 1
            if test:
                if lemma in word2id:
                    s.append((word2id[lemma], pos2id[pos]))
                else:
                    s.append((word2id['unk'], pos2id[pos]))
            else:
                if lemma in word2id:
                    s.append((word2id[lemma], pos2id[pos], predicate2id[predicate])) if predicate in predicate2id else s.append((word2id[lemma], pos2id[pos], predicate2id['UNK']))
                else:
                    s.append((word2id['unk'], pos2id[pos], predicate2id[predicate])) if predicate in predicate2id else s.append((word2id['unk'], pos2id[pos], predicate2id['UNK']))
        dataset.append(s)
    return dataset, predicate2id, pos2id


### DIRECTORIES ###

DATA_DIR = 'SRLData/EN/'
TMP_DIR = 'tmp2/'
TEST_DIR = 'TestData/'

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)


### HYPERPARAMETERS ###

BATCH_SIZE = 10
HIDDEN_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 1
PRINT_RESULTS = True
GENERATE_ANSWER = False
GLOBAL_EPOCH = dp.global_epoch(TMP_DIR + 'epoch.txt')


### LOAD GLOVE EMBEDDINGS AND TRAIN SET ###

embeddings, word2index = dp.load_embeddings('WSD/data/glove.6B.100d.txt')
sentences = dp.get_sentences(DATA_DIR + 'CoNLL2009-ST-English-train.txt')
training_data, predicate2index, pos2index = load_dataset(sentences, word2index)


### MODEL ###

graph = tf.Graph()

with graph.as_default():

    with tf.name_scope('input'):
        # shape = (batch_size, max length of sentence in batch)
        word_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch_size, max length of sentence in batch)
        pos_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch_size)
        sequence_lengths = tf.placeholder(tf.int32, shape=[None])

    with tf.name_scope('embedding'):
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)

        pos_embeddings = tf.Variable(tf.random_uniform([len(pos2index), 50], -1.0, 1.0), dtype=tf.float32)

        # shape = (batch, sentence, word_vector_size)
        pretrained_embeddings = tf.nn.embedding_lookup(L, word_ids)

        pos_emb = tf.nn.embedding_lookup(pos_embeddings, pos_ids)

        final_embeddings = tf.concat([pretrained_embeddings, pos_emb], axis=-1)

    with tf.name_scope('lstm'):
        cell_fw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
        cell_bw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, final_embeddings,
                                                                        sequence_length=sequence_lengths,
                                                                        dtype=tf.float32)
        context_rep = tf.concat([output_fw, output_bw], axis=-1)

    with tf.name_scope('softmax'):
        W = tf.get_variable('W', shape=[2 * HIDDEN_SIZE, len(predicate2index)], dtype=tf.float32)
        b = tf.get_variable('b', shape=[len(predicate2index)], dtype=tf.float32, initializer=tf.zeros_initializer())
        ntime_steps = tf.shape(context_rep)[1]
        context_rep_flat = tf.reshape(context_rep, [-1, 2 * HIDDEN_SIZE])
        pred = tf.matmul(context_rep_flat, W) + b
        scores = tf.reshape(pred, [-1, ntime_steps, len(predicate2index)])
        labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

    with tf.name_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
        coefficients = tf.placeholder(tf.float32, shape=[None])

        # shape = (batch, sentence, nclasses)
        mask = tf.sequence_mask(sequence_lengths)
        # apply mask
        losses = tf.boolean_mask(losses, mask)
        balanced_losses = tf.multiply(losses, coefficients)
        loss = tf.reduce_mean(balanced_losses)
        #loss = tf.reduce_mean(losses)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)

    labels_pred = tf.cast(tf.argmax(scores, axis=-1), tf.int32)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()


### SESSION ###

with tf.Session(graph=graph) as session:

    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(TMP_DIR, session.graph)

    # We must initialize all variables before we use them.
    init.run()

    # reload the model if it exists and continue to train
    try:
        saver.restore(session, os.path.join(TMP_DIR, 'model.ckpt'))
        print('Model restored')
        print('Global epoch:', GLOBAL_EPOCH)
    except:
        print('Model initialized')

    if PRINT_RESULTS:
        run_metadata = tf.RunMetadata()
        sentences = dp.get_sentences(DATA_DIR + 'CoNLL2009-ST-English-development.txt')
        dataset, _, _ = load_dataset(sentences, word2index, pos_dict=pos2index, predicate_dict=predicate2index)
        y_pred = []
        y_true = []
        for sentence in tqdm.tqdm(dataset, 'Printing results'):
            batch_inputs, batch_pos, batch_labels, seq_len = pad_sequence([sentence])
            labels_predicted = session.run(labels_pred,
                                           feed_dict={word_ids: batch_inputs,
                                                      pos_ids: batch_pos,
                                                      sequence_lengths: seq_len},
                                           run_metadata=run_metadata)

            for i in range(len(labels_predicted)):
                for j in range(len(labels_predicted[i])):
                    y_pred.append(labels_predicted[i][j])
                    y_true.append(batch_labels[i][j])

        precision, recall, f1, accuracy = evaluate(y_true, y_pred, {v: k for k, v in predicate2index.items()})
        print('F1 score:'.ljust(20, ' '), f1)
        print('Precision score:'.ljust(20, ' '), precision)
        print('Recall score:'.ljust(20, ' '), recall)
        print('Accuracy score:'.ljust(20, ' '), accuracy)
        writer.close()
        exit()

    if GENERATE_ANSWER:
        run_metadata = tf.RunMetadata()
        f = open('1486470_testverbs.tsv', 'w', encoding='utf8')
        sentences = dp.get_sentences(TEST_DIR + 'testverbs.csv')
        test_data, _, _ = load_dataset(sentences, word2index, pos_dict=pos2index, test=True)
        index2predicate = {v: k for k, v in predicate2index.items()}
        k = 0
        for sentence in tqdm.tqdm(test_data, 'Generating 1486470_testverbs.tsv'):
            batch_inputs, batch_pos, seq_len = pad_sequence([sentence], labels=False)

            labels_predicted = session.run(labels_pred,
                                           feed_dict={word_ids: batch_inputs,
                                                      pos_ids: batch_pos,
                                                      sequence_lengths: seq_len},
                                           run_metadata=run_metadata)

            predicates = [index2predicate[prediction] for prediction in labels_predicted[0]]
            for i in range(len(sentences[k])):
                if predicates[i] == '_':
                    f.write('\t'.join(sentences[k][i]) + '\t' + '_' + '\t' + '_' + '\n')
                else:
                    f.write('\t'.join(sentences[k][i]) + '\t' + 'Y' + '\t' + predicates[i] + '\n')
            f.write('\n')
            k += 1
        f.close()
        writer.close()
        exit()

    batch, pos, label, sequence = generate_batches(training_data, BATCH_SIZE)
    del training_data

    for epoch in range(EPOCHS):
        average_loss = 0
        num_steps = len(batch)
        for step in tqdm.tqdm(range(num_steps), 'Epoch: ' + str(epoch + 1) + '/' + str(EPOCHS)):

            batch_inputs = batch[step]
            batch_pos = pos[step]
            batch_labels = label[step]
            seq_len = sequence[step]
            coefs = generate_coefficients(batch_labels, seq_len, predicate2index)

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            _, l = session.run([train_op, loss],
                                      feed_dict={word_ids: batch_inputs,
                                                 pos_ids: batch_pos,
                                                 labels: batch_labels,
                                                 sequence_lengths: seq_len,
                                                 coefficients: coefs},
                                      run_metadata=run_metadata)

            average_loss += l

            # print loss every 1000 steps
            if (step % 1000 == 0 and step > 0) or (step == (num_steps - 1)):
                print('Loss:', str(average_loss / step))
            if step == (num_steps - 1) and epoch == EPOCHS - 1:
                writer.add_run_metadata(run_metadata, 'step%d' % step)

        # Show F1 scores after each epoch
        print('--- RESULTS ON DEVELOPMENT SET ---')
        run_metadata = tf.RunMetadata()
        sentences = dp.get_sentences(DATA_DIR + 'CoNLL2009-ST-English-development.txt')
        dataset, _, _ = load_dataset(sentences, word2index, pos_dict=pos2index, predicate_dict=predicate2index)
        y_pred = []
        y_true = []
        for sentence in dataset:
            batch_inputs, batch_pos, batch_labels, seq_len = pad_sequence([sentence])
            labels_predicted = session.run(labels_pred,
                                           feed_dict={word_ids: batch_inputs,
                                                      pos_ids: batch_pos,
                                                      sequence_lengths: seq_len},
                                           run_metadata=run_metadata)

            for i in range(len(labels_predicted)):
                for j in range(len(labels_predicted[i])):
                    y_pred.append(labels_predicted[i][j])
                    y_true.append(batch_labels[i][j])

        precision, recall, f1, accuracy = evaluate(y_true, y_pred, {v: k for k, v in predicate2index.items()})
        print('F1 score:'.ljust(20, ' '), f1)
        print('Precision score:'.ljust(20, ' '), precision)
        print('Recall score:'.ljust(20, ' '), recall)
        print('Accuracy score:'.ljust(20, ' '), accuracy)
        print()

    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))
    dp.global_epoch(TMP_DIR + 'epoch.txt', update=GLOBAL_EPOCH + epoch + 1)

writer.close()
