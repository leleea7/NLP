import tensorflow as tf
import os
import tqdm
import data_preprocessing as dp
from evaluation import evaluate, draw_confusion_matrix
import numpy as np


### DIRECTORIES ###

DATA_DIR = 'SRLData/EN/'
TMP_DIR = 'tmp/'
TEST_DIR = 'TestData/'

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)


### HYPERPARAMETERS ###

BATCH_SIZE = 10     # How many sentences i take
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 1
PRINT_RESULTS = True
GENERATE_ANSWER = False
GLOBAL_EPOCH = dp.global_epoch(TMP_DIR + 'epoch.txt')


### LOAD GLOVE EMBEDDINGS AND TRAIN SET ###

embeddings, word2index = dp.load_embeddings('WSD/Data/glove.6B.100d.txt')
sentences = dp.get_sentences(DATA_DIR + 'CoNLL2009-ST-English-train.txt')
training_data, role2index, pos2index = dp.load_dataset(sentences, word2index)
del sentences
#print('ROLES DICTIONARY SIZE:', len(role2index))
#print('POS DICTIONARY SIZE', len(pos2index))


### MODEL ###

graph = tf.Graph()

with graph.as_default():

    with tf.name_scope('input'):
        # shape = (batch_size, max length of sentence in batch)
        word_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch_size, max length of sentence in batch)
        pos_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch_size, max length of sentence in batch, 1)
        flags = tf.placeholder(tf.float32, shape=[None, None, 1])

        # shape = (batch_size)
        sequence_lengths = tf.placeholder(tf.int32, shape=[None])

    with tf.name_scope('embedding'):
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)

        pos_embeddings = tf.Variable(tf.random_uniform([len(pos2index), 50], -1.0, 1.0), dtype=tf.float32)

        # shape = (batch, sentence, word_vector_size)
        pretrained_embeddings = tf.nn.embedding_lookup(L, word_ids)

        pos_emb = tf.nn.embedding_lookup(pos_embeddings, pos_ids)

        final_embeddings = tf.concat([pretrained_embeddings, pos_emb, flags], axis=-1)

    with tf.name_scope('lstm'):
        cell_fw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
        cell_bw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, final_embeddings,
                                                                        sequence_length=sequence_lengths,
                                                                        dtype=tf.float32)
        context_rep = tf.concat([output_fw, output_bw], axis=-1)

    with tf.name_scope('softmax'):
        W = tf.get_variable('W', shape=[2 * HIDDEN_SIZE, len(role2index)], dtype=tf.float32)
        b = tf.get_variable('b', shape=[len(role2index)], dtype=tf.float32, initializer=tf.zeros_initializer())
        ntime_steps = tf.shape(context_rep)[1]
        context_rep_flat = tf.reshape(context_rep, [-1, 2 * HIDDEN_SIZE])
        pred = tf.matmul(context_rep_flat, W) + b
        scores = tf.reshape(pred, [-1, ntime_steps, len(role2index)])
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
        training_data, _, _ = dp.load_dataset(sentences, word2index, role_dict=role2index, pos_dict=pos2index)
        del sentences
        y_pred = []
        y_true = []
        for sentence in tqdm.tqdm(training_data, 'Printing results'):
            batch_inputs, batch_pos, batch_flag, batch_labels, seq_len = dp.pad_sequence([sentence])
            labels_predicted = session.run(labels_pred,
                        feed_dict={word_ids: batch_inputs,
                                   pos_ids: batch_pos,
                                   flags: np.expand_dims(batch_flag, axis=-1),
                                   sequence_lengths: seq_len},
                        run_metadata=run_metadata)

            for i in range(len(labels_predicted)):
                for j in range(len(labels_predicted[i])):
                    y_pred.append(labels_predicted[i][j])
                    y_true.append(batch_labels[i][j])

        precision, recall, f1, accuracy = evaluate(y_true, y_pred, {v: k for k, v in role2index.items()})
        print('F1 score:'.ljust(20, ' '), f1)
        print('Precision score:'.ljust(20, ' '), precision)
        print('Recall score:'.ljust(20, ' '), recall)
        print('Accuracy score:'.ljust(20, ' '), accuracy)
        writer.close()
        draw_confusion_matrix(y_true, y_pred, {v: k for k, v in role2index.items()})
        exit()

    if GENERATE_ANSWER:
        run_metadata = tf.RunMetadata()
        f = open('1486470_test.tsv', 'w', encoding='utf8')
        sentences = dp.get_sentences(TEST_DIR + 'test.csv')
        test_data = dp.load_test(sentences, word2index, pos2index)
        index2role = {v: k for k, v in role2index.items()}
        k = 0
        for sentence_list in tqdm.tqdm(test_data, 'Generating 1486470_test.tsv'):
            answer_matrix = []
            for sentence in sentence_list:
                batch_inputs, batch_pos, batch_flag, seq_len = dp.pad_sequence([sentence], labels=False)

                labels_predicted = session.run(labels_pred,
                                               feed_dict={word_ids: batch_inputs,
                                                          pos_ids: batch_pos,
                                                          flags: np.expand_dims(batch_flag, axis=-1),
                                                          sequence_lengths: seq_len},
                                               run_metadata=run_metadata)
                answer_matrix.append(dp.convert(labels_predicted, index2role))

            if dp.no_flag(batch_flag):
                for i in range(len(sentences[k])):
                    f.write('\t'.join(sentences[k][i]) + '\n')
            else:
                answer_matrix = dp.transpose(answer_matrix)
                i = 0
                for line in answer_matrix:
                    f.write('\t'.join(sentences[k][i] + [role for role in line]) + '\n')
                    i += 1
            f.write('\n')
            k += 1
        f.close()
        writer.close()
        exit()

    batch, pos, flag, label, sequence = dp.generate_batches(training_data, BATCH_SIZE)
    del training_data

    for epoch in range(EPOCHS):
        average_loss = 0
        num_steps = len(batch)
        for step in tqdm.tqdm(range(num_steps), 'Epoch: ' + str(epoch + 1) + '/' + str(EPOCHS)):

            batch_inputs = batch[step]
            batch_pos = pos[step]
            batch_flag = flag[step]
            batch_labels = label[step]
            seq_len = sequence[step]

            coefs = dp.generate_coefficients(batch_labels, seq_len, role2index)

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            _, l = session.run([train_op, loss],
                               feed_dict={word_ids: batch_inputs,
                                          pos_ids: batch_pos,
                                          flags: np.expand_dims(batch_flag, axis=-1),
                                          labels: batch_labels,
                                          sequence_lengths: seq_len,
                                          coefficients: coefs},
                               run_metadata=run_metadata)

            average_loss += l

            # print loss every 5000 steps
            if (step % 5000 == 0 and step > 0) or (step == (num_steps - 1)):
                print('Loss:', str(average_loss / step))
            if step == (num_steps - 1) and epoch == EPOCHS - 1:
                writer.add_run_metadata(run_metadata, 'step%d' % step, global_step=GLOBAL_EPOCH + epoch + 1)

        # Show F1 scores after each epoch
        print('--- RESULTS ON DEVELOPMENT SET ---')
        run_metadata = tf.RunMetadata()
        sentences = dp.get_sentences(DATA_DIR + 'CoNLL2009-ST-English-development.txt')
        training_data, _, _ = dp.load_dataset(sentences, word2index, role_dict=role2index, pos_dict=pos2index)
        del sentences
        y_pred = []
        y_true = []
        for sentence in training_data:
            batch_inputs, batch_pos, batch_flag, batch_labels, seq_len = dp.pad_sequence([sentence])
            labels_predicted = session.run(labels_pred,
                                           feed_dict={word_ids: batch_inputs,
                                                      pos_ids: batch_pos,
                                                      flags: np.expand_dims(batch_flag, axis=-1),
                                                      sequence_lengths: seq_len},
                                           run_metadata=run_metadata)

            for i in range(len(labels_predicted)):
                for j in range(len(labels_predicted[i])):
                    y_pred.append(labels_predicted[i][j])
                    y_true.append(batch_labels[i][j])

        precision, recall, f1, accuracy = evaluate(y_true, y_pred, {v: k for k, v in role2index.items()})
        print('F1 score:'.ljust(20, ' '), f1)
        print('Precision score:'.ljust(20, ' '), precision)
        print('Recall score:'.ljust(20, ' '), recall)
        print('Accuracy score:'.ljust(20, ' '), accuracy)
        print()

    draw_confusion_matrix(y_true, y_pred, {v: k for k, v in role2index.items()})
    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))
    dp.global_epoch(TMP_DIR + 'epoch.txt', update=GLOBAL_EPOCH + epoch + 1)

writer.close()

