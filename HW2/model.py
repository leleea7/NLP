import tensorflow as tf
import os
import tqdm
from data_preprocessing import load_training_set, load_glove_embeddings, generate_batch, adapt_to_batch_training
from evaluation import get_predictions, pad_sequence_dev

### HYPERPARAMETERS ###

BATCH_SIZE = 16
HIDDEN_SIZE = 100
WINDOW_SIZE = 20
LEARNING_RATE = 0.03
EPOCHS = 1
TRAIN = True
GENERATE_TEST_ANSWER = False

### DIRECTORIES ###

DATA_DIR = 'data/'
TMP_DIR = 'tmp/'

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

### DATASETS ###

embeddings, word2id, id2word = load_glove_embeddings(DATA_DIR)
training_data, sense2id, id2sense, senses = load_training_set(DATA_DIR, word2id)
n_senses = len(sense2id)

### MODEL ###

graph = tf.Graph()

with graph.as_default():

    with tf.name_scope('input'):
        # shape = (batch_size, max length of sentence in batch)
        word_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch_size)
        sequence_lengths = tf.placeholder(tf.int32, shape=[None])

    with tf.name_scope('embedding'):
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)
        # shape = (batch, sentence, word_vector_size)
        pretrained_embeddings = tf.nn.embedding_lookup(L, word_ids)

    with tf.name_scope('lstm'):
        cell_fw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
        cell_bw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pretrained_embeddings,
                                                                        sequence_length=sequence_lengths,
                                                                        dtype=tf.float32)
        context_rep = tf.concat([output_fw, output_bw], axis=-1)

    with tf.name_scope('softmax'):
        W = tf.get_variable('W', shape=[2 * HIDDEN_SIZE, n_senses], dtype=tf.float32)
        b = tf.get_variable('b', shape=[n_senses], dtype=tf.float32, initializer=tf.zeros_initializer())
        ntime_steps = tf.shape(context_rep)[1]
        context_rep_flat = tf.reshape(context_rep, [-1, 2 * HIDDEN_SIZE])
        pred = tf.matmul(context_rep_flat, W) + b
        scores = tf.reshape(pred, [-1, ntime_steps, n_senses])
        labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

    with tf.name_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
        # shape = (batch, sentence, nclasses)
        mask = tf.sequence_mask(sequence_lengths)
        # apply mask
        losses = tf.boolean_mask(losses, mask)
        loss = tf.reduce_mean(losses)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE)
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
    except:
        print('Model initialized')


    if GENERATE_TEST_ANSWER:
        # Generate predictions and write them into 1486470_test_answer.txt
        run_metadata = tf.RunMetadata()
        results = []
        for line in open(DATA_DIR + 'test_data.txt', 'r', encoding='utf8').readlines():
            sentence = line.split()
            l = []
            s = []
            for word in sentence:
                word = word.split('|')
                if word[1] in word2id:
                    l.append(word2id[word[1].lower()])
                else:
                    l.append(word2id['unk'])
                try:
                    s.append(word[3])
                except:
                    s.append('')
            sentence_to_predict, sentence_len = pad_sequence_dev(l, WINDOW_SIZE)
            senses_predicted = session.run([scores],
                                           feed_dict={word_ids: sentence_to_predict,
                                                      sequence_lengths: sentence_len},
                                           run_metadata=run_metadata)

            res = []
            k = 0
            to_predict = len([(word) for word in s if word])
            # for each sentence
            for i in range(len(senses_predicted[0])):
                # for each word
                for j in range(len(senses_predicted[0][i])):
                    if to_predict == 0:
                        k += 1
                        continue
                    if k < len(s) and not s[k]:
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
                        res.append((s[k], id2sense[index]))
                    else:
                        for key in sense2id:
                            if key.startswith('bn:'):
                                if senses_predicted[0][i][j][sense2id[key]] > max_score:
                                    max_score = senses_predicted[0][i][j][sense2id[key]]
                                    index = sense2id[key]
                        res.append((s[k], id2sense[index]))
                    to_predict -= 1
                    k += 1
            results += res
        #print(results)
        with open('1486470_test_answer.txt', 'w') as f:
            for pair in results:
                f.write(pair[0] + '\t' + pair[1] + '\n')

        exit()

    if not TRAIN:
        run_metadata = tf.RunMetadata()
        get_predictions(run_metadata, DATA_DIR, word2id, WINDOW_SIZE, session, word_ids, sequence_lengths, [scores], id2word, senses, sense2id, id2sense)
        exit()




    batch_dataset_train, label_dataset_train, sequence_dataset_train = adapt_to_batch_training(training_data, WINDOW_SIZE)

    for epoch in range(EPOCHS):
        average_loss = 0
        num_steps = len(batch_dataset_train) // BATCH_SIZE
        for step in tqdm.tqdm(range(num_steps), 'Epoch: ' + str(epoch + 1) + '/' + str(EPOCHS)):

            batch_inputs, batch_labels, seq_len = generate_batch(BATCH_SIZE, step, batch_dataset_train, label_dataset_train, sequence_dataset_train)
            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            _, l = session.run([train_op, loss],
                               feed_dict={word_ids: batch_inputs,
                                          labels: batch_labels,
                                          sequence_lengths: seq_len},
                               run_metadata=run_metadata)

            average_loss += l

            # print loss every 1000 steps
            if (step % 1000 == 0 and step > 0) or (step == (num_steps - 1)):
                print('Loss:', str(average_loss / step))
            if step == (num_steps - 1) and epoch == EPOCHS - 1:
                writer.add_run_metadata(run_metadata, 'step%d' % step)

        # Show F1 scores after each epoch
        run_metadata = tf.RunMetadata()
        get_predictions(run_metadata, DATA_DIR, word2id, WINDOW_SIZE, session, word_ids, sequence_lengths, [scores],
                        id2word, senses, sense2id, id2sense)


    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

writer.close()