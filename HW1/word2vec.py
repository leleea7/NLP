import os
import pickle
import tensorflow as tf
import numpy as np
import tqdm
from tensorboard.plugins import projector
from data_preprocessing import generate_batch, build_dataset, save_vectors, read_analogies, adapt_to_batch_training, shuffle_sentences, preprocess_text
from evaluation import evaluation
import matplotlib.pyplot as plt

# run on CPU
# comment this part if you want to run it on GPU
'''os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""'''

### PARAMETERS ###

BATCH_SIZE = 32             # Number of samples per batch
EMBEDDING_SIZE = 128        # Dimension of the embedding vector.
WINDOW_SIZE = 2             # How many words to consider left and right.
NEG_SAMPLES = 64            # Number of negative examples to sample.
VOCABULARY_SIZE = 30000     # The most N word to consider in the dictionary
EPOCHS = 1

TRAIN_DIR = "dataset/DATA/TRAIN"
VALID_DIR = "dataset/DATA/DEV"
TMP_DIR = "tmp/"
ANALOGIES_FILE = "dataset/eval/questions-words.txt"

def save_plots():
    f = open('log.txt', 'r')

    accuracy = []
    loss = []
    for row in f.readlines():
        row = row.split()
        accuracy.append(float(row[0]))
        loss.append(float(row[1]))

    # save the accuracy plot
    plt.figure(figsize=(13, 10))
    plt.title('Accuracy plot')
    plt.xlabel('Iteractions')
    plt.ylabel('Accuracy')
    plt.plot([i * 10000 for i in range(len(accuracy))], accuracy)
    plt.savefig('accuracy.png')

    # save the loss plot
    plt.figure(figsize=(13, 10))
    plt.title('Loss plot')
    plt.xlabel('Iteractions')
    plt.ylabel('Loss')
    plt.plot([i * 10000 for i in range(len(loss))], loss)
    plt.savefig('loss.png')

### READ THE TEXT FILES ###

# Read the data into a list of strings.
# the domain_words parameters limits the number of words to be loaded per domain
def read_data(directory, domain_words=-1):
    data = []
    stopwords = set([w.rstrip('\r\n') for w in open('stopwords.txt')])
    for domain in os.listdir(directory):
        print(domain)
        limit = domain_words
        for f in os.listdir(os.path.join(directory, domain)):
            if f.endswith(".txt"):
                with open(os.path.join(directory, domain, f), encoding='utf8') as file:
                    for line in file.readlines():
                        split = preprocess_text(line, stopwords)  # split is a list of sentences
                        for sentence in split:
                            if sentence:
                                if limit > 0 and limit - len(sentence) < 0:
                                    sentence = sentence[:limit]
                                else:
                                    limit -= len(sentence)
                                if limit >= 0 or limit == -1:
                                    data += [sentence]
    return data


# load the training set
if not os.path.exists(TMP_DIR):
    # if the temporary folder doesn't exist then create it
    os.makedirs(TMP_DIR)
if os.path.exists(TMP_DIR + 'data.pkl'):
    # if the dataset has already been stored in data.pkl then load it
    clean_data = pickle.load(open(TMP_DIR + 'data.pkl', 'rb'))
else:
    # otherwise read the dataset and store it in data.pkl
    clean_data = read_data(TRAIN_DIR, domain_words=1000000)
    pickle.dump(clean_data, open(TMP_DIR + 'data.pkl', 'wb'))
#print(clean_data)


print('Data size', sum([len(sentence) for sentence in clean_data]))
# the portion of the training set used for data evaluation
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


### CREATE THE DATASET AND WORD-INT MAPPING ###

data, dictionary, reverse_dictionary = build_dataset(clean_data, VOCABULARY_SIZE)
del clean_data  # Hint to reduce memory.
# read the question file for the Analogical Reasoning evaluation
questions = read_analogies(ANALOGIES_FILE, dictionary)

### MODEL DEFINITION ###

graph = tf.Graph()
eval = None

with graph.as_default():
    # Define input data tensors.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    ### FILL HERE ###

    embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
    weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE],
                                              stddev=1.0 / EMBEDDING_SIZE ** 0.5))
    emb = tf.nn.embedding_lookup(embeddings, train_inputs)
    biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

    with tf.name_scope('loss'):

        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights,
                                             biases,
                                             train_labels,
                                             emb,
                                             NEG_SAMPLES,
                                             VOCABULARY_SIZE)) ### FILL HERE ###

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

    # evaluation graph
    eval = evaluation(normalized_embeddings, dictionary, questions)

### TRAINING ###

# Step 5: Begin training.

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(TMP_DIR, session.graph)
    # We must initialize all variables before we use them.
    init.run()

    # added (reload the model if it exists and continue to train)
    try:
        saver.restore(session, os.path.join(TMP_DIR, 'model.ckpt'))
        print('Model restored')
    except:
        print('Model initialized')

    for epoch in range(EPOCHS):
        shuffle_sentences(data) # for each epoch shuffle sentences
        train = adapt_to_batch_training(data, WINDOW_SIZE) # create a list of pairs (context, word)
        num_steps = len(train) // BATCH_SIZE
        average_loss = 0
        bar = tqdm.tqdm(range(num_steps), 'Epoch: ' + str(epoch + 1) + '/' + str(EPOCHS))
        for step in bar:
            batch_inputs, batch_labels = generate_batch(BATCH_SIZE, step, WINDOW_SIZE, train)

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op
            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict={train_inputs: batch_inputs, train_labels: batch_labels},
                run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if (step % 10000 == 0 and step > 0) or step == num_steps - 1:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
            if step == (num_steps - 1) and epoch == EPOCHS - 1:
                del data
                del train
                writer.add_run_metadata(run_metadata, 'step%d' % step)
            if (step % 10000 is 0 and step > 0) or step == num_steps - 1:
                accuracy = eval.eval(session)
                print("avg loss: "+str(average_loss/step))
                # every 10000 steps save accuracy and loss in log.txt
                with open('log.txt', 'a', encoding='utf8') as f:
                    f.write(str(accuracy) + ' ' + str(average_loss/step) + '\n')
                f.close()

    final_embeddings = normalized_embeddings.eval()

    ### SAVE VECTORS ###

    save_vectors(final_embeddings)

    ### SAVE ACCURACY AND LOSS PLOTS ###

    save_plots()

    # Write corresponding labels for the embeddings.
    with open(TMP_DIR + 'metadata.tsv', 'w', encoding='utf8') as f:
        for i in range(VOCABULARY_SIZE):
            f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints
    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    # I commented the following row because of a bug (removing TMP_DIR it works)
    #embedding_conf.metadata_path = os.path.join(TMP_DIR, 'metadata.tsv')
    embedding_conf.metadata_path = os.path.join('metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()
