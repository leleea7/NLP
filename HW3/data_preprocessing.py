import numpy as np

def load_embeddings(file_path):
    word2index = {}
    index = 0
    f = open(file_path, 'r', encoding='utf8')
    lines = f.readlines()
    f.close()
    embeddings = np.ndarray(shape=(len(lines), len(lines[0].split()) - 1), dtype=np.float32)
    for i in range(len(lines)):
        line = lines[i].split()
        word2index[line[0]] = index
        index += 1
        embeddings[i] = line[1:]
    return embeddings, word2index

def get_sentences(file_path):
    f = open(file_path, 'r', encoding='utf8')
    sentences = []
    l = []
    for line in f.readlines():
        if line == '\n':
            sentences.append(l)
            l = []
        else:
            #l.append(line.split('\t'))
            l.append(line.replace('\n', '').split('\t'))    #aggiunto
    f.close()
    return sentences

def transpose(matrix):
    transpose = [[None for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            '''try:
                transpose[j][i] = matrix[i][j].replace('\n', '')
            except:
                transpose[j][i] = matrix[i][j]'''   #commentata
            transpose[j][i] = matrix[i][j]
    return transpose

def load_dataset(sentences, word2id, role_dict=None, pos_dict=None):
    dataset = []
    index_role = 0
    index_pos = 0
    if pos_dict:
        pos2index = pos_dict
    else:
        pos2index = {}
    if role_dict:
        role2index = role_dict
    else:
        role2index = {}
        #role2index['UNK'] = index_role
        #index_role += 1
    for sentence in sentences:
        s = []
        arguments = []
        predicates_flag = []
        for word in sentence:
            arguments.append(word[14:])
            lemma = word[2]
            pos = word[4]
            predicate = word[13]
            val = []
            if lemma in word2id:
                val.append(word2id[lemma])
            else:
                val.append(word2id['unk'])
            if not pos_dict:
                if pos not in pos2index:
                    pos2index[pos] = index_pos
                    index_pos += 1
            val.append(pos2index[pos])
            s.append(val)
            predicates_flag.append(0) if predicate == '_' else predicates_flag.append(1)
        arguments = transpose(arguments)
        indexes = [i for i in range(len(predicates_flag)) if predicates_flag[i] == 1]  # aggiunto
        for row in range(len(arguments)):
            r = []
            i = 0
            pred_flag = [0 for _ in range(len(s))]  # aggiunto
            pred_flag[indexes[row]] = 1  # aggiunto
            for argument in arguments[row]:
                if not role_dict:
                    if argument not in role2index:
                        role2index[argument] = index_role
                        index_role += 1
                    # first element of the tuple: id of the lemma
                    # second element of the tuple: id of the pos tag
                    # third element of the tuple: 0 if the predicate is _, 1 otherwise
                    # fourth element of the tuple: id of the role (label)
                    r.append((s[i][0], s[i][1], pred_flag[i], role2index[argument]))  # modificato
                else:
                    if argument not in role2index:
                        r.append((s[i][0], s[i][1], pred_flag[i], role2index['UNK']))  # modificato
                    else:
                        r.append((s[i][0], s[i][1], pred_flag[i], role2index[argument]))  # modificato
                i += 1
            dataset.append(r)
    return dataset, role2index, pos2index

def generate_batches(dataset, batch_size):
    batch_dataset = []
    pos_dataset = []
    predicate_dataset = []
    label_dataset = []
    sequence_dataset = []
    for i in range(0, len(dataset), batch_size):
        batches, pos_tags, flags, labels, seq_len = pad_sequence(dataset[i: i + batch_size])
        batch_dataset.append(batches)
        pos_dataset.append(pos_tags)
        predicate_dataset.append(flags)
        label_dataset.append(labels)
        sequence_dataset.append(seq_len)
    return batch_dataset, pos_dataset, predicate_dataset, label_dataset, sequence_dataset

def pad_sequence(sentences, labels=True):
    sequence_length = [len(sentence) for sentence in sentences]
    n = max(sequence_length)
    batch = np.zeros(shape=(len(sentences), n), dtype=np.int32)
    pos_tag = np.zeros(shape=(len(sentences), n), dtype=np.int32)
    flag = np.zeros(shape=(len(sentences), n), dtype=np.int32)
    if labels:
        label = np.zeros(shape=(len(sentences), n), dtype=np.int32)
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            batch[i][j] = sentences[i][j][0]
            pos_tag[i][j] = sentences[i][j][1]
            flag[i][j] = sentences[i][j][2]
            if labels:
                label[i][j] = sentences[i][j][3]
    if labels:
        return batch, pos_tag, flag, label, np.array(sequence_length)
    return batch, pos_tag, flag, np.array(sequence_length)

def generate_coefficients(labels_matrix, sequence_lenght, role_dict, null_coefficient=0.2):
    coefficients = np.array([1 for _ in range(sum(sequence_lenght))], dtype=np.float32)
    i = 0
    k = 0
    for row in labels_matrix:
        labels = 0
        for index in row:
            if labels >= sequence_lenght[k]:
                break
            if role_dict['_'] == index:
                coefficients[i] = null_coefficient
            i += 1
            labels += 1
        k += 1
    return coefficients

def global_epoch(file_path, update=None):
    if not update:
        try:
            f = open(file_path, 'r', encoding='utf8')
            val = int(f.read())
            f.close()
            return val
        except:
            f = open(file_path, 'w', encoding='utf8')
            f.write(str(0))
            f.close()
            return 0
    else:
        f = open(file_path, 'w', encoding='utf8')
        f.write(str(update))
        f.close()

def load_test(sentences, word2id, pos_dict):
    dataset = []
    for sentence in sentences:
        s = []
        predicates_flag = []
        for word in sentence:
            lemma = word[2]
            pos = word[4]
            #predicate = word[13].replace('\n', '')
            predicate = word[13]    # modificato
            val = []
            if lemma in word2id:
                val.append(word2id[lemma])
            else:
                val.append(word2id['unk'])
            val.append(pos_dict[pos])
            s.append(val)
            predicates_flag.append(0) if predicate == '_' else predicates_flag.append(1)
        indexes = [i for i in range(len(predicates_flag)) if predicates_flag[i] == 1]
        if not indexes:
            r = []
            pred_flag = [0 for _ in range(len(s))]
            for i in range(len(pred_flag)):
                # first element of the tuple: id of the lemma
                # second element of the tuple: id of the pos tag
                # third element of the tuple: 0 if the predicate is _, 1 otherwise
                r.append((s[i][0], s[i][1], pred_flag[i]))
            dataset.append([r])
        else:
            sent = []
            for index in indexes:
                r = []
                pred_flag = [0 for _ in range(len(s))]
                pred_flag[index] = 1
                for i in range(len(pred_flag)):
                    # first element of the tuple: id of the lemma
                    # second element of the tuple: id of the pos tag
                    # third element of the tuple: 0 if the predicate is _, 1 otherwise
                    r.append((s[i][0], s[i][1], pred_flag[i]))
                sent.append(r)
            dataset.append(sent)
    return dataset

def convert(predictions, index2role):
    prediction = predictions[0]
    result = []
    for pred in prediction:
        result.append(index2role[pred])
    return result

def no_flag(flags_matrix):
    for row in flags_matrix:
        for column in row:
            if column == 1:
                return False
    return True

