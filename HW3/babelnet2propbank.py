from WSD.disambiguator import WordSenseDisambiguator
import data_preprocessing as dp
from pos_map import MAPPER
import tqdm

def create_pos_mapper():
    mapper = {}
    for m in MAPPER:
        mapper[m[0]] = m[1]
    return mapper

def convert_training_data_pos_tag(training_data, mapper, index2pos, disambiguator_pos2index):
    data = []
    for sentence in training_data:
        s = []
        for word in sentence:
            word = list(word)
            conll_pos_tag = index2pos[word[1]]
            semcor_pos_tag = mapper[conll_pos_tag]
            word[1] = disambiguator_pos2index[semcor_pos_tag]
            s.append(tuple(word))
        data.append(s)
    return data

def load_dataset(sentences, word2id):
    dataset = []
    index_pos = 0
    pos2index = {}
    for sentence in sentences:
        s = []
        for word in sentence:
            lemma = word[2]
            pos = word[4]
            predicate = word[13].replace('\n', '')
            if pos not in pos2index:
                pos2index[pos] = index_pos
                index_pos += 1
            # first element of the tuple: id of the lemma
            # second element of the tuple: id of the pos tag
            # third element of the tuple: 0 if the predicate is _, 1 otherwise
            if predicate == '_':
                s.append((word2id[lemma], pos2index[pos], 0)) if lemma in word2id else s.append((word2id['unk'], pos2index[pos], 0))
            else:
                s.append((word2id[lemma], pos2index[pos], 1)) if lemma in word2id else s.append((word2id['unk'], pos2index[pos], 1))
        dataset.append(s)
    return dataset, pos2index

def get_most_common_predicate(predicates_list):
    count = {}
    for predicate in predicates_list:
        try:
            count[predicate] += 1
        except:
            count[predicate] = 1
    m = float('-inf')
    val = None
    for sense in count:
        if count[sense] > m:
            m = count[sense]
            val = sense
    return val


DATA_DIR = 'SRLData/EN/'

pos_mapper = create_pos_mapper()
disambiguator = WordSenseDisambiguator()
sentences = dp.get_sentences(DATA_DIR + 'CoNLL2009-ST-English-train.txt')
training_data, pos2index = load_dataset(sentences, disambiguator.word2index)
training_data = convert_training_data_pos_tag(training_data, pos_mapper, {v: k for k, v in pos2index.items()}, disambiguator.pos2index)

k = 0
d = {}
index2sense = {v: k for k, v in disambiguator.sense2index.items()}

for sentence in tqdm.tqdm(training_data, 'Generating babelnet2propbank.txt'):

    labels_predicted = disambiguator.predict(sentence, input_labels=False)
    predicates = [word[13] for word in sentences[k]]
    senses = [index2sense[sense] for sense in labels_predicted[0]]

    for i in range(len(senses)):
        if senses[i] != 'UNK' and predicates[i] != '_':
            try:
                d[senses[i]].append(predicates[i])
            except:
                d[senses[i]] = []
                d[senses[i]].append(predicates[i])

    k += 1

f = open('babelnet2propbank.txt', 'w')
for sense, predicates_list in d.items():
    f.write(sense + '\t' + get_most_common_predicate(predicates_list) + '\n')
f.close()

