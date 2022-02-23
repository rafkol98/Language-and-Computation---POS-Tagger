from io import open
import numpy as np
import pandas as pd
from conllu import parse_incr
from nltk import FreqDist, WittenBellProbDist

treebank = {}
treebank['en'] = 'UD_English-EWT/en_ewt'
treebank['es'] = 'UD_Spanish-GSD/es_gsd'
treebank['nl'] = 'UD_Dutch-Alpino/nl_alpino'


def train_corpus(lang):
    return treebank[lang] + '-ud-train.conllu'


def test_corpus(lang):
    return treebank[lang] + '-ud-test.conllu'


# Remove contractions such as "isn't".
def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]


# Choose language.
lang = 'en'

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
print(len(train_sents), 'training sentences')
print(len(test_sents), 'test sentences')

tags2int = {
    "<s>": 1, "ADJ": 2, "ADP": 3, "ADV": 4, "AUX": 5, "CCONJ": 6, "DET": 7,
    "INTJ": 8, "NOUN": 9, "NUM": 10, "PART": 11, "PRON": 12, "PROPN": 13,
    "PUNCT": 14, "SCONJ": 15, "SYM": 16, "VERB": 17, "X": 18, "</s>": 19
}

ti = tags2int.get("ADJ")
print(ti)

def getTags(sents):
    tags = []
    for sent in sents:
        # tags.append(Start)
        for token in sent:
            tags.append(token['upos']);

    return tags

def getPosTagsOfSentence(sent):
    tags = ["<s>"]  # start-of-sentence marker.

    # enter all the pos tags of all the words.
    for i in range(len(sent)):
        tags.append(sent[i]['upos'])

    tags.append("/<s>")  # end-of-sentence marker.
    return tags


def create_transition_table(sents):
    # Create a 19 x 19 table.
    c_table = np.zeros((len(tags2int), len(tags2int)))
    transition = []
    tags = getTags(sents) # get tags.

    for sent in sents:
        pos_sent = getPosTagsOfSentence(sent)

        for i in range(len(pos_sent) - 1):
            ti = tags2int.get(pos_sent[i], 0)
            ti1 = tags2int.get(pos_sent[i + 1], 0)

            transition.append((pos_sent[i], pos_sent[i + 1]))
            c_table[ti - 1][ti1 - 1] += 1  # increment by 1.

    c_table[tags2int["</s>"] - 1][tags2int["<s>"] - 1] = len(sents) - 1  # account for the transition from </s> to <s>

    smoothed_transition = {}
    for tag in tags:
        taged_tag = [t for (pt, t) in transition if pt == tag]
        smoothed_transition[tag] = WittenBellProbDist(FreqDist(taged_tag), bins=1e5)

    return c_table, smoothed_transition


def create_emmisions_dict(sents):
    emissions = []
    for sent in sents:
        for token in sent:
            emissions.append((token['upos'], token['form']))

    smoothed = {}
    tags = set([t for (t, _) in emissions])
    for tag in tags:
        # print(smoothed)
        words = [w for (t, w) in emissions if t == tag]
        print(words)
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)


# print(smoothed['AUX'].prob('is')) # example of how to get the probability --> pass in word.

# emmisions = create_emmisions_dict(test_sents)
table, transitions = create_transition_table(test_sents)
print(transitions)
print(transitions.get("<s>").prob('PROPN'))

