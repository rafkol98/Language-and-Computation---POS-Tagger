from io import open
import numpy as np
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


def getPosTagsOfSentence(sent):
    tags = ["<s>"]  # start-of-sentence marker.

    # enter all the pos tags of all the words.
    for i in range(len(sent)):
        tags.append(sent[i]['upos'])

    tags.append("/<s>")  # end-of-sentence marker.
    return tags


# Illustration how to access the word and the part-of-speech of tokens.
def create_transition_table(sents):
    # Create a 19 x 19 table.
    c_table = np.zeros((len(tags2int), len(tags2int)))
    for sent in sents:
        pos_sent = getPosTagsOfSentence(sent)

        for i in range(len(pos_sent) - 1):
            ti = tags2int.get(pos_sent[i], 0)
            ti1 = tags2int.get(pos_sent[i + 1], 0)

            c_table[ti - 1][ti1 - 1] += 1

    c_table[tags2int["</s>"] - 1][tags2int["<s>"] - 1] = len(sents) - 1  # account for the transition from </s> to <s>

    # TODO: have to add smoothing!!!!
    row_sums = c_table.sum(axis=1)
    normalized_table = c_table / row_sums[:, np.newaxis]
    return normalized_table


# Illustration how to access the word and the part-of-speech of tokens.
def create_emmisions_dict(sents):
	emissions = []
	# TODO: method passing in sents
	for sent in sents:
		for token in sent:
			emissions.append((token['upos'], token['form']))

	smoothed = {}
	tags = set([t for (t, _) in emissions])
	for tag in tags:
		words = [w for (t, w) in emissions if t == tag]
		smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

	# print(smoothed['AUX'].prob('is')) # example of how to get the probability --> pass in word.





emmisions = create_emmisions_dict(test_sents)
