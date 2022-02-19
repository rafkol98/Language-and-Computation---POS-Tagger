from io import open
import numpy as np
from conllu import parse_incr

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
		"<s>": 1, "ADJ": 2,"ADP": 3, "ADV": 4,"AUX": 5,"CCONJ": 6,"DET": 7,
		"INTJ": 8,"NOUN": 9,"NUM": 10,"PART": 11,"PRON": 12,"PROPN": 13,
		"PUNCT": 14,"SCONJ": 15,"SYM": 16,"VERB": 17, "X": 18, "</s>": 19
	}

ti = tags2int.get("ADJ")
print(ti)

# Illustration how to access the word and the part-of-speech of tokens.
def create_transition_table(sents):
	tags2int = {
		"<s>": 1, "ADJ": 2,"ADP": 3, "ADV": 4,"AUX": 5,"CCONJ": 6,"DET": 7,
		"INTJ": 8,"NOUN": 9,"NUM": 10,"PART": 11,"PRON": 12,"PROPN": 13,
		"PUNCT": 14,"SCONJ": 15,"SYM": 16,"VERB": 17, "X": 18, "</s>": 19
	}

	# Create a 19 x 19 table.
	c_table = np.zeros((len(tags2int), len(tags2int)))
	for sent in train_sents:
		pos_sent = getPosTagsOfSentence(sent)

		for i in range(len(pos_sent)-1):
			ti = tags2int.get(pos_sent[i])
			ti1 = tags2int.get(pos_sent[i + 1])
			c_table[ti][ti1] += 1

	return c_table


def getPosTagsOfSentence(sent):
	tags = ["<s>"] # start-of-sentence marker.

	# enter all the pos tags of all the words.
	for i in range(len(sent)):
		tags.append(sent[i]['upos'])

	tags.append("/<s>") # end-of-sentence marker.
	return tags

twod_tokens = create_transition_table(train_sents)












# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
