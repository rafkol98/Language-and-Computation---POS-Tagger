from io import open
import numpy as np
import pandas as pd
from conllu import parse_incr
from nltk import FreqDist, WittenBellProbDist, bigrams
from numpy import argmax

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


# Return the tags that are present in all the sentences passed in.
def get_tags(sents):
    tags = []
    for sent in sents:
        # tags.append(Start)
        for token in sent:
            tags.append(token['upos']);

    # Append starting and end tags.
    tags.append("<s>")
    tags.append("</s>")

    return set(tags)


# Get POS tags of a specific sentence passed in.
def get_pos_tags_sentence(sent):
    tags = ["<s>"]  # start-of-sentence marker.

    # enter all the pos tags of all the words.
    for i in range(len(sent)):
        tags.append(sent[i]['upos'])

    tags.append("</s>")  # end-of-sentence marker.
    return tags


def create_transition_table(sents):
    transition = []
    tags = get_tags(sents)  # get tags.

    for sent in sents:
        pos_sent = get_pos_tags_sentence(sent)
        for i in range(len(pos_sent) - 1):
            transition.append((pos_sent[i], pos_sent[i + 1]))

    smoothed_transition = {}
    for tag in tags:
        # print(tag)
        taged_tag = [t for (pt, t) in transition if pt == tag]
        # print(taged_tag)
        smoothed_transition[tag] = WittenBellProbDist(FreqDist(taged_tag), bins=1e5)

    return smoothed_transition


def create_emissions_dict(sents):
    emissions = []
    for sent in sents:
        for token in sent:
            emissions.append((token['upos'], token['form']))

    smoothed = {}
    tags = set([t for (t, _) in emissions])
    for tag in tags:
        # print(smoothed)
        print(tag)
        words = [w for (t, w) in emissions if t == tag]
        print(words)
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

    return smoothed


# print(smoothed['AUX'].prob('is')) # example of how to get the probability --> pass in word.
transitions = create_transition_table(test_sents)
print(transitions.get('<s>').prob('PRON'))

emissions = create_emissions_dict(test_sents)
print(emissions.get("PRON").prob("What"))

def choose_tags(sent):
    pos_tags = ['<s>'] # insert the initial tag.
    for i in range(1, len(sent)):
        word = sent[i-1]
        tag = pos_tags[i-1]
        # argmax only for transitions
        # Might be confusing that its i and not i-1.
        t = transitions[argmax(transitions.get(tag))]
        print("T")
        print(t)
        e = emissions.get(tag).prob(word)
        print("E")
        print(e)
        pos_tags[i] = t # insert t as the tag for the next word.
