from io import open
import numpy as np
import pandas as pd
from conllu import parse_incr
from nltk import FreqDist, WittenBellProbDist, bigrams
from numpy import argmax
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

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

# Return the tags and words that are present in all the sentences passed in.
def get_tags(sents, transitions_bool):
    tags = []
    for sent in sents:
        # tags.append(Start)
        for token in sent:
            tags.append(token['upos'])

    if transitions_bool:
      # Append starting and end tags.
      tags.append("<s>")
      tags.append("</s>")

    return list(set(tags))

def get_words(sents):
    words = []
    for sent in sents:
        for token in sent:
            words.append(token['form'])

    return list(set(words))

# Get POS tags of a specific sentence passed in.
def get_pos_tags_sentence(sent):
    tags = ["<s>"]  # start-of-sentence marker.

    # enter all the pos tags of all the words.
    for i in range(len(sent)):
        tags.append(sent[i]['upos'])

    tags.append("</s>")  # end-of-sentence marker.
    return tags

def get_emissions_dict(sents):
    emissions = []
    for sent in sents:
        for token in sent:
            emissions.append((token['upos'], token['form']))

    smoothed = {}
    tags = set([t for (t, _) in emissions])
    for tag in tags:
        words = [w for (t, w) in emissions if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

    return smoothed


def create_transition_table(sents):
    transition = []
    tags = get_tags(sents, True)  # get tags.

    for sent in sents:
        pos_sent = get_pos_tags_sentence(sent)
        for i in range(len(pos_sent) - 1):
            transition.append((pos_sent[i], pos_sent[i + 1]))

    smoothed = {}
    for tag in tags:
        # print(tag)
        taged_tag = [t for (pt, t) in transition if pt == tag]
        # print(taged_tag)
        smoothed[tag] = WittenBellProbDist(FreqDist(taged_tag), bins=1e5)

    # Initialise a transition table with zeros.
    transitions_table = np.zeros((len(tags), len(tags)))

    for t in range(len(tags)):
        for ti in range(len(tags)):
            tag = tags[t]
            second_tag = tags[ti]

            transitions_table[t][ti] = smoothed.get(tag).prob(second_tag)

    # It does not matter that </s> is not at the end.
    transitions_df = pd.DataFrame(data=transitions_table, columns=tags, index=tags)

    return transitions_df


def create_emmissions_hashmap(sents):
    emissions = []
    for sent in sents:
        for token in sent:
            emissions.append((token['upos'], token['form']))

    smoothed = {}
    tags = set([t for (t, _) in emissions])
    for tag in tags:
        words = [w for (t, w) in emissions if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

    # Get tags.
    tags = get_tags(sents, False)

    words = get_words(sents)

    emissions_hash = {}
    for tag in tags:
        emissions_hash[tag] = {}
        for word in words:
            emissions_hash[tag][word] = smoothed.get(tag).prob(word)

    return emissions_hash


def eager(tagset, word, previous_tag):
    pos_probs = {}

    for tag in tagset:
        if tag == '<s>' or tag == '</s>':
            continue

        if word not in emissions:
            pos_probs[tag] = transitions[tag][previous_tag]
        else:
            pos_probs[tag] = transitions[tag][previous_tag] * emissions[tag][word]

    return max(pos_probs, key=pos_probs.get)

def predict_pos(sents):
  tags = get_tags(sents, True)

  all_sentences_preds = []
  all_sentences_actual = []

  for sent in sents:
    preds_current_sent = []
    actual_current_sent = []

    previous_tag = '<s>' # set initial tag.
    for word in sent:
      # Predict tag.
      predicted_tag = eager(tags, word['form'], previous_tag)
      # Add predicted and actual tag to their respective lists.
      preds_current_sent.append(predicted_tag)
      actual_current_sent.append(word['upos'])
      # Assign the predicted_tag to the previous_tag.
      previous_tag = predicted_tag

    all_sentences_preds.append(preds_current_sent)
    all_sentences_actual.append(actual_current_sent)
  return all_sentences_preds, all_sentences_actual

def calculate_accuracy(preds, actual):
  all_predictions, all_actuals = [], []

  total_count = 0
  correct_count = 0

  for i in range(len(preds)):
    # ASSUMPTION THAT PREDS AND ACTUAL SENTENCES HAVE OF COURSE THE
    # SAME LENGTH/SIZE.
    sentence = preds[i]
    for x in range(len(sentence)):
      total_count += 1
      all_predictions.append(preds[i][x])
      all_actuals.append(actual[i][x])

      if (preds[i][x] == actual[i][x]):
        correct_count += 1


  print(f"accuracy: {correct_count/total_count}")
  return all_predictions, all_actuals

# CONFUSION MATRIX USED FOR THE REPORT.
# def plot_conf_mx(y_actual, y_preds):
#     plt.figure(figsize = (12,12))
#
#     conf_mx = confusion_matrix(y_actual, y_preds)
#
#     hm = sns.heatmap(conf_mx, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
#     plt.show()

# TRAINING
transitions = create_transition_table(train_sents)
emissions = create_emmissions_hashmap(train_sents)

predictions, actuals = predict_pos(test_sents)
all_preds, all_actuals = calculate_accuracy(predictions, actuals)
# plot_conf_mx(all_preds, all_actuals) CONFUSION MATRIX USED FOR THE REPORT.



