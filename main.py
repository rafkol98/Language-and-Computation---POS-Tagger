from io import open
from conllu import parse_incr
from nltk import FreqDist, WittenBellProbDist, bigrams
from random import sample
import sys

from Eager import eager
from ForwardBackward import forward_backward
from Viterbi import viterbi

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


def get_tags(sents, transitions_bool):
    ''' Return the tags and words that are present in all the sentences passed in.'''
    tags = []
    for sent in sents:
        # tags.append(Start)
        for token in sent:
            tags.append(token['upos'])

    # if its called from the transitions table, then append starting and end tags.
    if transitions_bool:
        tags.append("<s>")
        tags.append("</s>")

    return list(set(tags))


def get_pos_tags_sentence(sent, include_start_end=False):
    '''Get all the POS tags of a specific sentence passed in.'''
    tags = []

    if include_start_end:
        tags.append("<s>")  # start-of-sentence marker.

    # enter all the pos tags of all the words.
    for i in range(len(sent)):
        tags.append(sent[i]['upos'])

    if include_start_end:
        tags.append("</s>")  # end-of-sentence marker.

    return tags


def get_emissions_dict(sents):
    '''Returns a dictionary with the smoothed out emissions.'''
    emissions = []
    for sent in sents:
        for token in sent:
            emissions.append((token['upos'], token['form']))

    smoothed = {}
    tags = set([t for (t, _) in emissions])

    # Iterate through the tags and use WittenBellProbDist to smooth them.
    for tag in tags:
        words = [w for (t, w) in emissions if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

    return smoothed


def create_transition(sents):
    '''Returns a dictionary that holds the smoothed out transitions.'''
    transition = []
    tags = get_tags(sents, True)  # get tags.

    for sent in sents:
        pos_sent = get_pos_tags_sentence(sent, True)
        for i in range(len(pos_sent) - 1):
            transition.append((pos_sent[i], pos_sent[i + 1]))

    smoothed = {}
    for tag in tags:
        taged_tag = [t for (pt, t) in transition if pt == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(taged_tag), bins=1e5)

    # return transitions_smoothed out.
    return smoothed


def calculate_accuracy(preds, actual, tags):
    '''Calculate accuracy for passed in predictions and actuals'''
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

    print(f"accuracy: {correct_count / total_count}")
    # plot_conf_mx(all_predictions, all_actuals, tags) confusion matrix used for the report


# Plot confusion matrix - used for the report.
# def plot_conf_mx(y_preds, y_actual, tags):
#
#     # tags = get_tags(test_sents, False) # Get tags, to be used for the confusion matrix.
#
#     plt.figure(figsize = (20,16))
#
#     conf_mx = confusion_matrix(y_actual, y_preds)
#
#     hm = sns.heatmap(conf_mx, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, cmap="Greens", xticklabels=sorted(tags), yticklabels=sorted(tags))
#     plt.show()

def evaluate_language(lang):
    '''Evaluate a passed in language running all three algorithms.'''

    # Check that the language passed in is one of the ones we contain their corpus.
    if lang in ('en', 'es', 'nl', 'pl'):
        train_sents = conllu_corpus(train_corpus(lang))
        test_sents = conllu_corpus(test_corpus(lang))

        # resample the train_sent to the length of the smallest corpus. This was found out by experimentation.
        train_sents = sample(train_sents, 12264)
        # resample the testing sent to the length of the smallest corpus. This was found out by experimentation.
        test_sents = sample(test_sents, 426)

        transitions = create_transition(train_sents)  # Create transitions table using training sentences.
        emissions = get_emissions_dict(train_sents)  # Create emissions table using training sentences.
        tags = get_tags(test_sents, False)  # Get tags in the tagset of the language.

        print("\nEAGER")
        print("Training - Eager")
        preds_eager_train, actuals_eager_train = eager(train_sents, transitions, emissions, tags)
        calculate_accuracy(preds_eager_train, actuals_eager_train, tags)

        print("Testing - Eager")
        preds_eager_test, actuals_eager_test = eager(test_sents, transitions, emissions, tags)
        calculate_accuracy(preds_eager_test, actuals_eager_test, tags)

        print("\nVITERBI")
        print("Training - Viterbi")
        preds_viterbi_train, actuals_viterbi_train = viterbi(train_sents, transitions, emissions, tags)
        calculate_accuracy(preds_viterbi_train, actuals_viterbi_train, tags)

        print("Testing - Viterbi")
        preds_viterbi_test, actuals_viterbi_test = viterbi(test_sents, transitions, emissions, tags)
        calculate_accuracy(preds_viterbi_test, actuals_viterbi_test, tags)

        print("\nFORWARD-BACKWARD")
        print("Training - Forward-Backward")
        preds_local_train, actuals_local_train = forward_backward(train_sents, transitions, emissions, tags)
        calculate_accuracy(preds_local_train, actuals_local_train, tags)

        print("Testing - Forward-Backward")
        preds_local_test, actuals_local_test = forward_backward(test_sents, transitions, emissions, tags)
        calculate_accuracy(preds_local_test, actuals_local_test, tags)


# RUNNING THE PROGRAM.
# Check if the user provided arguments. First argument is the filename.
if len(sys.argv) == 2:
    lang = sys.argv[1]
    # check if its one of the languages in our corpora.
    if (lang in ('en', 'es', 'nl', 'pl')):
        evaluate_language(lang)
else:
    print(
        "Please run like this: python main.py <'en'|'<es>'|'<nl>'|'<pl>' \nPlease also look at the ReadMe file for further instructions.\n")
