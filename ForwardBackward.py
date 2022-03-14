from Viterbi import create_table, get_words_sentence, get_actual_pos
from math import log, exp

from sys import float_info

# This effectively acts as probability 0 in the form of log probability.
min_log_prob = -float_info.max


# Adding a list of probabilities represented as log probabilities.
def logsumexp(vals):
    if len(vals) == 0:
        return min_log_prob
    m = max(vals)
    if m == min_log_prob:
        return min_log_prob
    else:
        return m + log(sum([exp(val - m) for val in vals]))


def forward(words, transitions, emissions, tags):
    '''The forward algorithm for the ForwardBackward algorithm.'''
    forward_t = create_table(words, False, tags)

    # Iterate through the words in the sentence.
    for w in range(len(words)):
        word = words[w]

        # Iterate through the tags.
        for tag in tags:
            # Initial step.
            if w == 0:
                forward_t[tag][word] = exp(log(transitions['<s>'].prob(tag)) + log(emissions[tag].prob(word)))

            # If not the first word
            else:
                trellis = []

                for tPre in tags:
                    try:
                        prob = log(forward_t[tPre][words[w - 1]]) + log(transitions[tPre].prob(tag)) + log(
                            emissions[tag].prob(word))
                    except:
                        prob = 5e-324
                    trellis.append(prob)

                forward_t[tag][word] = exp(logsumexp(trellis))

    # Final case - tag.
    final = []
    for t in tags:
        try:
            prob = log(forward_t[t][word]) + log(transitions[t].prob('</s>'))
        except:
            prob = 5e-324
        final.append(prob)
    final_value = exp(logsumexp(final))

    return forward_t, final_value


def backwards(words, transitions, emissions, tags):
    '''The backwards algorithm for the ForwardBackward algorithm.'''
    backwards_t = create_table(words, False, tags)

    # Iterate through the words in the sentence in reverse (using the indices).
    for w in range(len(words) - 1, -1, -1):
        word = words[w]

        # Iterate through the tags.
        for tag in tags:
            # Initial step (final word).
            if w == len(words) - 1:
                backwards_t[tag][word] = exp(log(transitions[tag].prob('</s>')))
            # If not the last word
            else:
                trellis = []
                future_word = words[w + 1]
                for tFut in tags:
                    try:
                        prob = log(backwards_t[tFut][words[w + 1]]) + log(transitions[tag].prob(tFut)) + log(
                            emissions[tFut].prob(future_word))
                    # if prob is 0 then set it to the least possible exponention value (avoid math error).
                    except:
                        prob = 5e-324
                    trellis.append(prob)

                backwards_t[tag][word] = exp(logsumexp(trellis))

    # Final step - first word.
    final = []
    for tFut in tags:
        try:
            prob = log(backwards_t[tFut][word]) + log(transitions['<s>'].prob(tFut)) + log(emissions[tFut].prob(word))
        except:
            prob = 5e-324
        final.append(prob)
    final_value = exp(logsumexp(final))

    return backwards_t, final_value


def forward_backward(sents, transitions, emissions, tags):
    '''The ForwardBackward algorithm. Uses the forward and backwards tables to perform local decoding and
    assign tags to each word in a sentence.'''
    all_sentences_preds = []
    all_sentences_actual = []
    # Iterate through the sentences.
    for sent in sents:
        # Get words of the sentence
        words = get_words_sentence(sent)
        forward_table, forward_last = forward(words, transitions, emissions, tags)
        backwards_table, backwards_last = backwards(words, transitions, emissions, tags)

        pos_preds = []
        # Iterate through the words in the sentence.
        for word in words:
            trellis = {}
            #  multiply alpha and beta together and return the maximum entry.
            for tag in tags:
                trellis[tag] = (forward_table[tag][word] * backwards_table[tag][word])

            max_tag = max(trellis, key=trellis.get)  # The tag assigned is the maximum one in the trellis.
            pos_preds.append(max_tag)

        # Append the prediction and actuals to the corresponding arrays (used for evaluation).
        all_sentences_preds.append(pos_preds)
        all_sentences_actual.append(get_actual_pos(sent))

    return all_sentences_preds, all_sentences_actual
