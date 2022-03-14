from numpy import argmax
from math import log, exp

def create_table(words, viterbi, tags):
  table = {}
  for tag in tags:
    table[tag] = {}
    for word in words:
      if viterbi:
        table[tag][word] = (float(0),'')
      else:
        table[tag][word] = float(0)
  return table


def get_words_sentence(sent):
    words = []

    for i in range(len(sent)):
        words.append(sent[i]['form'])

    return words


def get_actual_pos(sentence):
    actual_pos = []
    for word in sentence:
        actual_pos.append(word['upos'])

    return actual_pos

def viterbi(sents, transitions, emissions, tags):
    all_sentences_preds = []
    all_sentences_actual = []

    for sent in sents:
        words = get_words_sentence(sent) # get all words in the sentence.
        viterbi_t = create_table(words, True, tags)

        # Iterate through the words in the sentence.
        for w in range(len(sent)):
            word = words[w]

            # Iterate through the tags.
            for tag in tags:

                # Initial step.
                if w == 0:
                    viterbi_t[tag][word] = (exp(log(transitions['<s>'].prob(tag)) + log(emissions[tag].prob(word))), '<s>')

                # If not the first word then find tag giving highest probability (max).
                else:
                    # intialise trellis.
                    trellis = []
                    previous_tags = []

                    # Iterate through the previous tags to consider the step from ti-1 to t. Recursive step.
                    for tPre in tags:
                        prob = exp(log(viterbi_t[tPre][words[w - 1]][0]) + log(transitions[tPre].prob(tag)) + log(emissions[tag].prob(word)))
                        # if prob is 0 then set it to the least possible exponention value (avoid math error).
                        if (prob == 0):
                          prob = 5e-324
                        trellis.append(prob)
                        previous_tags.append(tPre)

                    # Store the best (max) tag on the viterbi table.
                    viterbi_t[tag][word] = (trellis[argmax(trellis)], previous_tags[argmax(trellis)])

        # Final case.
        final = []
        for tPre in tags:
            final.append((exp(log(viterbi_t[tPre][word][0]) + log(transitions[tPre].prob('</s>'))), tPre))
        final_value = max(final)

        # Backtrack to find tags.
        tagged_word = backtrack(viterbi_t, words, final_value)

        # Store predictions
        all_sentences_preds.append(tagged_word)
        all_sentences_actual.append(get_actual_pos(sent))

    return all_sentences_preds, all_sentences_actual

def backtrack(viterbi_table, words, last_value):
  predicted_pos = [] # store predicted pos.
  prev_max_tag = last_value[1]

  # Iterate through the words in reverse.
  for w in reversed(words):
    predicted_pos.append(prev_max_tag)
    prev_max_tag = viterbi_table[prev_max_tag][w][1]

  predicted_pos.reverse() # reverse list.
  return predicted_pos