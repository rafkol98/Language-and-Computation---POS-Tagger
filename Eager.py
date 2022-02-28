from main import train_sents, test_sents, get_tags, create_transition_table, create_emmissions_hashmap
# from main import plot_conf_mx
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

transitions = create_transition_table(train_sents)
emissions = create_emmissions_hashmap(train_sents)
predictions, actuals = predict_pos(test_sents)
all_preds, all_actuals = calculate_accuracy(predictions, actuals)

# plot_conf_mx(all_preds, all_actuals) # CONFUSION MATRIX USED FOR THE REPORT.