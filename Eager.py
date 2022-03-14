def eager(sents, transitions, emissions, tags):
    # Store predictions and actuals.
    all_sentences_preds = []
    all_sentences_actual = []

    # Iterate through the sentences.
    for sent in sents:
        preds_current_sent = []
        actual_current_sent = []

        previous_tag = '<s>'  # set initial tag.
        # Iterate through the words in the current sentence.
        for word in sent:
            pos_probs = {}  # Store the prediction probabilities for each tag.

            # Iterate through the tags calculating the probability for each tag.
            for tag in tags:
                pos_probs[tag] = transitions[previous_tag].prob(tag) * emissions[tag].prob(word['form'])

            # Assign the most likely tag - the one with the highest score in pos_probs.
            predicted_tag = max(pos_probs, key=pos_probs.get)

            # Add predicted and actual tag to their respective lists.
            preds_current_sent.append(predicted_tag)
            actual_current_sent.append(word['upos'])

            # Assign the predicted_tag to the previous_tag.
            previous_tag = predicted_tag

        all_sentences_preds.append(preds_current_sent)
        all_sentences_actual.append(actual_current_sent)

    return all_sentences_preds, all_sentences_actual