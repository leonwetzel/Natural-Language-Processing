#!/usr/bin/env python3
import re
from collections import Counter

from sklearn.metrics import accuracy_score


DEV_DATA = "lassysmall/nl_lassysmall-ud-dev.conllu"
TEST_DATA = "lassysmall/nl_lassysmall-ud-test.conllu"
TRAIN_DATA = "lassysmall/nl_lassysmall-ud-train.conllu"

TAG_INDEX = 3
WORD_INDEX = 1


def main():
    """
    Implement a baseline system for your corpus that assigns each word its most frequent PoS tag.
    Empty lines and lines starting with a # can be ignored.  Also lines starting with a code that
    is not an integer (i.e. 13.1 or 13-14) can be ignored. Data is tab-separated.
    The second column is the word, the fourth column the POS-tag. All other columns can be ignored.
    Collect statistics for the most frequent PoS per word from the *-train.conllu file and compute
    the performance of the baseline method on the *-dev.conllu. Dealing with unknown words requires
    special attention. Report your accuracy for the 2 versions of the baseline: one that has the
    simplemest strategy for handling unknowns and scores for an improved version as suggested in the book.

    Returns
    -------

    """
    with open(TRAIN_DATA, 'r', encoding='utf-8') as F:
        corpus = F.readlines()

    with open(DEV_DATA, 'r', encoding='utf-8') as F:
        dev_corpus = F.readlines()

    training_data = extract(corpus)
    dev_data = extract(dev_corpus)
    model = train(training_data)

    tag_counts = count_tags(training_data)
    # print({k: v for k, v in sorted(tag_counts.items(), key=lambda item: item[1])})

    # exit()

    target_values, predictions = [], []
    for word, target in dev_data:
        target_values.append(target)
        try:
            predictions.append(model[word])
        except KeyError:
            predictions.append('NOUN')

    print(accuracy_score(target_values, predictions))


def extract(corpus):
    """Load information from .conllu files.

    Parameters
    ----------
    corpus : iterable
        Collection of strings.

    Returns
    -------
    mappings : iterable
        Word-tag pairs in an iterable.
    """
    mappings = []
    for line in corpus:
        if re.match(r"^(#)|^([0-9]\.)", line) or not line.strip():
            continue
        else:
            row = line.split('\t')[:4]
            word, tag = row[WORD_INDEX], row[TAG_INDEX]
            mappings.append((word, tag))  # add pair to mappings
    return mappings


def train(training_data):
    """Trains the model on a given data set.

    Parameters
    ----------
    training_data

    Returns
    -------

    """
    counts = Counter(training_data)
    model = {}
    # sort counts by lowest occurrences, up to most frequent.
    # this allows higher frequencies to overwrite related
    # values in the model
    for pair, _ in counts.most_common()[:-len(counts)-1:-1]:
        word, tag = pair
        model[word] = tag

    return model


def count_tags(data):
    """Count tags from the data.

    Parameters
    ----------
    data : iterable
        List containing tuples, which consist of a word and a POS tag.
    Returns
    -------

    """
    counts = {}
    for _, tag in data:
        try:
            counts[tag] += 1
        except KeyError:
            counts[tag] = 1
    return counts


if __name__ == '__main__':
    main()
