#! /usr/bin/env python3

# Hidden Markov Models in Python
# Katrin Erk, March 2013 updated March 2016
# Modifications Feb 2020 (GB): read conllu UD treebanks, evaluate on dev set, replace taking products of probs with sums of logs, back-off for bigram pos tags
# add baseline function for unknown words
#
# This HMM addresses the problem of part-of-speech tagging. It estimates
# the probability of a tag sequence for a given word sequence as follows:
#
# Say words = w1....wN
# and tags = t1..tN
#
# then
# P(tags | words) is_proportional_to  product P(ti | t{i-1}) P(wi | ti)
#
# To find the best tag sequence for a given sequence of words,
# we want to find the tag sequence that has the maximum P(tags | words)
# import nltk
import argparse
import collections

from numpy import log

parser = argparse.ArgumentParser()
parser.add_argument("train_file", type=str, help="Name of the CoNLL-U file with the gold training data.")
parser.add_argument("evaluation_file", type=str, help="Name of the CoNLL-U file with the evaluation data.")
parser.add_argument("--verbose", "-v", default=False, action="store_true",
                    help="Print GOLD PREDICTED WORD info for tagging errors.")
args = parser.parse_args()

# Estimating P(wi | ti) from corpus data using Maximum Likelihood Estimation (MLE):
# P(wi | ti) = count(wi, ti) / count(ti)
#
# We add an artificial "start" tag at the beginning of each sentence, and
# We add an artificial "end" tag at the end of each sentence.
# So we start out with the brown tagged sentences,
# add the two artificial tags,
# and then make one long list of all the tag/word pairs.


# read the training data, store various conditional probabilities

# tags from UD, START and END added to mark start/end of sentence

tags = ['AUX', 'ADV', 'ADP', 'ADJ', 'DET', 'PUNCT', 'NUM', 'PRON', 'NOUN', 'VERB', 'X', 'CCONJ', 'SCONJ', 'PROPN',
        'SYM', 'INTJ', 'START', 'END']

lex_count = {}
known_word = {}
pos_count = collections.defaultdict(int)
bigram_count = {}
cpd_tagwords = {}
cpd_tags = {}

for tag in tags:
    lex_count[tag] = collections.defaultdict(int)
    cpd_tagwords[tag] = collections.defaultdict(int)
    bigram_count[tag] = collections.defaultdict(int)
    cpd_tags[tag] = collections.defaultdict(lambda: 0.00001)  # stupid back-off over tag bigrams
# i.e some tags are very infrequent, so make sure P(T2|T1) is never 0 to prevent errors when computing log probs

with open(args.train_file, encoding='utf-8') as training_corpus:
    for line in training_corpus:
        # print(line,end="")
        if line.startswith('# text'):
            prevpos = 'START'
            pos_count['START'] += 1
        elif line == '\n':
            bigram_count[prevpos]['END'] += 1
        elif not line.startswith('#'):
            fields = line.strip().split('\t')
            if fields[0].isdigit():  # skip 12.1 cases (ellipsis nodes)
                word = fields[1]
                pos = fields[3]
                lex_count[pos][word] += 1
                known_word[word] = 1
                pos_count[pos] += 1
                bigram_count[prevpos][pos] += 1
                prevpos = pos

            # compute the conditional probabilities for P(W|T) and P(T_n|T_n-1)
for tag in lex_count:
    total = pos_count[tag]
    if total > 0:  # start, end, error are special
        for (word, count) in lex_count[tag].items():
            cpd_tagwords[tag][word] = count / pos_count[tag]
for tag in bigram_count:
    total = pos_count[tag]
    if total > 0:
        for (tag2, count) in bigram_count[tag].items():
            cpd_tags[tag][tag2] = count / pos_count[tag]

# print("The probability of an adjective (ADJ) being 'nieuw' is", cpd_tagwords["ADJ"]["nieuw"])

# print("If we have just seen 'DET', the probability of 'NOUN' is", cpd_tags["DET"]["NOUN"])


# keep track of unknowns for debugging.
unknown_word = {}


# Heuristics to make sure P(W|T) > 0 for at least some tags T 
def unknown_word_guesser(word):
    cpd_tagwords['NOUN'][word] = 0.0001

    cpd_tagwords['PROPN'][word] = 0.0001
    cpd_tagwords['PUNCT'][word] = 0.0001
    cpd_tagwords['VERB'][word] = 0.0001
    # next two lines are for bookkeeping and informative error messages only
    known_word[word] = 1
    unknown_word[word] = 1


#####
# Viterbi:
# If we have a word sequence, what is the best tag sequence?
#
# The method above lets us determine the probability for a single tag sequence.
# But in order to find the best tag sequence, we need the probability
# for _all_ tag sequences.
# What Viterbi gives us is just a good way of computing all those many probabilities
# as fast as possible.

# what is the list of all tags?

def viterbi(sentence):
    distinct_tags = tags

    # sentence = ["Ik", "wil", "een", "boek", "schrijven"]
    # sentence = ["I", "saw", "her", "duck" ]
    sentlen = len(sentence)

    # for each step i in 1 .. sentlen,
    # store a dictionary that maps each tag X
    # to the probability of the best tag sequence of length i that ends in X
    viterbi_sequence = []

    # backpointer:
    # for each step i in 1..sentlen,
    # store a dictionary
    # that maps each tag X
    # to the previous tag in the best tag sequence of length i that ends in X
    backpointer = []

    first_viterbi = {}
    first_backpointer = {}
    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START":
            continue
        word = sentence[0]
        if not (word in known_word):
            unknown_word_guesser(word)  # estimate cpd_tagwords for this word
        if cpd_tagwords[tag][word]:
            first_viterbi[tag] = log(cpd_tags["START"][tag]) + log(cpd_tagwords[tag][word])
            first_backpointer[tag] = "START"

    # print(first_viterbi)
    # print(first_backpointer)

    viterbi_sequence.append(first_viterbi)
    backpointer.append(first_backpointer)

    currbest = max(first_viterbi.keys(), key=lambda tag: first_viterbi[tag])
    # print( "Word", "'" + sentence[0] + "'", "current best two-tag sequence:", first_backpointer[ currbest], currbest)
    # print( "Word", "'" + sentence[0] + "'", "current best tag:", currbest)

    for wordindex in range(1, len(sentence)):
        this_viterbi = {}
        this_backpointer = {}
        prev_vi = viterbi_sequence[-1]
        word = sentence[wordindex]

        for tag in distinct_tags:
            # don't record anything for the START tag
            if tag == "START":
                continue

            # if this tag is X and the current word is w, then
            # find the previous tag Y that maximizes
            # prev_vi[ Y ] * P(X | Y) * P( w | X)
            # The following command has the same notation
            # that you saw in the sorted() command.
            if word in known_word:
                True
            elif word.lower() in known_word:
                word = word.lower()
            else:  # unknown word, back off
                unknown_word_guesser(word)
            if cpd_tagwords[tag][word]:
                best_prev = max(prev_vi.keys(),
                                key=lambda prevtag: prev_vi[prevtag] + log(cpd_tags[prevtag][tag]) + log(
                                    cpd_tagwords[tag][word]))
                this_viterbi[tag] = prev_vi[best_prev] + log(cpd_tags[best_prev][tag]) + log(cpd_tagwords[tag][word])

                # best_prev = max(prev_vi.keys(), key = lambda prevtag: prev_vi[prevtag] * cpd_tags[prevtag][tag] * cpd_tagwords[tag][word])
                #	this_viterbi[ tag ] = prev_vi[ best_prev] * cpd_tags[best_prev][tag] * cpd_tagwords[tag][word]
                this_backpointer[tag] = best_prev

        currbest = max(this_viterbi.keys(), key=lambda tag: this_viterbi[tag])
        # print( "Word", "'" + sentence[ wordindex] + "'", "current best two-tag sequence:", this_backpointer[ currbest], currbest)
        # print( "Word", "'" + sentence[ wordindex] + "'", "current best tag:", currbest)
        # done with all tags in this iteration so store the current viterbi step
        viterbi_sequence.append(this_viterbi)
        backpointer.append(this_backpointer)

    # done with all words in the sentence.
    # now find the probability of each tag
    # to have "END" as the next tag,
    # and use that to find the overall best sequence
    prev_viterbi = viterbi_sequence[-1]
    best_previous = max(prev_viterbi.keys(), key=lambda prevtag: prev_viterbi[prevtag] + log(cpd_tags[prevtag]["END"]))

    prob_tagsequence = prev_viterbi[best_previous] + log(cpd_tags[best_previous]["END"])

    # best tagsequence: we store this in reverse for now, will invert later
    best_tagsequence = ["END", best_previous]
    # invert the list of backpointers
    backpointer.reverse()

    # go backwards through the list of backpointers
    # (or in this case forward, because we have inverter the backpointer list)
    # in each case:
    # the following best tag is the one listed under
    # the backpointer for the current best tag
    current_best_tag = best_previous
    for bp in backpointer:
        best_tagsequence.append(bp[current_best_tag])
        current_best_tag = bp[current_best_tag]

    best_tagsequence.reverse()
    # no need to return START/END tag
    # print( "The sentence was:", end = " ")
    # for w in sentence: print( w, end = " ")
    # print("\n")
    # print( "The best tag sequence is:", end = " ")
    # for t in best_tagsequence: print (t, end = " ")
    # print("\n")
    return best_tagsequence[1:-1]


# print( "The probability of the best tag sequence is:", prob_tagsequence)

with open(args.evaluation_file) as evaluation_corpus:
    gold_tags = []
    predicted_tags = []
    all_words = []
    for line in evaluation_corpus:
        if (line.startswith('# text')):
            sentence = []
        elif (line == '\n'):
            # print(sentence)
            predicted = viterbi(sentence)
            # print(predicted)
            predicted_tags.extend(predicted)
            all_words.extend(sentence)
        elif (not (line.startswith('#'))):
            fields = line.strip().split('\t')
            if fields[0].isdigit():  # skip 13.1 (ellipsis) and 2-3 (multiword token) lines
                word = fields[1]
                pos = fields[3]
                gold_tags.append(pos)
                sentence.append(word)
    correct = 0
    corpus_size = len(gold_tags) - 1
    for gold, predicted, word in zip(gold_tags, predicted_tags, all_words):
        if gold == predicted:
            correct += 1
        elif args.verbose:
            if word in unknown_word:
                print('{} {} {} (UNK)'.format(gold, predicted, word))
            elif word in known_word:
                print('{} {} {}'.format(gold, predicted, word))
            else:
                print('{} {} {} (LC)'.format(gold, predicted, word))

    print('accuracy is {} ({}/{})'.format(correct / corpus_size, correct, corpus_size))
