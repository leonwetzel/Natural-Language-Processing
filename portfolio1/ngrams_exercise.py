# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # N-gram statistics 
#
# Make sure to install the [nltk library](http://www.nltk.org/install.html)  and download the [Corpus of Presidential Speeches](http://www.thegrammarlab.com/?nor-portfolio=corpus-of-presidential-speeches-cops-and-a-clintontrump-corpus)  for experiments.
#
# Use the script below to compare (manually or by adding code) the 25  most frequent **bigrams** and **trigrams** for two presidents of your choice. 
#
# What are the different and common n-grams? You may ignore n-grams containing punctuation symbols.
#

# +
import string
from nltk import ngrams, word_tokenize
from nltk.text import Text
from collections import Counter
import fileinput, glob


def generate(corpus, n=2, nbest=25):
    allngrams = []
    for sentence in fileinput.input(files=corpus):
            # lower-case input, do basic tokenization, create ngrams of length n
            sentencegrams = ngrams(word_tokenize(sentence.lower()),n)
            words = word_tokenize(sentence.lower())
            allngrams.extend(list(sentencegrams))

    # print most frequent ngrams sorted by frequency
    counts = Counter(allngrams).most_common()
    output = []
    for (key,val) in counts:
        if nbest >= 0 and not any(p in " ".join(key) for p in string.punctuation):
            # print(" ".join(key))
            output.append(key)
        nbest -= 1
    return output

bigrams1 = generate(glob.glob('presidential/roosevelt/*.txt'), 2, 100)
bigrams2 = generate(glob.glob('presidential/fdroosevelt/*.txt'), 2, 100)

trigrams1 = generate(glob.glob('presidential/roosevelt/*.txt'), 3, 100)
trigrams2 = generate(glob.glob('presidential/fdroosevelt/*.txt'), 3, 100)

intersection = list(set(bigrams1).intersection(set(bigrams2)))
print("Common bigrams:")
print(intersection)

intersection = list(set(trigrams1).intersection(set(trigrams2)))
print("Common trigrams:")
print(intersection)

difference = list(set(bigrams1).difference(set(bigrams2)))
print("Different bigrams:")
print(difference)

difference = list(set(trigrams1).difference(set(trigrams2)))
print("Different trigrams:")
print(difference)

# -

# # N-gram generation 
#
# You can run the code below to generate a random text based on a trigram language model trained on the files specified in **corpus**. 
#
# 1. Run the code to generate an example text 
#
# 2. Describe in some detail (but in at most approximately 100 words) how one can implement a generation program based on a trigram language model.

# +
import random, fileinput, glob 

corpus = glob.glob('presidential/roosevelt/*.txt') # replace with the path to directory with text files

seed = random.randint(0,1000)
allwords = []
# read input from files given as command-line argument
for sentence in fileinput.input(files=corpus):
        # lower-case input, do basic tokenization,
        words = word_tokenize(sentence.lower())
        allwords.extend(words)

alltext = Text(allwords)
generated = alltext.generate(random_seed=seed)
# -


