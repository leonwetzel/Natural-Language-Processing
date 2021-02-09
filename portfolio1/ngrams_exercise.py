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
from nltk import ngrams, word_tokenize
from nltk.text import Text
from collections import Counter
import fileinput, glob 

n = 2   # ngram size

corpus = glob.glob('presidential/roosevelt/*.txt')   # text to analyze, replace with path to one of the corpus subdirectories
allngrams = []
for sentence in fileinput.input(files=corpus):
        # lower-case input, do basic tokenization, create ngrams of length n
        sentencegrams = ngrams(word_tokenize(sentence.lower()),n)
        words = word_tokenize(sentence.lower())
        allngrams.extend(list(sentencegrams))
        
# print most frequent ngrams sorted by frequency
nbest = 25 # modify if you want to see more results 
for (key,val) in Counter(allngrams).most_common() :
    if (nbest >= 0) :
            print(key,val)
    nbest -= 1 
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

corpus = glob.glob('*.txt') # replace with the path to directory with text files 

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


