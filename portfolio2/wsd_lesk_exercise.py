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

# # Word sense disambiguation
#
# These are the assignments for week 7, following [chapter 18](https://web.stanford.edu/~jurafsky/slp3/18.pdf) of J&M. There are three parts:
#
# 1. Implement code to disambiguate all content words in a text using the Lesk-algorithm (and WordNet).
# 2. Evaluate the performance of (1) on a 100-word text-snippet
# 3. Investigate to what extent word-embeddings capture the predominant (most frequent) sense of a word.
#
# ## 7.1 Lesk
#
# Complete the code below to make a model that processes a text and disambiguates every content word. We use the [Lesk](https://www.nltk.org/howto/wsd.html) implementation from the NLTK toolkit, as well as the [tokenization](https://pythonspot.com/tokenizing-words-and-sentences-with-nltk/) and [stop word filtering](https://pythonspot.com/nltk-stop-words/) modules. 
#

# +
import nltk

nltk.download('wordnet') # you can comment this and the next line out once you downloaded the data
nltk.download('stopwords')
nltk.download('punkt')  # hi Gosse and Wietse, this was needed as well!!!

from nltk.corpus import wordnet as wn

from nltk.wsd import lesk

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

# this returns the definition for a given synset-id, as returned by lesk

def wn_definition(synset,word) : 
    for ss in wn.synsets(word) :
        if ss == synset :
            return ss.definition()



# +
text = "The data in the table seems wrong. The table was filled with food."


def wsd(Text) :
    # split the text into sentences, split the sentences into words 
    # filter stopwords
    # run the lesk algorithm on each word in the text, using the other words from the sentence as context
    # return the synset-id and definition for each disambiguated word
    sentences = sent_tokenize(Text)
    words = [word_tokenize(sentence) for sentence in sentences]
    
    filtered_words = []
    for sentence in words:
        cleaned_sentence = []
        for word in sentence:
            if word not in stopWords:
                cleaned_sentence.append(word)
        filtered_words.append(cleaned_sentence)
    
    output = []
    for sentence in words:
        row = []
        for word in sentence:
            synset = lesk(sentence, word)
            row.append((synset, wn_definition(synset, word)))
        output.append(row)
            
    return output

wsd(text)
    
# -

# ## 7.2 Evaluation
#
# Select a suitable text containing approximately 20 content words and evaluate the results of the program you wrote in 7.1. For evaluation it may be convenient to not only print the definition for the synset definition selected by Lesk, but also the other synset definitions for this word. 
#
# Report two numbers: the percentage of content words for which the algorithm finds an entry in WordNet, and, for that subset, the percentage of words for which the correct word sense was provided.
#
# N.B. We expect *original work*. I.e. make sure you pick a text that is not also being evaluated by one of your fellow students.

# + pycharm={"name": "#%%\n"}
wsd("Lionel Messi is considered to be a goat. The goat was running away from the farm.")

# + pycharm={"name": "#%%\n"}
wsd("The rubber duck was floating in the swimming pool. The duck protected its children from the strangers in the park.")
# -

# ## 7.3 Predominant Meanings in Word Embeddings
#
# Investigate to what extent (traditional, word2vec style) word embeddings reflect various meanings of a word.
#
# 1. Select 5 words with at least 2 distinct meanings (word senses).
# 2. Do the 20 nearest neighbors of each of these words reflect both meanings (i.e. are there similar words that clearly are related to meaning A and/or meaning B) or is one meaning clearly predominant in the nearest neighbors?
# 4. Discuss (approx 100 words) what your findings mean for approaches to WSD that rely on the most frequent sense of a word.
#
# Use the [on line demo](http://bionlp-www.utu.fi/wv_demo/) that was also used to study bias in embeddings. 
#
#
# Try to __be original__ in your choice of key words, i.e. do not choose words that are also used as examples in the slides or in chapter 18 of J&M. 

# We chose the followings words for the comparisons:
#
# 1. chest
# 2. honey
# 3. clients
# 4. apple
# 5. client
#
# We used the online demo mentioned above, with the GoogleNews-vectors as word embeddings of choice.
#
# ## chest
#
# The top 20 nearest neighbours for the word _chest_ seem to be all related to the human body definition of the word. There appear to be no words that are related to boxes, for example.
#
# ## honey
#
# The top 20 nearest neighbours for the word _honey_ seem to be related to the natural substance and ingredient, rather than referencing a loved one.
#
# ## clients
#
# The top 20 nearest neighbours for the word _clients_ seem to be all related to customer affairs and business related words. Some words seem to be directly related to banking and finances.
#
# ## apple
#
# The top 20 nearest neighbours for the word _apple_ seem to be related to all sort of fruits and such. _almond_ and _potato_ are also among these words.
#
# ## client
#
# The top 20 nearest neighbours for the word _client_ seem to be more related to the legal profession, and includes words such as _form_, _attorneys_ and _defendant_. The differences with the word _clients_ are significant in this regard.
#
