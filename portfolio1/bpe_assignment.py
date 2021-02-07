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

# # Byte Pair Encoding with Huggingface Tokenizers
#
# Make sure to have [Pytorch](https://pytorch.org/get-started/locally/#start-locally) and the [huggingface transformers](https://huggingface.co/transformers/index.html) library installed. 
#
# This exercise follows the explanation of using BPE tokenization as explained on Huggingface [build a tokenizer from scratch](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#build-a-tokenizer-from-scratch).
#
# Following that example, we will be using byte-pair-encoding as described in section 2.4.3. of SLP (and originally by [Sennich et al, 2015](https://arxiv.org/abs/1508.07909)) for creating a vocabulary consisting of frequent words and subword tokens for handling less frequent words. 
#
# ## Set up the tokenizer
#
# This loads the BPE tokenizer and trainer, and tells the system to use whitespace as token separator, and defines [UNK] as special token for handling unknown words. 
#

# +
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=10000) # try with various vocab_sizes

# -

# ## Corpus
#
# The tokenizer creates a dictionary by concatenating characters and substrings into longer strings (possibly full words) based on frequency. So we need a corpus to learn what the most frequent words and substrings are. 
#
# [Wikitext-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) is a dump of the (English) Wikipedia. You can download using wget, or directly from the webpage. 
#
#     wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
#     unzip wikitext-103-raw-v1.zip
#     
# The unzipped data is 500 MB. Note that the file extension for the data-files is .raw but the data is just a (unicode) text file. Because this confuses (ubuntu) linux, files were renamed to .raw.txt. If you maintain the original .raw filenames, adapt the path below accordingly.
#
# ## Run the trainer
#
# The command below trains the tokenizer on the data.
#
#

# +
data = [f'wikitext-103-raw/wiki.{split}.raw' for split in ['train','test','valid']]
        
tokenizer.train(trainer,data)

# -

# ## Test the tokenizer
#
# Now that we have created a vocabulary, we can use it to tokenize a string into words and subtokens (for infrequent words).
#
# The example shows that most of the words are included in the vocabulary created by training on Wikipedia text, but that the acronym *UG*, the name *Hanze*, and the word *Applied* are segmented into subword strings. This suggests that these words were not seen during training, or very infrequently. (*UG* occurs 5 times in the training data and *Applied* over 200 times,  also note that the encoding is case-sensitive.). 
#
# Try a few other examples to get a feeling for the lexical coverage of the tokenizer. 

# +
example = "The UG and the Hanze University of Applied Sciences are jointly initiating a pilot rapid testing centre, which will start on 18 January."

output = tokenizer.encode(example)

print(output.tokens)

number_of_words = len(tokenizer.pre_tokenizer.pre_tokenize_str(example))
number_of_segments = len(output.tokens)

print("{} words and {} segments".format(number_of_words,number_of_segments))

# -

# ## Assignment: Experiment with vocabulary size
#
# The training data contains 103 M tokens and has a vocabulary size of 267,000 unique types. The default setting for the trainer is to create a dictionary of max 30,000 words. This means that a fair amount of compression takes place. Even more compression can be achieved by setting the vocab_size to a smaller value. 
#
# 1. Choose an example text consisting of at least 100 words. You may want to ensure that it contains some rare words or tokens. 
#
# 2. Experiment with various settings for vocab_size.
#
# 3. Count the number of words in the example, and the number of segments created by the BPE-tokenizer. Note that if the number segments goes up, more words are segmented into subwords. 
#
# 4. What is the vocabulary size where the number of segments is approx. 150% of the number of words? 
#
# 5. For this setting, what was the longest word in your example text that was not segmented? 

# +
trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=7500) # try with various vocab_sizes

tokenizer.train(trainer, data)

testdata = """
In January 2021, a short squeeze of the stock of the American video game retailer GameStop (NYSE: GME) and other securities took place, causing major financial consequences for certain hedge funds and large losses for short sellers. Approximately 140 percent of GameStop's public float had been sold short, and the rush to buy shares to cover those positions as the price rose caused it to rise even further. The short squeeze was initially and primarily triggered by users of the subreddit r/wallstreetbets, an Internet forum on the social news website Reddit. At its height, on January 28, the short squeeze caused the retailer's stock price to reach a pre-market value of over US$500 per share, nearly 30 times the $17.25 valuation at the beginning of the month. The price of many other heavily shorted securities increased.
"""

number_of_words = len(tokenizer.pre_tokenizer.pre_tokenize_str(testdata))

output = tokenizer.encode(testdata)

number_of_segments = len(output.tokens)

print("{} words and {} segments".format(number_of_words,number_of_segments))

print(output.tokens)
# answer the question about the longest word by going over the output, or write a few lines of code to provide the answer.

tokens = sorted(output.tokens, key=len)

print(tokens)
# -


