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

# ### Probing (NLP Week 5)
#
# Bert is a transformed-based, context-sensitive, neural language models that has been trained, among others, on a masked language modeling task, where the model learns to predict what the most likely word is at a a masked (hidden) position in the sentence. In a sentence like
#
#     There were several [MASK] with the proposed solution.
#     
# the model will learn that the word *problems* or *issues* is more likely at this position than the word *unicorns* or *days*. The model uses both the right and left context of the masked position to make its predictions. 
#
# A **probe** is a test of a language model aimed at investigating how accurate these predictions are, especially for cases where syntax makes it quite clear that one (form of a) word is correct, and another word is impossible. In the example above, for instance, the masked position can be filled by a plural noun (*problems*) but not by a singular noun (*problem*). If the model makes predictions that respect the linguistic constraints, we have reason to believe that the model is somehow aware of the linguistic structure of the language.
#
# While predicting whether the masked position should be filled by a singular or plural noun seems easy in the example above (both *were* and *several* are good predictors of plural), we can try to make the task harder by looking for contexts where the solution requires more careful *attention* to the right words in the context
#
#     There were some [MASK] with the proposed solution.
#     There could be several [MASK] with the proposed solution.
#     There were some unexpected aand unresolbed [MASK] with the proposed solution.
#     
# In the examples above, the task is made harder by replacing *several* (which is always followed by a singular noun) by *some* (which can be followed by a singular or a plural noun), by replacing *were* (which always heads a sentence with a plural subject) by *could be* (which can head a sentence with a singular or plural subject), and by inserting material between the verb *were* (which indicates that there should be a plural) and the MASK. 
#
# ## Assignment
#
# Think of a grammatical phenomenon in a language of your choice, and come up with at least 10 example sentences to probe whether the model makes the correct predictions. Think of cases where the context makes it clear that the mask has to be plural or singular, that a verb has to have a particular form (like plural or singular, or participle or infinitive), that a specific (personal, possessive, reflexive) pronoun has to be used, that an adjective or noun has to have a specific inflection (like in German and more generally in languages with a rich case and/or gender marking system). There is a host of literature on this, see for instance [Marvin and Linzen](https://arxiv.org/abs/1808.09031) (for English) and [Sahin et al](https://www.mitpressjournals.org/doi/full/10.1162/coli_a_00376) (for multilingual probes). 
#
# ## Model
#
# The model we will be using for this task is the [multilingual BERT](https://huggingface.co/bert-base-multilingual-cased) model mBERT, that was trained on the Wikipedia text of the 102 largest Wikipedia's. This means that you do not have to choose examples from English, but that you may also present a probe for another language. 
#
# The following loads the pipeline for doing masked prediction, and load the mBERT model (this may take a minute or so). You can ignore the warning about some weights not being initialized. 
#

# +
from transformers import pipeline

mbert = pipeline('fill-mask', model='bert-base-multilingual-cased')

# -

# The pipeline can be used to test masked language model prediction. Given a sequence containing the special token [MASK], the model will predict what the most likely tokens are at that position, using both left and right context. 

mbert('There were several unexpected and unresolved [MASK] with the proposed solution.')

# + pycharm={"name": "#%%\n"}
mbert("De minister-president heeft [MASK] handtekening gezet onder de bepaling.")
# -

mbert("De ontwikkelaars onthouden [MASK] van commentaar op hun code.")

# By default, the pipe returns the 5 most likely words that could appear at the position of the mask, along with a score. If you want to know specifically whether the model prefers one of two forms, you can give these forms as targets to the pipe, and also print the answer in a more readable form:

mbert("Het ijsje begon met [MASK] toen de zon doorbrak.")

mbert("De universiteit investeerde [MASK] in onderzoeken naar kunstmatige intelligentie.")

mbert("Het zwaard van Damocles hing boven [MASK] hoofd.")

mbert("De kaas uit Duitsland was niet [MASK] dan de kaas uit Nederland.")

mbert("De diplomaten dronken uit [MASK] bekers terwijl het ongeval plaatsvond.")

mbert("Het weer was onstuimig, het [MASK] namelijk vrij hard.")

mbert("De scholen sloten hun deuren [MASK].")

mbert("Jan had gisteren [MASK] voet gestoten tegen de tafelpoot.")


# +
def probe(Sentence,Targets) :
    for res in mbert(Sentence,targets=Targets) :
        print('{:6.4f}\t{}'.format(res['score'],res['sequence']))
        
probe('There were some unexpected and unforeseen [MASK] for the proposed solution',['challenge','challenges'])
# -

# ## Your Probe here 
#
# Give at least ten example sentences with a [MASK] and a list of targets that illustrate a specific grammatical phenomenon in a language of your choice. Describe what the grammatical phenomenon is you are investigating. Use the probe function for testing. Try to include both *easy* sentences (where the model should do well) as well as *hard* sentences (where there are words in the context that might lead to confusion, or where the clue words are far away from the mask). For languages other than Dutch or English, make sure to include enough explanation so that examples and tests are clear to a non-native speaker. 
#
# Describe how well the model did on your probe sentences. Where there any cases where the model made the wrong decision?

probe("The weather was [MASK]", ['good', 'better'])

probe("He climbed the [MASK] today", ['mountain', 'mountains'])

probe("She went to the Dutch city of [MASK]", ["Amsterdam", "London"])

probe("The matters that were discussed during today's meeting [MASK] returning tomorrow.", ['are', 'is'])

probe("The dog [MASK] hunting for food in the evenings.", ['was', 'is', 'are'])

probe("The person visited [MASK] grandparents today.", ['their', 'his', 'her'])

probe("The doctor helped [MASK] with the construction of the bed.", ['him', 'her', 'them'])

probe("He completed the race in six [MASK]", ['minutes', 'hours', 'seconds'])

probe("London [MASK] fallen.", ['has', 'have'])

probe("The streets of [MASK] are dirty.", ['Paris', 'London', 'Amsterdam'])


