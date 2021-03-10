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

# ### Tiny Transition-based Dependency Parser 
#
# An implementation of a greedy transition-based dependency parser (unlabeled parsing only).
# Released under BSD license.
#
# Code is an adapted version of Matthew Honnibal's [parser](https://explosion.ai/blog/parsing-english-in-python)
#
# modified by bplank, 03/2017 gbouma 3/2021
#

# +
import argparse
from collections import namedtuple, defaultdict
import os
import random
import time
import sys

## make sure myparserutils.py is in same dir as this notebook
from myparserutils import DefaultList, Parse, Perceptron, MOVES, SHIFT, RIGHT, LEFT

random.seed(113) # keep fixed seed for now

#### Core parsing logic classes (parser + learner) - do not modfiy

class Parser(object):

   ## fixing __file__ issue, https://stackoverflow.com/questions/39125532/file-does-not-exist-in-jupyter-notebook  
    def __init__(self, load=True):
        # model_dir = os.path.dirname(__file__)
        model_dir = os.path.abspath('')  
        self.model = Perceptron(MOVES)
        if load:
            self.model.load(os.path.join(model_dir, 'parser.pickle'))
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))

    def save(self):
        self.model.save(os.path.join(model_dir, 'parser.pickle'))

    def parse(self, words, lemmas, tags):
        """
        transition-based parsing
        """
        n = len(words)
        i = 2; stack = [1]; parse = Parse(n)

        while stack or (i+1) < n:
            features = extract_features(words, lemmas, tags, i, n, stack, parse)
            scores = self.model.score(features)
            valid_moves = self.get_valid_moves(i, n, len(stack))
            guess = max(valid_moves, key=lambda move: scores[move])
            i = self.transition(guess, i, stack, parse)
        return tags, parse.heads

    def train_one(self, itn, words, lemmas, gold_tags, gold_heads):
        n = len(words)
        i = 2; stack = [1]; parse = Parse(n)

        while stack or (i + 1) < n:
            features = extract_features(words, lemmas, gold_tags, i, n, stack, parse)
            scores = self.model.score(features)
            valid_moves = self.get_valid_moves(i, n, len(stack))
            gold_moves = self.get_gold_moves(i, n, stack, parse.heads, gold_heads)
            guess = max(valid_moves, key=lambda move: scores[move])
            assert gold_moves
            best = max(gold_moves, key=lambda move: scores[move])
            self.model.update(best, guess, features)
            i = self.transition(guess, i, stack, parse)
            self.confusion_matrix[best][guess] += 1
        return len([i for i in range(n-1) if parse.heads[i] == gold_heads[i]])

    def train(self, sentences, nr_iter):
        """
        train the parser on the trainin data
        :param sentences:  training data
        :param nr_iter: number of iterations
        :return:
        :rtype:
        """
        for itn in range(nr_iter):
            corr = 0;
            total = 0
            random.shuffle(sentences)
            for words, lemmas, gold_tags, gold_parse, gold_label in sentences:
                corr += self.train_one(itn, words, lemmas, gold_tags, gold_parse)
                total += len(words)
            print(itn, '%.3f' % (float(corr) / float(total)))
        print('Total:', len(self.model.weights))
        print('Averaging weights')
        self.model.average_weights()

    def transition(self, move, i, stack, parse):
        if move == SHIFT:
            stack.append(i)
            return i + 1
        elif move == RIGHT:
            parse.add(stack[-2], stack.pop())
            return i
        elif move == LEFT:
            parse.add(i, stack.pop())
            return i
        assert move in MOVES

    def get_valid_moves(self, i, n, stack_depth):
        moves = []
        if (i + 1) < n:
            moves.append(SHIFT)
        if stack_depth >= 2:
            moves.append(RIGHT)
        if stack_depth >= 1:
            moves.append(LEFT)
        return moves

    def get_gold_moves(self, n0, n, stack, heads, gold):
        # print "get_gold_moves(n0={}, n={}, stack={}, gold={})".format(n0, n, stack, gold)
        def deps_between(target, others, gold):
            for word in others:
                if gold[word] == target or gold[target] == word:
                    return True
            return False

        valid = self.get_valid_moves(n0, n, len(stack))
        if not stack or (SHIFT in valid and gold[n0] == stack[-1]):
            return [SHIFT]
        if gold[stack[-1]] == n0:
            return [LEFT]
        costly = set([m for m in MOVES if m not in valid])
        # If the word behind s0 is its gold head, Left is incorrect
        if len(stack) >= 2 and gold[stack[-1]] == stack[-2]:
            costly.add(LEFT)
        # If there are any dependencies between n0 and the stack,
        # pushing n0 will lose them.
        if SHIFT not in costly and deps_between(n0, stack, gold):
            costly.add(SHIFT)
        # If there are any dependencies between s0 and the buffer, popping
        # s0 will lose them.
        if deps_between(stack[-1], list(range(n0 + 1, n - 1)), gold):
            costly.add(LEFT)
            costly.add(RIGHT)
        return [m for m in MOVES if m not in costly]
    
## internal functions to access buffer, staff and context - do not modify
def get_stack_elements(stack, data):
    """
    extracts top 3 elements from data from stack, e.g.
    if data is 'words' then returns up to top 3 words
    if data is 'tags' returns top 3 POS tags
    
    returns '' if elements is not available
    """
    depth = len(stack)
    if depth >= 3:
        return data[stack[-1]], data[stack[-2]], data[stack[-3]]
    elif depth >= 2:
        return data[stack[-1]], data[stack[-2]], ''
    elif depth == 1:
        return data[stack[-1]], '', ''
    else:
        return '', '', ''

def get_buffer_elements(i, data):
    """
    extracts top 3 elements from data from buffer

    returns '' if elements is not available
    """
    n = len(data)
    if i + 1 >= n:
        return data[i], '', ''
    elif i + 2 >= n:
        return data[i], data[i + 1], ''
    else:
        return data[i], data[i + 1], data[i + 2]


def get_parse_valency(i, deps, data):
    """
    get number of childs at position 'i' in edges R created so far
    """
    if i == -1:
        return 0
    valency = len(deps[i])
    if not valency:
        return 0
    else:
        return valency

def get_parse_context(i, deps, data):
    """
    get the two elements from data at position 'i' linked through deps [left|right] (to last added edges)
    """
    if i == -1 or not len(deps):
        return '', ''
    deps = deps[i]
    valency = len(deps)
    if valency == 1:
        return data[deps[-1]], ''
    else:
        return data[deps[-1]], data[deps[-2]]


# -

# ### Assignment 6.4: Improve the oracle by including more context features 
#
#     
# We need to extract useful features from configurations so that
#     we can train a classifier for the possible actions.
#     
# Recall that a configuration is a particular state of the parser
# at a particular time, and consists of three pieces of information/elements:
#     
#      1. The stack S
#      2. The buffer B
#      3. The current set of relations R
#     
# In principle any property of these elements can be used as a feature
# and used for training the parser. However, the right trade-off between
# adding too many features and having too few should be found. Adding too
# many features might end in very sparse models that are hard to train,
# and might not work well (generalize) to new data. It is thus best
# to focus the learning algorithm on the most useful aspects of decision making at each point in the parsing process.
#   
# The focus of feature extraction in transition-based parsing is 
#   typically the top of the stack, the words near the front of
#      the buffer and the set of dependency relations already created.
#     
# It is advisable to read the section 'Features' in [chpt 14.4](https://web.stanford.edu/~jurafsky/slp3/14.pdf) in J&M 3rd. 
#
# To test the baseline, just leave this part untouched, and run the training and evaluation procedure below. 

# +
def extract_features(words, lemmas, tags, n0, n, stack, parse):

      # 'features' is our list of features were we want to add additional features
    # by instantiating features from the feature templates
    # we always add features with a value of 1!
    features = []
    features.append(('bias', 1)) # this is a feature that is necessary for the perceptron, keep it as is

    # check were we are
    # s0 = top of stack
    s0 = stack[-1] if len(stack) else -1

    # Set up the context pieces --- the word (W) and tag (T) of:

    # s_w0, s_w1, s_w2: Top three words on the stack
    # b_w0, b_w0, b_w1: Next three words of the buffer

    ## #### First get atomic units (elements) which we then can use to add features #####

    ## get word and tags from stack (empty string if not available)
    # depth=last item in stack
    s_w0, s_w1, s_w2 = get_stack_elements(stack, words) #tokens
    s_p0, s_p1, s_p2 = get_stack_elements(stack, tags) #pos

    ## get word and tags from buffer
    # n0 = current position on buffer
    b_w0, b_w1, b_w2 = get_buffer_elements(n0, words) #tokens
    b_p0, b_p1, b_p2 = get_buffer_elements(n0, tags) #tags

    # parse: keeps the dependency graph (edges) constructed so far (parse.lefts and parse.rights)

    # For first word on buffer, get valency and words/tags of left/right daugthers
    # Two leftmost children of the first word of the buffer
    b_l_w0, b_l_w1 = get_parse_context(n0, parse.lefts, words)
    b_l_p0, b_l_p1 = get_parse_context(n0, parse.lefts, tags)

    b_r_w0, b_r_w1 = get_parse_context(n0, parse.rights, words)
    b_r_p0, b_r_p1 = get_parse_context(n0, parse.rights, tags)

    b_l_valence = get_parse_valency(n0, parse.lefts, words)
    b_r_valence = get_parse_valency(n0, parse.rights, words)

    # For word on top of stack, get valency and words/tags of left/right daugthers
    s_l_w0, s_l_w1 = get_parse_context(s0, parse.lefts, words)
    s_l_p0, s_l_p1 = get_parse_context(s0, parse.lefts, tags)

    s_r_w0, s_r_w1 = get_parse_context(s0, parse.rights, words)
    s_r_p0, s_l_p1 = get_parse_context(s0, parse.rights, tags)

    s_l_valence = get_parse_valency(s0, parse.lefts, words)
    s_r_valence = get_parse_valency(s0, parse.rights, words)


    # Cap numeric features at 5
    # String-distance
    Ds0n0 = min((n0 - s0, 5)) if s0 != 0 else 0


    ##### Now that we have extracted some element, let us instantiate feature templates (=adding actual features)

    # unigrams
    # add word features (including position) - on stack
    if s_w0: features.append(('s_w0=%s' % (s_w0), 1))
    if s_w1: features.append(('s_w1=%s' % (s_w1), 1))
    if s_w2: features.append(('s_w2=%s' % (s_w2), 1))

    # [ADD MORE!]
    # word unigram features on buffer
    if b_w0: features.append(('b_w0=%s' % (b_w0), 1))
    
    # add pos features
    if s_p0: features.append(('s_p0=%s' % (s_p0), 1))

    # some bigram features
    if s_w0 and s_p0: features.append(('s_w0=%s s_p0=%s' % (s_w0, s_p0), 1))



    return features



# -

# ### Debugging
#
# test_extraction is a function that shows an example configuration for testing/exemplar purposes only
#     and what the output of the three functions is:
#     - get_stack_elements
#     - get_buffer_elements

# +
def test_extraction():
    """
   
    """

    ### build a configuration, which consists of stack, buffer and (partial) parse
    words = DefaultList(''); tags = DefaultList(''); lemmas = DefaultList('')
    sentence = ["United", "canceled", "the", "morning", "flights", "to","Houston"]
    lemmalist = ["United","cancel","the", "morning", "flight", "to", "Houston"]
    pos = ["N","V","D","N","N","A","N"]
    for i, (w,l,t) in enumerate(zip(sentence,lemmalist,pos)):
        words.append(sentence[i])
        lemmas.append(lemmalist[i])
        tags.append(pos[i])
    pad_tokens(words)  # add ROOT to end!
    pad_tokens(lemmas)
    pad_tokens(tags)

    ### current configuration
    ### this is the configuration given in J&M

    # the stack in the code is encoded with indices to the words
    print(words)
    stack = [len(words)-1,2,5] # corresponds to ["ROOT","canceled","flights"]
    buffer = ["to","Houston"]
    parse = Parse(len(words))
    # manually create partial structure (edges so far)
    parse.add(2,1) # canceled-> Houston
    parse.add(5,4) # flights->morning
    parse.add(5,3) # flights->the
    print("{} > {}".format(words[2], words[1]))
    print("{} > {}".format(words[5], words[4]))
    print("{} > {}".format(words[5], words[3]))



    print("Stack: ", stack)
    print("len(stack):", len(stack))
    depth=len(stack)
    current_position = stack[-1] # 5
    Ws0, Ws1, Ws2 = get_stack_elements(stack, words)
    print("Words from stack: Ws0: {} Ws1: {} Ws2: {}".format(Ws0,Ws1,Ws2))


    Wn0, Wn1, Wn2 = get_buffer_elements(current_position+1, words)
    print("Buffer: Wn0: {} Wn1: {} Wn2: {}".format(Wn0,Wn1,Wn2))

    print(parse.lefts)
    Wn0b1, Wn0b2 = get_parse_context(current_position, parse.rights, words)
    Vn0b = get_parse_valency(current_position, parse.rights, words)
    print("Context:")
    print("valency: ", Vn0b)
    print("Wn0b1: ", Wn0b1)
    print("Wn0b2:", Wn0b1)

    print("parse.rights:", parse.rights)
    print("parse.lefts:", parse.lefts)
    Ws0b1, Ws0b2 = get_parse_context(current_position, parse.lefts, words)
    Vs0b = get_parse_valency(current_position, parse.lefts, words)
    print("Context:")
    print("valency: ", Vs0b)
    print("Wn0b1: ", Ws0b1)
    print("Wn0b2:", Ws0b2)
    

def pad_tokens(tokens):
    tokens.insert(0, '<start>')
    tokens.append('ROOT')

    
test_extraction()
# -

# ### Reading conll data 

# +
ConllRow = namedtuple('ConllRow', 'id form lemma cpostag postag feats head deprel phead pdelrep'.split(" "))

def read_conll(loc):
    for sent_str in open(loc, encoding='utf-8').read().strip().split('\n\n'):
        lines = [ConllRow(*line.split("\t")) for line in sent_str.split('\n')]
        words = DefaultList(''); tags = DefaultList(''); lemmas = DefaultList('')
        heads = [None]
        labels = [None]
        lemmas = [None]
        for i, row in enumerate(lines):
            words.append(sys.intern(row.form))
            tags.append(sys.intern(row.cpostag))
            lemmas.append(sys.intern(row.lemma))
            heads.append(int(row.head) if row.head != '0' else len(lines)+1) #+1 otherwise root in gold is one off!
            labels.append(row.deprel)
        ## insert start token, ROOT is last token (append to end)
        pad_tokens(words)
        pad_tokens(lemmas)
        pad_tokens(tags)
        yield words, lemmas, tags, heads, labels

#### end read data


# -

# ### Training the parser and oracle

# +
parser = Parser(load=False)

traincorpus = 'nl-train.conll'
iterations = 10 

model_dir = ''
# reading training TB (assumes all are projective!)
train_data = list(read_conll(traincorpus))
    
print(("# sentences: {}".format(len(train_data))))

print("train parser...")
parser.train(train_data, nr_iter=iterations)
print("save model...")
parser.save()
print("done.")

# -

# ### Evaluation 

# +
parser = Parser(load=True)

testfile = 'nl-dev.conll'

outputfile ='predicted.txt' 

write_output_to_file = 1 # set to 0 for not writing predictions to file

print("testing..")
if write_output_to_file :
    OUT = open(outputfile,"w", encoding='utf-8')
    
# Testing
c = 0
t = 0
gold_sents = list(read_conll(testfile))
t1 = time.time()
for (words, lemmas, tags, gold_heads, gold_labels) in gold_sents:
    _, heads = parser.parse(words, lemmas, tags)
    for i, w in list(enumerate(words))[1:-1]:
         #if gold_labels[i] in ('P', 'punct'):  
        #    continue
        if heads[i] == gold_heads[i]:
            c += 1
        t += 1
    if write_output_to_file:
        for i, (word, lemma, pos, head, g) in enumerate(zip(words, lemmas, tags, heads, gold_heads)):
            if i==0:
                continue
            ## reset head before printing (set back root to 0)
            if head == len(words)-1:
                head = 0
            if g == len(words)-1:
                g = 0
            #id  form lemma postag (ignore) features head dependencyRelation (ignore) (ignore)
            OUT.write("{}\t{}\t{}\t{}\t_\t_\t{}\t_\t_\t_{}\n".format(i, word, lemma, pos, head, g))
            
        OUT.write("\n")
t2 = time.time()


if write_output_to_file:
    OUT.close()
        
print('Parsing took %0.3f ms' % ((t2-t1)*1000.0))
print("Unlabeled attachment scores (UAS) %0.2f (correct: %s, total %s)" % (c/t*100, c, t))
        
# -


