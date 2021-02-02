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

# # Make a Chatbot using Regular Expressions
#
# In this assignment, you have to construct a chatbot using regular expressions. The assignment is follows the design of the famous [Eliza](https://en.wikipedia.org/wiki/ELIZA) chatbot from the early days of NLP. 
#
# ![eliza.png](attachment:eliza.png)
#
# The idea is to scan user input by using regular expressions, and to answer accordingly. An answer can be a fixed response ('tell me more about that!') or use regular expressions to turn the users response into a follow-up question.
#
# Expand the basic Python implementation below so that at least 5 substitutions are included. Include an example of a conversation with the chatbot. 
#
# The python regular expression syntax is explained here : https://docs.python.org/3/howto/regex.html
#

# ## Assignment 
#
# Expand the bare-bones Elize implementation below into a chatbot that can deal with more types of input. The substitute function does substitutions using regular expressions. Note that the output of the most deeply embedded substitution is always input to the substitution one level up. Thus, the user response is input to the most deeply embedded substitution.
#
# Add or change the substitutions to handle more user responses (at least 5). You can also add more general substitutions, such as changing *my* to *your* or ensuring that the resonse from Eliza always ends with a '?'. 
#
# If the user just types quit, the loop stops. 

# +
import re

# a general function for matching a pattern against a String (the user input) and 
# returning the Replacement (which can be defined by referring back to Match)
def substitute(Match,Replacement,String) : 
    pattern = re.compile(Match)
    return pattern.sub(Replacement,String)

# the chatbot 
def eliza(Message) :
    UserResponse = input(Message+'\t')
    if UserResponse == 'quit' :
        return
    else :
        Answer = substitute(r'I feel',r'Why do you feel',
                            substitute(r'I am',r'Why are you',UserResponse ))
        if Answer == UserResponse : # i.e. no pattern matched the user response 
            eliza('Could you tell more about that?')
        else :
            eliza(substitute(r'\bmy\b', r'your', Answer))
        
eliza('How are you?')
    
# -

# Copy the output of one example dialogue here. 

#
