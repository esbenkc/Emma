# -*- coding: utf-8 -*-
# Author: CINeMa - SOHa and E. KranC
# Thanks to CINeMa (https://inema.webflow.io),
# the Sentida team, jry, VADER, AFINN, and last
# but not least Formula T., for inspiration and encouragement.
# For license information, see LICENSE.TXT

'''
The SentidaV2 sentiment analysis tool is freely available for
research purposes (please cite). If you want to use the tool
for commercial purposes, please contact:
    - contact@esbenkc.com
    - sorenorm@live.dk

SENTIDA v2.
Aarhus University, Cognitive Science.
2019 - Cognition & Communication.
@authors: soha & kranc.

This script was developed along with other tools in an attempt to improve danish sentiment analysis.
The tool will be updated as more data is collected and new methods for more optimally accessing sentiment is developed.
'''

# VADER imports

import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer

# VADER basis values
# VADER shows 0.291, 0.215, and 0.208 for !, !!, and !!!
# UPPERCASE is 0.733
# Degree modification tests with EXTREMELY is 0.293 as empirical basis

# SENTIDA V2 basis values
# Currently using VADER basis values
# Question mark is: XXX
# Degree modifications for other words are implemented in intensitifer list
#     - Need implementation of larger intensifier list based on sentences

#################
### CONSTANTS ###
#################

B_INCR = 0.293
B_DECR = -0.293

C_INCR = 0.733
N_SCALAR = -0.74

QM_MULT = 0.94
QM_SUCC_MULT = 0.18

N_TRIGGER = "no"

NEGATE = \
    ['ikke', 'ik', 'ikk', 'ik\'', 'aldrig', 'ingen']

ADD = \
    ['og', 'eller']

BUT_DICT = \
    ['men', 'dog']

BOOSTER_DICT = \
    {"temmelig": 0.1, "meget": 0.2, "mega": 0.4, "lidt": -0.2, "ekstremt": 0.4,
    "totalt": 0.2, "utrolig": 0.3, "rimelig": 0.1, "seriøst": 0.3}

SENTIMENT_LADEN_IDIOMS = {}

SPECIAL_CASE_IDIOMS = {}

####################
### INSTRUCTIONS ###
####################

"""
sentidaV2(text)

"""

##############################
### CUSTOM IMPLEMENTATIONS ###
##############################
    ### STATIC METHODS ###
    ######################

# Function for working around the unicode problem - shoutout to jry
def fix_unicode(df_col):
    return df_col.apply(lambda x: x.encode('raw_unicode_escape').decode('utf-8'))

# Reading the different files and fixing the encoding
aarup = pd.read_csv('aarup.csv', encoding='ISO-8859-1')
intensifier = pd.read_csv('intensifier.csv', encoding='ISO-8859-1')
intensifier['stem'] = fix_unicode(intensifier['stem'])


# Function for modifing sentiment according to the number of exclamation marks:
def exclamation_modifier(sentence):
    ex_intensity = [1.291, 1.215, 1.208]
    ex_counter = sentence.count('!')
    value = 1
#    if ex_counter > 3:
#        ex_counter = 3
    if ex_counter == 0:
        return 1
    for idx, m in enumerate(ex_intensity):
        if idx <= ex_counter:
            value *= m
    return value

# Function for counting the number of question marks in the input:
def question_identifier(sentence):
    return sentence.count('?')

# Function for cleaning the input of punctuation:
def punct_cleaner(sentence):
    table = str.maketrans('!?-+_#.,;:\'\"', 12*' ')
    return sentence.translate(table)


# Function for making letters lower case:
def string_to_lower(sentence):
    return sentence.lower()


# Function for splitting sentences by the spaces:
def split_string(sentence):
    return sentence.split()


# Function for removing punctuation from a sentence and turning it into a list of words
def clean_words_caps(sentence):
    return split_string(punct_cleaner(sentence))


# Function for removing punctuation from a sentence, making the letters lower case, and turning it into a list of words
def clean_words_lower(sentence):
    return split_string(string_to_lower(punct_cleaner(sentence)))


# Function for getting the positions of words that are written in upper case:
def caps_identifier(words):
    positions = []

    for word in words:
        if word.upper() == word:
            positions.append(words.index(word))

    return positions


# Function for modifing the sentiment score of words that are written in upper case:
def caps_modifier(sentiments, words):
    positions = caps_identifier(words)

    for i in range(len(sentiments)):
        if i in positions:
            sentiments[i] *= 1.733

    return sentiments
# VADER: 1.733

# Function for identifying negations in a list of words. Returns list of positions affected by negator.
def get_negator_affected(words):
    positions = []

    for word in words:
        if word in NEGATE:
            neg_pos = words.index(word)
            positions.append(neg_pos)
            positions.append(neg_pos + 1)
            positions.append(neg_pos - 1)
            positions.append(neg_pos + 2)
            positions.append(neg_pos + 3)
    return positions

# Get all intensifiers
def get_intensifier(sentiments, word_list):
    intensifiers_df = intensifier.loc[intensifier['stem'].isin(word_list)]
    intensifiers = intensifiers_df['stem'].tolist()
    scores = intensifiers_df['score'].tolist()
    position = []

    for word in word_list:
        if word in intensifiers:
            inten_pos = word_list.index(word)

            if inten_pos + 1 not in position:
                position.append(inten_pos + 1)
                if inten_pos + 1 < len(sentiments):
                    sentiments[inten_pos + 1] *= scores[intensifiers.index(word)]

            if inten_pos - 1 not in position:
                position.append(inten_pos - 1)
                if inten_pos - 1 > 0:
                    sentiments[inten_pos - 1] *= scores[intensifiers.index(word)]

            if inten_pos + 2 not in position:
                position.append(inten_pos + 2)
                if inten_pos + 2 < len(sentiments):
                    sentiments[inten_pos + 2] *= scores[intensifiers.index(word)]

            if inten_pos + 3 not in position:
                position.append(inten_pos + 3)
                if inten_pos + 3 < len(sentiments):
                    sentiments[inten_pos+3] *= scores[intensifiers.index(word)]

    return sentiments


# Function for identifying 'men' (but) in a list of words:
def men_identifier(words):
    position = 0

    for word in words:
        if word == 'men':
            position = words.index(word)

    return position


# Function for modifying the sentiment score according to whether the words are before or after the word 'men' (but) in a list of words
def men_sentiment(sentiments, words):
    for i in range(len(sentiments)):
        if i < men_identifier(words):
            sentiments[i] *= 0.5
        else:
            sentiments[i] *= 1.5

    return sentiments
# Need imperical tested weights for the part before and after the 'men's'


# Function for stemming the words of a sentence (stemming is NOT optimal for expanding the vocabulary!):
def stemning(words):
    stemmer = SnowballStemmer('danish')
    return [stemmer.stem(word) for word in words]


# Function that takes a list of words as the input and returns the corresponding sentiment scores
def get_sentiment(word_list):
    sentiment_df = aarup.loc[aarup['stem'].isin(word_list)]
    words = sentiment_df['stem'].tolist()
    scores = sentiment_df['score'].tolist()
    senti_scores = []

    for i in word_list:
        if i in words:
            senti_scores.append(scores[words.index(i)])
        else:
            senti_scores.append(0)

    return senti_scores



# Function for turning a sentence into a mean sentiment score:
def sentidaV2(sentence, output = ["mean", "total"]):
    words_caps = clean_words_caps(sentence)
    words_lower = clean_words_lower(sentence)
    stemmed = stemning(words_lower)
    sentiments = get_sentiment(stemmed)

    if men_identifier(words_lower) > 0:
        sentiments = men_sentiment(sentiments, words_lower)

    sentiments = get_intensifier(sentiments, stemmed)
    sentiments = caps_modifier(sentiments, words_caps)

    if question_identifier(sentence) == 0:
        for i in set(get_negator_affected(words_lower)):
            if i < len(sentiments) and i >= 0:
                sentiments[i] *= -1

    if len(words_lower) == 0:
        return 0

    sentiments[:] = [sentiment for sentiment in sentiments if sentiment != 0]

    total_sentiment = sum(sentiments) * exclamation_modifier(sentence)
    mean_sentiment = total_sentiment/len(sentiments)
    if mean_sentiment > 10:
        mean_sentiment = 10
    if mean_sentiment < -10:
        mean_sentiment = -10
    return mean_sentiment

def sentidaV2_examples():
    print("Example of usage: ", sentidaV2("Lad der blive fred.", output = "mean"))
    # Example of usage: 2.0
    print("With exclamation mark: ", sentidaV2("Lad der blive fred!", output = "mean"))
    # With exclamation mark: 3.13713
    print("With several exclamation mark: ", sentidaV2("Lad der blive fred!!!", output = "mean"))
    # With several exclamation mark:  3.7896530399999997
    print("Uppercase: ", sentidaV2("Lad der BLIVE FRED", output = "mean"))
    # Uppercase:  3.466
    print("Negative sentence: ", sentidaV2("Det går dårligt.", output = "mean"))
    # With exclamation mark:  -1.8333333333333335
    print("Negation in sentence: ", sentidaV2("Det går ikke dårligt.", output = "mean"))
    # Negation in sentence:  1.8333333333333335
    print("'Men' ('but'): ", sentidaV2("Lad der blive fred, men det går dårligt.", output = "mean"))
    # 'Men' ('but'):  -1.5


# Still missing: common phrases, adjusted values for exclamation marks,
# Adjusted values for men-sentences, adjusted values for capslock,
# More rated words, more intensifiers/mitigators, better solution than snowball stemmer,
# Synonym/antonym dictionary.
# Social media orientated: emoticons, using multiple letters - i.e. suuuuuper.



##################################
### CUSTOM FUNCTIONS OPTIMIZED ###
##################################

def sentidaV2_dictonly (text):
