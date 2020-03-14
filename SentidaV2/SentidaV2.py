# -*- coding: utf-8 -*-
# Author: CINeMa - SOHa and E. KranC

'''
Thanks to CINeMa (https://inema.webflow.io),
the Sentida team, jry, VADER, AFINN, and last
but not least Formula T., for inspiration and encouragement.
For license information, see LICENSE.TXT

The SentidaV2 sentiment analysis tool is freely available for
research purposes (please cite). If you want to use the tool
for commercial purposes, please contact:
    - contact@esbenkc.com
    - sorenorm@live.dk

SENTIDA v2.
Aarhus University, Cognitive Science.
2019 - Cognition & Communication.
@authors: soha & kranc.

This script was developed along with other tools in an attempt to improve 
danish sentiment analysis. The tool will be updated as more data is collected 
and new methods for more optimally accessing sentiment is developed.

________________________________________________________________________________________

VADER BASIS VALUES

Multiplication values:
    0.291, 0.215, and 0.208 for !, !!, and !!! respectively
        empirically tested by one sentence compared to the three conditions
    0.733 for uppercase 
        empirically tested from single control sentence to uppercase version
    0.293 for degree modifications from adverbs
        empirically tested with "extremely"

SENTIDA V2 basis values
Currently using VADER basis values
Question mark is: XXX
Degree modifications for other words are implemented in intensitifer list
    - Need implementation of larger intensifier list based on sentences
________________________________________________________________________________________    

FUTURE IMPROVEMENTS

Still missing: common phrases, adjusted values for exclamation marks,
Adjusted values for men-sentences, adjusted values for uppercase,
More rated words, more intensifiers/mitigators, better solution than snowball stemmer,
Synonym/antonym dictionary.
Social media orientated: emoticons, using multiple letters - i.e. suuuuuper.
Normalization with respect to sub-(-1) and super-(1) output values
_____________________________________________________________________________________'''

#################
### LIBRARIES ###
#################

import pandas as pd, nltk, numpy as np
from nltk.stem import SnowballStemmer

#################
### CONSTANTS ###
#################

B_INCR = 0.293
B_DECR = -0.293

C_INCR = 0.733
N_SCALAR = -0.74

QM_MULT = 0.94
QM_SUCC_MULT = 0.18

EX_INTENSITY = [1.291, 1.215, 1.208]
UP_INTENSITY = 1.733
BUT_INTENSITY = [0.5, 1.5]

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

#####################
### DOCUMENTATION ###
#####################

"""
_____________________________

sentidaV2(
        text = 'Lad der blive fred.', 
        output = 'mean', 
        normal = False)

Example of usage:
Lad der bliver fred
Sentiment =  2.0 
_____________________________

sentidaV2(
        text = 'Lad der blive fred!', 
        output = 'mean', 
        normal = False)

With exclamation mark:
Lad der blive fred!
Sentiment =  3.13713 
_____________________________

sentidaV2(
        text = 'Lad der blive fred!!!', 
        output = 'mean', 
        normal = False)

With several exclamation mark:
Lad der blive fred!!!
Sentiment =  3.7896530399999997 
_____________________________

sentidaV2(
        text = 'Lad der BLIVE FRED', 
        output = 'mean', 
        normal = False)

Uppercase:
lad der BLIVE FRED
Sentiment =  3.466 
_____________________________

sentidaV2(
        text = 'Det går dårligt.', 
        output = 'mean', 
        normal = False)

Negative sentence:
Det går dårligt
Sentiment =  -1.8333333333333335 
_____________________________

sentidaV2(
        text = 'Det går ikke dårligt.', 
        output = 'mean', 
        normal = False)

Negation in sentence:
Det går ikke dårligt
Sentiment =  1.8333333333333335 
_____________________________

sentidaV2(
        text = 'Lad der blive fred, men det går dårligt.', 
        output = 'mean', 
        normal = False)

'Men' ('but'):
Lad der blive fred, men det går dårligt
Sentiment =  -1.5 
_____________________________

sentidaV2(
        text = 'Lad der blive fred.', 
        output = 'mean', 
        normal = True)

Normalized:
Lad der blive fred
Sentiment =  0.4 
_____________________________

sentidaV2(
        text = 'Lad der bliver fred. Det går dårligt!', 
        output = 'by_sentence_mean', 
        normal = False)

Multiple sentences mean:
Lad der bliver fred. Det går dårligt!
Sentiments = [2.0, -2.8757025] 
_____________________________

sentidaV2(
        text = 'Lad der bliver fred. Det går dårligt!', 
        output = 'by_sentence_total', 
        normal = False)

Multiple sentences total:
Lad der bliver fred. Det går dårligt!
Sentiments = [2.0, -5.751405] 
_____________________________

"""

##############################
### CUSTOM IMPLEMENTATIONS ###
###     STATIC METHODS     ###
##############################

# Function for working around the unicode problem - shoutout to jry
def fix_unicode(df_col):
    return df_col.apply(lambda x: x.encode('raw_unicode_escape').decode('utf-8'))

# Reading the different files and fixing the encoding
aarup = pd.read_csv('aarup.csv', encoding='ISO-8859-1')
intensifier = pd.read_csv('intensifier.csv', encoding='ISO-8859-1')
intensifier['stem'] = fix_unicode(intensifier['stem'])

# Function for modifing sentiment according to the number of exclamation marks:
def exclamation_modifier(sentence):
    ex_counter = sentence.count('!')
    value = 1
#    if ex_counter > 3:
#        ex_counter = 3
    if ex_counter == 0:
        return 1
    for idx, m in enumerate(EX_INTENSITY):
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

# Function for removing punctuation from a sentence and turning it into a 
# list of words
def clean_words_caps(sentence):
    return split_string(punct_cleaner(sentence))

# Function for removing punctuation from a sentence, making the letters lower 
# case, and turning it into a list of words
def clean_words_lower(sentence):
    return split_string(string_to_lower(punct_cleaner(sentence)))

# Function for getting the positions of words that are written in upper case:
def caps_identifier(words):
    positions = []
    for word in words:
        if word.upper() == word:
            positions.append(words.index(word))
    return positions

# Function for modifing the sentiment score of words that are written in 
# upper case:
def caps_modifier(sentiments, words):
    positions = caps_identifier(words)
    for i in range(len(sentiments)):
        if i in positions:
            sentiments[i] *= UP_INTENSITY
    return sentiments

# Function for identifying negations in a list of words. Returns list of 
# positions affected by negator.
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
                    sentiments[inten_pos +
                               1] *= scores[intensifiers.index(word)]

            if inten_pos - 1 not in position:
                position.append(inten_pos - 1)
                if inten_pos - 1 > 0:
                    sentiments[inten_pos -
                               1] *= scores[intensifiers.index(word)]

            if inten_pos + 2 not in position:
                position.append(inten_pos + 2)
                if inten_pos + 2 < len(sentiments):
                    sentiments[inten_pos +
                               2] *= scores[intensifiers.index(word)]

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

# Function for modifying the sentiment score according to whether the words are 
# before or after the word 'men' (but) in a list of words
def men_sentiment(sentiments, words):
    for i in range(len(sentiments)):
        if i < men_identifier(words):
            sentiments[i] *= BUT_INTENSITY[0]
        else:
            sentiments[i] *= BUT_INTENSITY[1]

    return sentiments
# Need imperical tested weights for the part before and after the 'men's'

# Function for stemming the words of a sentence (stemming is NOT optimal for 
# expanding the vocabulary!):
def stemning(words):
    stemmer = SnowballStemmer('danish')
    return [stemmer.stem(word) for word in words]


# Function that takes a list of words as the input and returns the corresponding 
# sentiment scores
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

'''
Function for turning a text input into a mean sentiment score.

Architecture as following tree:
    output: mean -> mean branch
        Analyzes the text as a single sentence
    output: total || by_sentence_mean || by_sentence_total
        Splits into sentences to analyze each as a single sentence
        Splits branch into output branches

'''
def sentidaV2(text, output = ["mean", "total", "by_sentence_mean", "by_sentence_total"], normal = False):
    
    # Goes into sentence splitting if it's not the global mean output
    if output == "by_sentence_mean" or output == "by_sentence_total" or output == "total":
        sentences = nltk.sent_tokenize(text)
        # The tokenizer splits !!! into two sentences if at the end of the text
        # Remove problem by analyzing, appending, and removing
        if sentences[-1] == "!": 
            sentences[-2] = sentences[-2] + "!"
            del sentences[-1]
        sentences_output = []
        
        # Sentence splitting branch
        for sent in sentences:
            words_caps = clean_words_caps(sent)
            words_lower = clean_words_lower(sent)
            stemmed = stemning(words_lower)
            sentiments = get_sentiment(stemmed)
    
            if men_identifier(words_lower) > 0:
                sentiments = men_sentiment(sentiments, words_lower)
        
            sentiments = get_intensifier(sentiments, stemmed)
            sentiments = caps_modifier(sentiments, words_caps)
            
            if question_identifier(sent) == 0:
                for i in set(get_negator_affected(words_lower)):
                    if i < len(sentiments) and i >= 0:
                        sentiments[i] *= -1
        
            if len(words_lower) == 0:
                sentences_output.append(0)
            
            ex_mod = exclamation_modifier(sent)
            sentiments[:] = [sentiment * ex_mod for sentiment in sentiments if sentiment != 0]
            
            if normal:    
                sentiments = np.multiply([float(i) for i in sentiments], ([0.2]*len(sentiments)))
                sentiments = np.where(sentiments < -1, -1, np.where(sentiments > 1, 1, sentiments))
        
            total_sentiment = sum(sentiments)
            if output == "total" or output == "by_sentence_total":
                sentences_output.append(total_sentiment)
            elif output == "by_sentence_mean" and len(sentiments) != 0: sentences_output.append(total_sentiment / len(sentiments))
            else: sentences_output.append(0)
        
        if output == "by_sentence_mean" or output == "by_sentence_total":
            if len(sentences_output) <= 1:
                return sentences_output[0]
            return sentences_output
        elif output == "total":
            return sum(sentences_output)
        else:
            return sentences_output

    elif output == "mean":
        words_caps = clean_words_caps(text)
        words_lower = clean_words_lower(text)
        stemmed = stemning(words_lower)
        sentiments = get_sentiment(stemmed)

        if men_identifier(words_lower) > 0:
            sentiments = men_sentiment(sentiments, words_lower)
    
        sentiments = get_intensifier(sentiments, stemmed)
        sentiments = caps_modifier(sentiments, words_caps)
    
        if question_identifier(text) == 0:
            for i in set(get_negator_affected(words_lower)):
                if i < len(sentiments) and i >= 0:
                    sentiments[i] *= -1
    
        if len(words_lower) == 0:
            sentences_output.append(0)
        
        ex_mod = exclamation_modifier(text)
        sentiments[:] = [sentiment * ex_mod for sentiment in sentiments if sentiment != 0]
        
        if normal:    
            sentiments = np.multiply([float(i) for i in sentiments], ([0.2]*len(sentiments)))
            sentiments = np.where(sentiments < -1, -1, np.where(sentiments > 1, 1, sentiments))
        
        if len(sentiments) > 0: return sum(sentiments) / len(sentiments)
        else: return 0    


def sentidaV2_examples():
    print("_____________________________")
    print("\nExample of usage:\nLad der bliver fred\nSentiment = ", sentidaV2(
        "Lad der blive fred.", output="mean"), "\n_____________________________")
    # Example of usage: 2.0
    print("\nWith exclamation mark:\nLad der blive fred!\nSentiment = ", sentidaV2(
        "Lad der blive fred!", output="mean"), "\n_____________________________")
    # With exclamation mark: 3.13713
    print("\nWith several exclamation mark:\nLad der blive fred!!!\nSentiment = ", sentidaV2(
        "Lad der blive fred!!!", output="mean"), "\n_____________________________")
    # With several exclamation mark:  3.7896530399999997
    print("\nUppercase:\nlad der BLIVE FRED\nSentiment = ", sentidaV2("Lad der BLIVE FRED", output="mean"), "\n_____________________________")
    # Uppercase:  3.466
    print("\nNegative sentence:\nDet går dårligt\nSentiment = ", sentidaV2("Det går dårligt.", output="mean"), "\n_____________________________")
    # With exclamation mark:  -1.8333333333333335
    print("\nNegation in sentence:\nDet går ikke dårligt\nSentiment = ", sentidaV2(
        "Det går ikke dårligt.", output="mean"), "\n_____________________________")
    # Negation in sentence:  1.8333333333333335
    print("\n'Men' ('but'):\nLad der blive fred, men det går dårligt\nSentiment = ", sentidaV2(
        "Lad der blive fred, men det går dårligt.", output = "mean"), "\n_____________________________")
    # 'Men' ('but'):  -1.5
    print("\nNormalized:\nLad der blive fred\nSentiment = ", sentidaV2(
        "Lad der blive fred.", output = "mean", normal = True), "\n_____________________________")
    # Normalized:  0.4
    print("\nMultiple sentences mean:\nLad der bliver fred. Det går dårligt!\nSentiments =", sentidaV2("Lad der bliver fred. Det går dårligt!", "by_sentence_mean"), "\n_____________________________")
    print("\nMultiple sentences total:\nLad der bliver fred. Det går dårligt!\nSentiments =", sentidaV2("Lad der bliver fred. Det går dårligt!", "by_sentence_total"), "\n_____________________________")


sentidaV2_examples()
