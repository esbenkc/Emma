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
Or the SentidaV1 team:
    - gustavaarup0111@gmail.com
    - jacdals@hotmail.com
    - larskjartanbachersvendsen@gmail.com

SENTIDA v2.
Aarhus University, Cognitive Science.
2019 - Cognition & Communication.
@authors: sorenorm & esbenkc.

This script was developed along with other tools in an attempt to improve 
danish sentiment analysis. The tool will be updated as more data is collected 
and new methods for more optimally accessing sentiment is developed.


VADER BASIS VALUES

Multiplication values:
    0.291, 0.215, and 0.208 for !, !!, and !!! respectively
        empirically tested by one sentence compared to the three conditions
    0.733 for uppercase 
        empirically tested from single control sentence to uppercase version
    0.293 for degree modifications from adverbs
        empirically tested with "extremely"


SENTIDA V2 BASIS VALUES

Currently using VADER basis values
Question mark is: XXX
Degree modifications for other words are implemented in intensitifer list
    - Need implementation of larger intensifier list based on sentences
    
    
FUTURE IMPROVEMENTS

Still missing: common phrases, adjusted values for exclamation marks,
Adjusted values for men-sentences, adjusted values for uppercase,
More rated words, more intensifiers/mitigators, better solution than snowball stemmer,
Synonym/antonym dictionary.
Social media orientated: emoticons, using multiple letters - i.e. suuuuuper.
Normalization with respect to sub-(-1) and super-(1) output values

'''

#################
### LIBRARIES ###
#################

import pandas as pd, nltk, numpy as np
from nltk.stem import SnowballStemmer
from collections import namedtuple

#################
### CONSTANTS ###
#################

SV2STRUCT = namedtuple("SV2STRUCT", """B_INCR, B_DECR, C_INCR, N_SCALAR, 
                       QM_MULT, QM_SUCC_MULT, EX_INTENSITY, UP_INTENSITY,
                       BUT_INTENSITY, N_TRIGGER, NEGATE, ADD, BUT_DICT,
                       BOOSTER_DICT, SENTIMENT_LADEN_IDIOMS, SPECIAL_CASE_IDIOMS
                       """)
SV2STRUCT.B_INCR = 0.293
SV2STRUCT.B_DECR = -0.293
SV2STRUCT.C_INCR = 0.733
SV2STRUCT.N_SCALAR = -0.74
SV2STRUCT.QM_MULT = 0.94
SV2STRUCT.QM_SUCC_MULT = 0.18
SV2STRUCT.EX_INTENSITY = [1.291, 1.215, 1.208]
SV2STRUCT.UP_INTENSITY = 1.733
SV2STRUCT.BUT_INTENSITY = [0.5, 1.5]
SV2STRUCT.N_TRIGGER = "no"
SV2STRUCT.NEGATE = \
    ['ikke', 'ik', 'ikk', 'ik\'', 'aldrig', 'ingen']
SV2STRUCT.ADD = \
    ['og', 'eller']
SV2STRUCT.BUT_DICT = \
    ['men', 'dog']
SV2STRUCT.BOOSTER_DICT = \
    {"temmelig": 0.1, "meget": 0.2, "mega": 0.4, "lidt": -0.2, "ekstremt": 0.4,
     "totalt": 0.2, "utrolig": 0.3, "rimelig": 0.1, "seriøst": 0.3}
SV2STRUCT.SENTIMENT_LADEN_IDIOMS = {}
SV2STRUCT.SPECIAL_CASE_IDIOMS = {}

#####################
### DOCUMENTATION ###
#####################

"""

sentidaV2(
        text = string,                      Text to analyze
        output = ['mean',                   Returns as complete text mean
                  'total',                  Returns as complete text total
                                              Calculated by sentence and summed
                  'by_sentence_mean',       Returns list of mean sentiment by sentence
                  'by_sentence_total'],     Returns list of total sentiment by sentence
        normal = True                       Normalize sentiment score (-1 to 1)
        )


USAGE EXAMPLES:

Single sentence:
    sentidaV2(
            text = 'Lad der blive fred.', 
            output = 'mean', 
            normal = False)
    
    Output: 
        # 2.0

Multiple sentences normalized:
    sentidaV2(
            text = 'Lad der bliver fred. Det går dårligt!', 
            output = 'by_sentence_total', 
            normal = True)
    
    Output:
        # [0.4, -0.895429]

"""

##############################
### CUSTOM IMPLEMENTATIONS ###
###     STATIC METHODS     ###
##############################

class SentidaV2():
    
    # Function for working around the unicode problem - shoutout to jry
    @staticmethod
    def fix_unicode(df_col):
        return df_col.apply(lambda x: x.encode('raw_unicode_escape').decode('utf-8'))
    
    # Returns multiplier by amount of exclamation marks in input
    @staticmethod
    def exclamation_modifier(sentence):
        ex_counter = sentence.count('!')
        value = 1
        if ex_counter == 0:
            return 1
        for idx, m in enumerate(SV2STRUCT.EX_INTENSITY):
            if idx <= ex_counter:
                value *= m
        return value
    
    # Returns amount of question marks in input
    @staticmethod
    def question_identifier(text):
        return text.count('?')
    
    # Returns input without punctuation
    @staticmethod
    def punct_cleaner(text):
        table = str.maketrans('!?-+_#.,;:\'\"', 12*' ')
        return text.translate(table)
    
    # Function for removing punctuation from a sentence and turning it into a 
    # list of words
    @staticmethod
    def clean_words_upper(sentence):
        return SentidaV2.punct_cleaner(sentence).split()
    
    # Function for removing punctuation from a sentence, making the letters lower 
    # case, and turning it into a list of words
    @staticmethod
    def clean_words_lower(text):
        return SentidaV2.punct_cleaner(text).lower().split()
    
    # Function for getting the positions of words that are written in upper case:
    @staticmethod
    def caps_identifier(words):
        positions = []
        for word in words:
            if word.upper() == word:
                positions.append(words.index(word))
        return positions
    
    # Function for modifing the sentiment score of words that are written in 
    # upper case:
    @staticmethod
    def caps_modifier(sentiments, words):
        positions = SentidaV2.caps_identifier(words)
        for i in range(len(sentiments)):
            if i in positions:
                sentiments[i] *= SV2STRUCT.UP_INTENSITY
        return sentiments
    
    # Function for identifying negations in a list of words. Returns list of 
    # positions affected by negator.
    @staticmethod
    def get_negator_affected(words):
        positions = []
    
        for word in words:
            if word in SV2STRUCT.NEGATE:
                neg_pos = words.index(word)
                positions.append(neg_pos)
                positions.append(neg_pos + 1)
                positions.append(neg_pos - 1)
                positions.append(neg_pos + 2)
                positions.append(neg_pos + 3)
        return positions
    
    # Get all intensifiers
    @staticmethod
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
    
    
    # Locates 'men' (but) in a list of words:
    @staticmethod
    def men_identifier(words):
        position = 0
        for word in words:
            if word == 'men':
                position = words.index(word)
        return position
    
    # Modifies sentiment of words before or after 'men' (but) in list of words
    @staticmethod
    def men_sentiment(sentiments, words):
        for i in range(len(sentiments)):
            if i < SentidaV2.men_identifier(words):
                sentiments[i] *= SV2STRUCT.BUT_INTENSITY[0]
            else:
                sentiments[i] *= SV2STRUCT.BUT_INTENSITY[1]
    
        return sentiments
    # Need empirically tested weights for the part before and after the 'men's'
    
    # Stems words of sentence 
    @staticmethod
    def stemning(words):
        stemmer = SnowballStemmer('danish')
        return [stemmer.stem(word) for word in words]
    # Snowball has limited Danish stemmer
    
    # Returns sentiment scores of input list by the aarup dictionary
    @staticmethod
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

# Reading the sentiment dictionary files and fixing the encoding
aarup = pd.read_csv('aarup.csv', encoding='ISO-8859-1')
intensifier = pd.read_csv('intensifier.csv', encoding='ISO-8859-1')
intensifier['stem'] = SentidaV2.fix_unicode(intensifier['stem'])

'''
Turns text input into a sentiment score.

Method works as follows:
    output - mean:
        Analyzes the text as a single sentence
    output - total || by_sentence_mean || by_sentence_total
        Splits into sentences to analyze each as a single sentence
        Splits branch into output branches

'''
def sentidaV2(text, output = ["mean", "total", "by_sentence_mean", "by_sentence_total"], normal = True):
    
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
            words_upper = SentidaV2.clean_words_upper(sent)
            words_lower = SentidaV2.clean_words_lower(sent)
            stemmed = SentidaV2.stemning(words_lower)
            sentiments = SentidaV2.get_sentiment(stemmed)
    
            if SentidaV2.men_identifier(words_lower) > 0:
                sentiments = SentidaV2.men_sentiment(sentiments, words_lower)
        
            sentiments = SentidaV2.get_intensifier(sentiments, stemmed)
            sentiments = SentidaV2.caps_modifier(sentiments, words_upper)
            
            if SentidaV2.question_identifier(sent) == 0:
                for i in set(SentidaV2.get_negator_affected(words_lower)):
                    if i < len(sentiments) and i >= 0:
                        sentiments[i] *= -1
        
            if len(words_lower) == 0:
                sentences_output.append(0)
            
            ex_mod = SentidaV2.exclamation_modifier(sent)
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
        words_upper = SentidaV2.clean_words_upper(text)
        words_lower = SentidaV2.clean_words_lower(text)
        stemmed = SentidaV2.stemning(words_lower)
        sentiments = SentidaV2.get_sentiment(stemmed)

        if SentidaV2.men_identifier(words_lower) > 0:
            sentiments = SentidaV2.men_sentiment(sentiments, words_lower)
    
        sentiments = SentidaV2.get_intensifier(sentiments, stemmed)
        sentiments = SentidaV2.caps_modifier(sentiments, words_upper)
    
        if SentidaV2.question_identifier(text) == 0:
            for i in set(SentidaV2.get_negator_affected(words_lower)):
                if i < len(sentiments) and i >= 0:
                    sentiments[i] *= -1
    
        if len(words_lower) == 0:
            sentences_output.append(0)
        
        ex_mod = SentidaV2.exclamation_modifier(text)
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
    print("\nNon-normalized:\nLad der blive fred\nSentiment = ", sentidaV2(
        "Lad der blive fred.", output = "mean", normal = False), "\n_____________________________")
    # Normalized:  0.4
    print("\nMultiple sentences mean:\nLad der bliver fred. Det går dårligt!\nSentiments =", sentidaV2("Lad der bliver fred. Det går dårligt!", "by_sentence_mean"), "\n_____________________________")
    print("\nMultiple sentences total:\nLad der bliver fred. Det går dårligt!\nSentiments =", sentidaV2("Lad der bliver fred. Det går dårligt!", "by_sentence_total"), "\n_____________________________")


sentidaV2_examples()