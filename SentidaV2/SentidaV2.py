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
import codecs
import json
import math
import os
import re
import string
from inspect import getsourcefile
from io import open
from itertools import product

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

######################
### STATIC METHODS ###
######################

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE)
    for word in neg_words:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "n't" in word:
                return True
    '''if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True'''
    return False

def normalize(score, alpha = 15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    # norm_score = score / 5
    if norm_score < -1.0:
         return -1.0
    elif norm_score > 1.0:
         return 1.0
    else:
        return norm_score

def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different

def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar

###############
### CLASSES ###
###############

class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped

class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt"):
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
        with codecs.open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        emoji_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), emoji_lexicon)
        with codecs.open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()

    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
            if not line:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.rstrip('\n').split('\n'):
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for chr in text:
            if chr in self.emojis:
                # get the textual description
                description = self.emojis[chr]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += chr
                prev_space = chr == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for i, item in enumerate(words_and_emoticons):
            valence = 0
            # check for vader_lexicon words that may be used as modifiers or negations
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of"):
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)

        return valence_dict

    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            # get the sentiment valence
            valence = self.lexicon[item_lowercase]

            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == N_TRIGGER and words_and_emoticons[i + 1].lower() in self.lexicon:
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() in NEGATE) \
               or (i > 1 and words_and_emoticons[i - 2].lower() in NEGATE) \
               or (i > 2 and words_and_emoticons[i - 3].lower() in NEGATE and words_and_emoticons[i - 1].lower() in ADD):
                valence = self.lexicon[item_lowercase] * N_SCALAR

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                    if start_i == 2:
                        valence = self._special_idioms_check(valence, words_and_emoticons, i)

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        return sentiments

    def _least_check(self, valence, words_and_emoticons, i):
        # check for negation case using "least"
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            if words_and_emoticons[i - 2].lower() != "at" and words_and_emoticons[i - 2].lower() != "very":
                valence = valence * N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            valence = valence * N_SCALAR
        return valence

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        for but in BUT_DICT:
            if but in words_and_emoticons_lower:
                bi = words_and_emoticons_lower.index(but)
            for sentiment in sentiments:
                    si = sentiments.index(sentiment)
                    if si < bi:
                        sentiments.pop(si)
                        sentiments.insert(si, sentiment * 0.5)
                    elif si > bi:
                        sentiments.pop(si)
                        sentiments.insert(si, sentiment * 1.5)
            return sentiments

    @staticmethod
    def _special_idioms_check(valence, words_and_emoticons, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        onezero = "{0} {1}".format(words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                          words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoone = "{0} {1}".format(words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                           words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwo = "{0} {1}".format(words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[seq]
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
            if zeroone in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroone]
        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                              words_and_emoticons_lower[i + 2])
            if zeroonetwo in SPECIAL_CASE_IDIOMS:
                valence = SPECIAL_CASE_IDIOMS[zeroonetwo]

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in BOOSTER_DICT:
                valence = valence + BOOSTER_DICT[n_gram]
        return valence

    @staticmethod
    def _sentiment_laden_idioms_check(valence, senti_text_lower):
        # Future Work
        # check for sentiment laden idioms that don't contain a lexicon word
        idioms_valences = []
        for idiom in SENTIMENT_LADEN_IDIOMS:
            if idiom in senti_text_lower:
                print(idiom, senti_text_lower)
                valence = SENTIMENT_LADEN_IDIOMS[idiom]
                idioms_valences.append(valence)
        if len(idioms_valences) > 0:
            valence = sum(idioms_valences) / float(len(idioms_valences))
        return valence

    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 1 word preceding lexicon word (w/o stopwords)
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == "never" and \
                    (words_and_emoticons_lower[i - 1] == "so" or
                     words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 2] == "without" and \
                    words_and_emoticons_lower[i - 1] == "doubt":
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 2 words preceding the lexicon word position
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == "never" and \
                    (words_and_emoticons_lower[i - 2] == "so" or words_and_emoticons_lower[i - 2] == "this") or \
                    (words_and_emoticons_lower[i - 1] == "so" or words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 3] == "without" and \
                    (words_and_emoticons_lower[i - 2] == "doubt" or words_and_emoticons_lower[i - 1] == "doubt"):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 3 words preceding the lexicon word position
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * QM_SUCC_MULT
            else:
                qm_amplifier = QM_MULT
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4)}

        return sentiment_dict

####################
### INSTRUCTIONS ###
####################

"""
Initialize a SentimentIntensityAnalyzer object to
begin analysis of your corpus. E.g:

analyzer = SentimentIntensityAnalyzer()
analyzer.polarity_scores(text)

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

# Function for identifying negations in a list of words:
def negator(words):
    position = []

    for word in words:
        if word in NEGATE:
            neg_pos = words.index(word)

            if neg_pos not in position:
                position.append(neg_pos)

            if neg_pos + 1 not in position:
                position.append(neg_pos + 1)

            if neg_pos - 1 not in position:
                position.append(neg_pos - 1)

            if neg_pos + 2 not in position:
                position.append(neg_pos + 2)

            if neg_pos + 3 not in position:
                position.append(neg_pos + 3)

    return position

def get_intensifier(sentiments, word_list):
    intensifiers_df = intensifier.loc[intensifier['stem'].isin(word_list)]
    intensifiers = intensifiers_df['stem'].tolist()
    scores = intensifiers_df['score'].tolist()
    position = []

    for word in word_list:
        if word in intensifiers:
            inten_pos = word_list.index(word)

            if inten_pos not in position:
                position.append(inten_pos)
                sentiments[inten_pos] *= scores[intensifiers.index(word)]

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


#function for identifying 'men' (but) in a list of words:
def men_identifier(words):
    position = 0

    for word in words:
        if word == 'men':
            position = words.index(word)

    return position
# Giver det mening at have sætninger med flere men'er i?
# Potentielt kan det have en effekt ja, men det bliver måske først senere


# Function for modifying the sentiment score according to whether the words are before or after the word 'men' (but) in a list of words
def men_sentiment(sentiments, words):
    for i in range(len(sentiments)):
        if i < men_identifier(words):
            sentiments[i] *= 0.5
        else:
            sentiments[i] *= 1.5

    return sentiments
# We need imperical tested weights for the part before and after the 'men's'


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
        for i in negator(words_lower):
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
# adjusted values for men-sentences, adjusted values for capslock,
# more rated words, more intensifiers/mitigators, better solution than snowball stemmer,
# synonym/antonym dictionary.
# Social media orientated: emojicons, using multiple letters - i.e. suuuuuper.
