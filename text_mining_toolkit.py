import pandas as pd
import numpy as np
import re
import string
from emoji import UNICODE_EMOJI
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import string
from sklearn.feature_extraction.text import CountVectorizer
from emoji import UNICODE_EMOJI

class preprocess_data():
    """ 
    ************ PREPROCESS THE DATA ****************

    Preprocess a specific text.

    ========= INPUTS =========

    - text              : <string> Text to be preprocessed.
    - lowercase         : True if want to convert text to lowercase
    - rem_stopwords     : True if want to remove stopwords
    - lemmatize         : True if want to lemmatize
    - remove_html       : True if want to remove HTML tags/language
    - remove_punctuation  : True if want to remove punctuation !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    - domain_sp_stopwrd : List of domain specific stopwords to remove. Ex: ['X1', 'YX2', ...] 
    - regex_sub         : List of regular expresions to remove. Ex: ['[^a-zA-Z]']

    ========= EXAMPLE =========
    
    ppd       = preprocess_data(lang = 'english')
    clean_txt = preprocess_text("The people is completly vulnerable by the media!. And you no it, don't you? hahaha"
            , lowercase= True
            , rem_stopwords= True
            , lemmatize= False
            , remove_html = True
            , remove_punctuation = True
            , domain_sp_stopwrd=['media']
            , regex_sub=['ha+'])

    """
    def __init__(self, lang = 'english'):
        self.lang = lang

    def preprocess_text(self, text, lowercase = True, rem_stopwords = False, lemmatize = False, remove_html = False, remove_punctuation = False, domain_sp_stopwrd = [], regex_sub = []):
        if lowercase:
            text = text.strip().lower()

        if len(regex_sub) > 0:
            for rg in regex_sub:
                text = re.sub(rg, '', text)

        if remove_html:
            text = BeautifulSoup(text).get_text()

        if remove_punctuation:
            other_punt = "\“\”\¡\¿\«\»\…\—\−"
            if self.lang == "english":
                text = re.sub(rf"[{string.punctuation}{other_punt}]", '', text)
            elif self.lang == "portuguese":
                text = re.sub(rf'[{string.punctuation}{other_punt}]', '', text)
            elif self.lang == "spanish":
                text = re.sub(rf'[{string.punctuation}{other_punt}]', '', text)
            else:
                text = re.sub(rf'[{string.punctuation}{other_punt}]', '', text)

        if len(domain_sp_stopwrd) > 0:
            for dst in domain_sp_stopwrd:
                r = dst.lower()
                text = re.sub(rf"\b{r}\b", '', text.lower())
            
        stop_words = set(stopwords.words(self.lang))
        if rem_stopwords:
            word_tokens = word_tokenize(text)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            text = ' '.join(filtered_sentence)

        if lemmatize:
            if self.lang == 'english':
                text = text.split()
                lem = WordNetLemmatizer()
                text = [lem.lemmatize(w) for w in text if not w in stop_words] 
                text = ' '.join(text)
            else:
                print("Lemmatization only works with English language")

        return text


# ==============================================================================================================

class txt_mining_tool():
    """
    ************* SET OF TEXT MINING TOOLS ***************

    - top_ngram() : Count the number of words by n-gram. Frequency of 2 words, 3 words, ....
    - plural_word() : Function that converts a list of words in its plural form
    - emojis_freq() : Return the sorted list of most frequently used emojis in a text
    - number_of_emojies() : Function that count the number of emojies in a text.
    - emojis_freq() : Return the sorted list of most frequently used emojis in a text
    - count_chars() : Function that count the number of characters in a text excluding spaces.
    - per_ortography() : Function that compute proxy of ortography.
    - perc_puntuation() : Function that compute the percentage of puntation of a text.
    """

    def __init__(self, lang = 'english'):
        self.lang = lang

    def top_ngram(self, corpus, n_gram):
        """ Count the number of words by n-gram. Frequency of 2 words, 3 words, ....
            - Corpus : List of values with the text to be analyzed Ex: ["Hi Im Juan", "Hello, Im Maria", "Nice to meet you"]
            - n_gram : Number of words.
        """

        vec = CountVectorizer(ngram_range=(n_gram, n_gram), max_features=2000).fit(corpus)
        bag_of_words = vec.transform(corpus)
            
        sum_words = bag_of_words.sum(axis=0) 

        words_freq = []
        for word, idx in vec.vocabulary_.items():
            words_freq.append((word, sum_words[0, idx]))
            
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_df = pd.DataFrame(words_freq)
        top_df.columns = ["Ngram", "Frequency"]
        return top_df
        
    def plural_word(self, list_of_words):
        """ Function that converts a list of words in its plural form"""

        if self.lang == 'spanish':
            plurals = []
            grp_1 = ["í","ó","ú","á","d","l","r","n","b","c","y","j"]
            grp_2 = ["a","e","i","o","u","é",]
            grp_3 = ["s", "x"]
            for word in list_of_words:
                wd     = word.lower()
                last_1 = wd[-1]
                if last_1 == "z":
                    root_wd = wd[:len(wd)-1]
                    plural  = root_wd + "ces"
                elif last_1 in grp_1:
                    root_wd = wd
                    plural  = root_wd + "es"
                elif last_1 in grp_2:
                    root_wd = wd
                    plural  = root_wd + "s"
                elif last_1 in grp_3:
                    root_wd = wd
                    plural  = root_wd
                else:
                    root_wd = wd
                    plural  = root_wd + "s"

                plurals.append(plural)

            return plurals
        else:
            print("(!) Language not supported")
            return []


    def emojis_freq(self, text):
        """ Return the sorted list of most frequently used emojis in a text"""
        df_emojis = [i for i in text if i in UNICODE_EMOJI]
        df_emojis = pd.Series(df_emojis).value_counts()
        df_emojis = df_emojis.sort_values(ascending = False)
        
        return df_emojis

    def number_of_emojies(self, text):
        """ Function that count the number of emojies in a text. """
        counter = 0
        for character in text:
            if character in UNICODE_EMOJI:
                counter += 1
        return counter

    def count_chars(self, text):
        """ Function that count the number of characters in a text excluding spaces. """
        return len(text) - text.count(" ")

    def word_counter(self, text):
        """ Function that count the number of times all the words appear in text"""
        all_words = text.split()
        freq = pd.Series(all_words).value_counts()
        return freq

    def perc_puntuation(self, text):
        """ Function that compute the percentage of puntation of a text. """
        punt_cnt = sum([1 for char in text if char in string.punctuation])
        return round(punt_cnt / (len(text) - text.count(" ")),3)*100

    def per_ortography(self, text):
        """ Function that compute proxy of ortography. """
        punt_cnt = sum([1 for char in text if char in 'áéíóúÁÉÍÓÚ'])
        return round(punt_cnt / (len(text) - text.count(" ")),3)*100