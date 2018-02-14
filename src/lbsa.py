# -*- coding: utf-8 -*-
# lbsa.py: lexicon-based sentiment analysis
# author : Antoine Passemiers

import numpy as np
import pandas as pd
import collections
import os
import pickle
import xlrd
import csv

from nltk.tokenize import word_tokenize

try: # Python 3
    from urllib.request import urlretrieve
except ImportError: # Python 2
    from urllib2 import urlretrieve


home = os.path.expanduser("~")
LBSA_DATA_DIR = os.path.join(home, 'lbsa_data')
if not os.path.isdir(LBSA_DATA_DIR):
    os.makedirs(LBSA_DATA_DIR)


class Lexicon:

    def __init__(self, dataframe, sentiment_names, language='english'):
        self.dataframe = dataframe
        self.sentiment_names = sentiment_names
        self.language = language
        self.table = dict()
        self.set_language(self.table, language)

    def set_language(self, table, language):
        sentiments = np.asarray(self.dataframe[self.sentiment_names])
        for key, value in zip(self.dataframe[self.language], sentiments):
            if value.sum() > 0:
                table[key] = value
    
    def get(self, token):
        if token.isdigit():
            return None
        return self.table.get(token)
    
    def get_n_sentiments(self):
        return len(self.sentiment_names)
    
    def get_sentiment_names(self):
        return self.sentiment_names
    
    def get_analysis(self, counters):
        return {
            name: counter for name, counter in zip(self.sentiment_names, counters)
        }
    
    def __len__(self):
        return len(self.dataframe)


def load_nrc_lexicon():
    nrc_filename = "NRC-Emotion-Lexicon-v0.92-InManyLanguages-web"
    if not os.path.exists(os.path.join(LBSA_DATA_DIR, "%s.csv" % nrc_filename)):
        # Download lexicon in XLSX format
        if not os.path.exists(os.path.join(LBSA_DATA_DIR, "%s.xlsx" % nrc_filename)):
            LEXICON_URL = "http://www.saifmohammad.com/WebDocs/%s.xlsx" % nrc_filename
            urlretrieve(LEXICON_URL, os.path.join(LBSA_DATA_DIR, "%s.xlsx" % nrc_filename))

        # Convert from XLSX to CSV file
        wb = xlrd.open_workbook(os.path.join(LBSA_DATA_DIR, "%s.xlsx" % nrc_filename))
        sheet = wb.sheet_by_name('NRC-Emotion-Lexicon-v0.92-InMan')
        with open(os.path.join(LBSA_DATA_DIR, "%s.csv" % nrc_filename), mode='w', encoding='utf8') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            for rownum in range(sheet.nrows):
                writer.writerow(sheet.row_values(rownum))
        
        # Remove XLSX file
        os.remove(os.path.join(LBSA_DATA_DIR, "%s.xlsx" % nrc_filename))

    sentiment_names = ["positive", "negative", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    lexicon_path = os.path.join(__file__, os.path.join(LBSA_DATA_DIR, '%s.csv' % nrc_filename))
    nrc_all_languages = pd.read_csv(lexicon_path, encoding='utf8')
    nrc_all_languages.rename(columns=lambda x: x.replace('Word', '').split('Translation')[0].rstrip(' ').lower(), inplace=True)
    for column_name in sentiment_names:
        nrc_all_languages[column_name] = nrc_all_languages[column_name].astype(np.int32)
    return nrc_all_languages, sentiment_names


def create_lexicon(language='english'):
    nrc_all_languages, sentiment_names = load_nrc_lexicon()
    lexicon = Lexicon(nrc_all_languages, sentiment_names, language=language)
    return lexicon


def sentiment_analysis(text, lexicon):
    tokens = word_tokenize(text)
    n_sentiments = lexicon.get_n_sentiments()
    counters = np.zeros(n_sentiments, dtype=np.int)
    for token in tokens:
        value = lexicon.get(token.lower())
        if value is not None:
            counters += value
    return lexicon.get_analysis(counters)


def time_analysis(text, lexicon):
    tokens = word_tokenize(text)
    n_sentiments = lexicon.get_n_sentiments()
    sentiment_names = lexicon.get_sentiment_names()
    mask = np.zeros((len(tokens), n_sentiments), dtype=np.int)
    for token_id, token in enumerate(tokens):
        value = lexicon.get(token.lower())
        if value is not None:
            mask[token_id, :] += value
    features = {
        key: value for key, value in zip(sentiment_names, mask.T)
    }
    return features