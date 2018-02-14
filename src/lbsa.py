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
download_dir = os.path.join(home, 'lbsa_data')
if not os.path.isdir(download_dir):
    os.makedirs(download_dir)

nrc_filename = "NRC-Emotion-Lexicon-v0.92-InManyLanguages-web"
if not os.path.exists(os.path.join(download_dir, "%s.csv" % nrc_filename)):
    # Download lexicon in XLSX format
    if not os.path.exists(os.path.join(download_dir, "%s.xlsx" % nrc_filename)):
        LEXICON_URL = "http://www.saifmohammad.com/WebDocs/%s.xlsx" % nrc_filename
        urlretrieve(LEXICON_URL, os.path.join(download_dir, "%s.xlsx" % nrc_filename))

    # Convert from XLSX to CSV file
    wb = xlrd.open_workbook(os.path.join(download_dir, "%s.xlsx" % nrc_filename))
    sheet = wb.sheet_by_name('NRC-Emotion-Lexicon-v0.92-InMan')
    with open(os.path.join(download_dir, "%s.csv" % nrc_filename), mode='w', encoding='utf8') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for rownum in range(sheet.nrows):
            writer.writerow(sheet.row_values(rownum))
    
    # Remove XLSX file
    os.remove(os.path.join(download_dir, "%s.xlsx" % nrc_filename))


SENTIMENT_NAMES = ["positive", "negative", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

LEXICON_PATH = os.path.join(__file__, os.path.join(download_dir, '%s.csv' % nrc_filename))
LEXICON_ALL_LANGUAGES = pd.read_csv(LEXICON_PATH, encoding='utf8')
LEXICON_ALL_LANGUAGES.rename(columns=lambda x: x.replace('Word', '').split('Translation')[0].rstrip(' ').lower(), inplace=True)
for column_name in SENTIMENT_NAMES:
    LEXICON_ALL_LANGUAGES[column_name] = LEXICON_ALL_LANGUAGES[column_name].astype(np.int32)


class Analysis:
    def __init__(self, values, n_words):
        self.opinion = {
            'positive': values[0] / n_words,
            'negative': values[1] / n_words,
        }

        self.sentiments = {
            name: values[i] for i, name in zip(range(2, len(SENTIMENT_NAMES)), SENTIMENT_NAMES[2:])
        }
        self.values = values

    def __repr__(self):
        return str({key: value for key, value in zip(SENTIMENT_NAMES, self.values)})
    
    def __str__(self):
        return self.__repr__()


def create_lexicon(language='english'):
    sentiments = np.asarray(LEXICON_ALL_LANGUAGES[SENTIMENT_NAMES])

    lexicon = dict()
    for key, value in zip(LEXICON_ALL_LANGUAGES[language], sentiments):
        if value.sum() > 0:
            lexicon[key] = value
    return lexicon


def sentiment_analysis(text, lexicon):
    tokens = word_tokenize(text)
    counters = np.zeros(len(SENTIMENT_NAMES), dtype=np.int)
    for token in tokens:
        value = lexicon.get(token.lower())
        if value is not None:
            counters += value
    return Analysis(counters, len(tokens))


def time_analysis(text, lexicon):
    tokens = word_tokenize(text)
    mask = np.zeros((len(tokens), len(SENTIMENT_NAMES)), dtype=np.int)
    for token_id, token in enumerate(tokens):
        value = lexicon.get(token.lower())
        if value is not None:
            mask[token_id, :] += value
        print('%i/%i' % (token_id+1, len(tokens)))
    features = {
        key: value for key, value in zip(SENTIMENT_NAMES, mask.T)
    }
    return features