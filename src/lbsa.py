# -*- coding: utf-8 -*-
# lbsa.py: lexicon-based sentiment analysis
# author : Antoine Passemiers

import numpy as np
import pandas as pd
import os
import re
import io
import pickle
import xlrd
import csv
import zipfile
import requests

try: # Python 3
    from urllib.request import urlretrieve
except ImportError: # Python 2
    from urllib2 import urlretrieve


TOKENIZER = re.compile(f'([!"#$%&\'()*+,-./:;<=>?@[\\]^_`|~“”¨«»®´·º½¾¿¡§£₤‘’])')


class UnknownSource(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


# http://sentiment.christopherpotts.net/lexicons.html#resources

class Lexicon:

    def __init__(self, dataframe, tag_names, source, language='english'):
        self.dataframe = dataframe
        self.tag_names = tag_names
        self.source = source
        self.language = language
        self.table = dict()
        self.set_language(self.table, language)

    def set_language(self, table, language):
        tags = np.asarray(self.dataframe[self.tag_names])
        for key, value in zip(self.dataframe[self.language], tags):
            if value.sum() != 0:
                table[key] = value
    
    def get(self, token):
        if token.isdigit():
            return None
        return self.table.get(token)
    
    def get_n_tags(self):
        return len(self.tag_names)
    
    def get_tag_names(self):
        return self.tag_names
    
    def get_analysis(self, counters):
        return {
            name: counter for name, counter in zip(self.tag_names, counters)
        }
    
    def __len__(self):
        return len(self.dataframe)


def get_cache_dir():
    home = os.path.expanduser("~")
    LBSA_DATA_DIR = os.path.join(home, '.lbsa')
    if not os.path.isdir(LBSA_DATA_DIR):
        os.makedirs(LBSA_DATA_DIR)
    return LBSA_DATA_DIR


def load_nrc_lexicon():
    LBSA_DATA_DIR = get_cache_dir()
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


"""
def load_bing_opinion_lexicon():
    LBSA_DATA_DIR = get_cache_dir()
    bing_filename = "opinion-lexicon-English"
    if not os.path.isdir(os.path.join(LBSA_DATA_DIR, "bing")):
        os.makedirs(os.path.join(LBSA_DATA_DIR, "bing"))
    if not os.path.exists(os.path.join(LBSA_DATA_DIR, "bing/positive.txt")):
        # Download rar archive
        LEXICON_URL = "http://www.cs.uic.edu/~liub/FBS/%s.rar" % bing_filename
        filepath = os.path.join(LBSA_DATA_DIR, "%s.rar" % bing_filename)
        urlretrieve(LEXICON_URL, filepath)
        rar = rarfile.RarFile(filepath)
        rar.extractall(path=os.path.join(LBSA_DATA_DIR, "bing"))
        # TODO
"""


def load_mpqa_sujectivity_lexicon(name='', organization='', email=''):
    LBSA_DATA_DIR = get_cache_dir()
    if not os.path.isdir(os.path.join(LBSA_DATA_DIR, "mpqa")):
        os.makedirs(os.path.join(LBSA_DATA_DIR, "mpqa"))
    filepath = os.path.join(LBSA_DATA_DIR, 'mpqa/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')
    if not os.path.exists(filepath):
        response = requests.post(
            "http://mpqa.cs.pitt.edu/request_resource.php",
            data={"name": "", "organization": "", "email": "", "dataset":"subj_lexicon"})
        zf = zipfile.ZipFile(io.BytesIO(response.content))
        zf.extractall(path=os.path.join(LBSA_DATA_DIR, "mpqa"))
    
    with open(filepath) as f:
        words, positive, negative, strong_subj = list(), list(), list(), list()
        for line in f.readlines():
            items = line.rstrip().split(' ')
            if len(items) == 6:
                words.append(items[2].split('=')[1])
                strong_subj.append(1 if (items[0].split('=')[1] == 'strongsubj') else 0)
                positive.append(1 if items[5].split('=')[1] in ['positive', 'both'] else 0)
                negative.append(1 if items[5].split('=')[1] in ['negative', 'both'] else 0)
    return pd.DataFrame({
        'english': words,
        'positive': np.asarray(positive, dtype=np.int),
        'negative': np.asarray(negative, dtype=np.int),
        'strong_subjectivty': np.asarray(strong_subj, dtype=np.int)
    })
        

def load_afinn_opinion_lexicon():
    LBSA_DATA_DIR = get_cache_dir()
    if not os.path.isdir(os.path.join(LBSA_DATA_DIR, "afinn")):
        os.makedirs(os.path.join(LBSA_DATA_DIR, "afinn"))
    if not os.path.exists(os.path.join(LBSA_DATA_DIR, "afinn/AFINN/AFINN-111.txt")):
        # Download zip archive
        LEXICON_URL = 'http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/6010/zip/imm6010.zip'
        filepath = os.path.join(LBSA_DATA_DIR, "afinn/imm6010.zip")
        urlretrieve(LEXICON_URL, filepath)
        with zipfile.ZipFile(filepath) as zf:
            zf.extractall(path=os.path.join(LBSA_DATA_DIR, "afinn"))
        # Remove zip archive
        os.remove(filepath)
    
    words, values = list(), list()
    with open(os.path.join(LBSA_DATA_DIR, 'afinn/AFINN/AFINN-111.txt')) as f:
        for line in f.readlines():
            items = line.rstrip().split('\t')
            if len(items) == 2:
                words.append(items[0])
                values.append(int(items[1]))
    values = np.asarray(values, dtype=np.int)
    positives = np.zeros(len(values), dtype=np.int)
    negatives = np.zeros(len(values), dtype=np.int)
    positives[values > 0] = values[values > 0]
    negatives[values < 0] = -values[values < 0]
    return pd.DataFrame({
        'english': words,
        'positive': positives,
        'negative': negatives
    })


def tokenize(text):
    return TOKENIZER.sub(r' \1 ', text).split()


def create_sa_lexicon(source='nrc', language='english'):
    if source == 'nrc':
        nrc_all_languages, tag_names = load_nrc_lexicon()
        to_remove = ['positive', 'negative']
        nrc_all_languages.drop(to_remove, axis=1, inplace=True)
        for tag_name in to_remove:
            tag_names.remove(tag_name)
        lexicon = Lexicon(nrc_all_languages, tag_names, source, language=language)
    else:
        raise UnknownSource('Source %s does not provide any available sentiment analysis lexicon')
    return lexicon


def create_opinion_lexicon(source='nrc', language='english'):
    if source == 'nrc':
        nrc_all_languages, tag_names = load_nrc_lexicon()
        to_remove = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
        nrc_all_languages.drop(to_remove, axis=1, inplace=True)
        for tag_name in to_remove:
            tag_names.remove(tag_name)
        lexicon = Lexicon(nrc_all_languages, tag_names, source, language=language)
    elif source == 'afinn':
        ol = load_afinn_opinion_lexicon()
        lexicon = Lexicon(ol, ['positive', 'negative'], source, language=language)
    elif source == 'mpqa':
        ol = load_mpqa_sujectivity_lexicon()
        lexicon = Lexicon(ol, ['positive', 'negative', 'strong_subjectivty'], source, language=language)
    else:
        raise UnknownSource('Source %s does not provide any available opinion/subjectivity lexicon')
    return lexicon


def make_analysis(text, lexicon, as_dict=True):
    tokens = tokenize(text)
    n_tags = lexicon.get_n_tags()
    counters = np.zeros(n_tags, dtype=np.int)
    for token in tokens:
        value = lexicon.get(token.lower())
        if value is not None:
            counters += value
    return lexicon.get_analysis(counters) if as_dict else counters


def make_time_analysis(text, lexicon):
    tokens = tokenize(text)
    n_tags = lexicon.get_n_tags()
    tag_names = lexicon.get_tag_names()
    mask = np.zeros((len(tokens), n_tags), dtype=np.int)
    for token_id, token in enumerate(tokens):
        value = lexicon.get(token.lower())
        if value is not None:
            mask[token_id, :] += value
    data = {
        key: value for key, value in zip(tag_names, mask.T)
    }
    return data