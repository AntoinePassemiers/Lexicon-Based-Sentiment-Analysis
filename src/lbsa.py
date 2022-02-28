# -*- coding: utf-8 -*-
# lbsa.py: lexicon-based sentiment analysis
# author : Antoine Passemiers

import os
import re
import sys
import io
import pickle
import csv
import zipfile
import shutil
import requests

import numpy as np
import pandas as pd

from urllib.request import urlretrieve


TOKENIZER = re.compile(f'([!"#$%&\'()*+,-./:;<=>?@[\\]^_`|~“”¨«»®´·º½¾¿¡§£₤‘’\n\t])')


class UnknownSource(Exception):

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class LexiconException(Exception):

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class Lexicon:

    def __init__(self, dataframe, tag_names, source, language='english'):
        self.dataframe = dataframe
        self.dataframe.rename(columns={c: Lexicon.reformat_language_name(c) for c in self.dataframe.columns}, inplace=True)
        self.tag_names = tag_names
        self.source = source
        self.language = language
        tags = np.asarray(self.dataframe[self.tag_names])
        self.table = {}
        for language in self.dataframe.columns:
            if language in tag_names:
                continue
            if language.startswith('unnamed'):
                continue

            words = self.dataframe[language]
            if isinstance(words, pd.DataFrame):
                words = words.iloc[:, 0]

            for key, value in zip(words, tags):
                if key not in self.table:
                    self.table[key] = {}
                self.table[key][language] = value

    @staticmethod
    def reformat_language_name(name):
        name = name.lower().strip()
        if '(' in name:
            name = name.split('(')[0].strip()
        return name
    
    def get(self, token):
        return self.table.get(token, None)
    
    def get_n_tags(self):
        return len(self.tag_names)
    
    def get_tag_names(self):
        return self.tag_names

    def process(self, text, as_dict=True):
        tokens = tokenize(text) if not isinstance(text, list) else text
        n_tags = self.get_n_tags()
        language_counts = {}
        counts = {}
        for token in tokens:
            results = self.get(token.lower())
            if results is not None:
                for language in results.keys():
                    if language not in counts:
                        counts[language] = np.zeros(n_tags, dtype=np.int)
                    counts[language] += results[language]
                    if language not in language_counts:
                        language_counts[language] = 0
                    language_counts[language] += 1

        if len(counts) == 0:
            counters = np.zeros(n_tags, dtype=np.int)
        else:
            # Select language
            if self.language == 'auto':
                languages = list(language_counts.keys())
                total_counts = [language_counts[language] for language in languages]
                language = languages[np.argmax(total_counts)]
                counters = counts[language]
            else:
                language = self.language
            if language not in counts:
                # raise LexiconException(f'Could not find language "{language}". Found: {counts.keys()}')
                counters = np.zeros(n_tags, dtype=np.int)
            else:
                counters = counts[language]

        if as_dict:
            data = { name: counter for name, counter in zip(self.tag_names, counters) }
            data['lang'] = language
            return data
        else:
            return counters
    
    def __len__(self):
        return len(self.dataframe)


class FeatureExtractor:

    def __init__(self, *args):
        self.lexicons = list(args)
        self.sizes = [lexicon.get_n_tags() for lexicon in self.lexicons]
        self.offsets = np.cumsum([0] + self.sizes)
        self.n_features = sum(self.sizes)
        self.feature_names = list()
        for lexicon in self.lexicons:
            tag_names = [lexicon.source + '_' + name for name in lexicon.get_tag_names()]
            self.feature_names += tag_names

    def process(self, X):
        if isinstance(X, str):
            X = [X]
        elif len(X) == 0:
            return list()
        features = np.empty((len(X), self.n_features))
        for i, text in enumerate(X):
            tokens = tokenize(text)
            for j, lexicon in enumerate(self.lexicons):
                features[i, self.offsets[j]:self.offsets[j+1]] = lexicon.process(tokens, as_dict=False)
        return np.squeeze(features)


def make_time_analysis(text, lexicon):
    if isinstance(text, list):
        tokens = text
    else:
        tokens = tokenize(text)
    n_tags = lexicon.get_n_tags()
    tag_names = lexicon.get_tag_names()
    mask = np.zeros((len(tokens), n_tags), dtype=np.int)
    for token_id, token in enumerate(tokens):
        value = lexicon.get(token.lower())
        if value is not None:
            mask[token_id, :] += value
    data = { key: value for key, value in zip(tag_names, mask.T) }
    return data


class DownloadProgressBar:

    def __init__(self, prefix, length=30):
        self.prefix = prefix
        self.length = length
        self.downloaded = 0
        self.total_size = None
        self.update(0)

    def progress_hook(self, count, block_size, total_size):
        self.total_size = total_size
        self.downloaded += block_size
        progress = np.clip(float(self.downloaded) / float(self.total_size), 0., 1.)
        self.update(progress)

    def update(self, progress):
        percent = 100. * progress
        n_blocks = int(np.round(progress * self.length))
        bar = ('=' * n_blocks).ljust(self.length)
        sys.stdout.write('\r%s [%s] %.2f %%\r' % (self.prefix, bar, percent))
        if self.downloaded == self.total_size:
            sys.stdout.write('\n')


def get_cache_dir():
    home = os.path.expanduser("~")
    LBSA_DATA_DIR = os.path.join(home, '.lbsa')
    if not os.path.isdir(LBSA_DATA_DIR):
        os.makedirs(LBSA_DATA_DIR)
    return LBSA_DATA_DIR


def load_nrc_lexicon(path=None):
    LBSA_DATA_DIR = get_cache_dir()
    nrc_filename = 'NRC-Emotion-Lexicon'

    def download_lexicon():
        if path is None:
            LEXICON_URL = f'http://www.saifmohammad.com/WebDocs/Lexicons/{nrc_filename}.zip'
            print('Downloading NRC lexicon...')
            req = requests.get(LEXICON_URL)
            with open(os.path.join(LBSA_DATA_DIR, f'{nrc_filename}.zip'), 'wb') as f:
                f.write(req.content)
            with zipfile.ZipFile(os.path.join(LBSA_DATA_DIR, f'{nrc_filename}.zip'), 'r') as zip_object:
                zipObject.extract(
                    os.path.join('NRC-Emotion-Lexicon-v0.92', 'NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx'),
                    os.path.join(LBSA_DATA_DIR, f'{nrc_filename}.xlsx'))
        else:
            shutil.copyfile(path, os.path.join(LBSA_DATA_DIR, f'{nrc_filename}.xlsx'))

    if not os.path.exists(os.path.join(LBSA_DATA_DIR, f'{nrc_filename}.csv')):
        # Download lexicon in XLSX format
        if not os.path.exists(os.path.join(LBSA_DATA_DIR, f'{nrc_filename}.xlsx')):
            download_lexicon()

        # Convert from XLSX to CSV file
        filepath = os.path.join(LBSA_DATA_DIR, f'{nrc_filename}.xlsx')
        df = pd.read_excel(filepath, sheet_name='NRC-Lex-v0.92-word-translations')
        df.to_csv(os.path.join(LBSA_DATA_DIR, f'{nrc_filename}.csv'))
        
        # Remove XLSX file
        os.remove(os.path.join(LBSA_DATA_DIR, "%s.xlsx" % nrc_filename))

    sentiment_names = ["positive", "negative", "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    lexicon_path = os.path.join(__file__, os.path.join(LBSA_DATA_DIR, '%s.csv' % nrc_filename))
    nrc_all_languages = pd.read_csv(lexicon_path, encoding='utf8', index_col=False)
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


def load_mpqa_sujectivity_lexicon(name='', organization='', email='', path=None):
    assert path is None
    LBSA_DATA_DIR = get_cache_dir()
    if not os.path.isdir(os.path.join(LBSA_DATA_DIR, "mpqa")):
        os.makedirs(os.path.join(LBSA_DATA_DIR, "mpqa"))
    filepath = os.path.join(LBSA_DATA_DIR, 'mpqa/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')
    if not os.path.exists(filepath):
        print('Downloading mpqa lexicon...')
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
        

def load_afinn_opinion_lexicon(path=None):
    LBSA_DATA_DIR = get_cache_dir()
    if not os.path.isdir(os.path.join(LBSA_DATA_DIR, "afinn")):
        os.makedirs(os.path.join(LBSA_DATA_DIR, "afinn"))
    if not os.path.exists(os.path.join(LBSA_DATA_DIR, "afinn/AFINN/AFINN-111.txt")):

        if path is None:
            # Download zip archive
            LEXICON_URL = 'http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/6010/zip/imm6010.zip'
            filepath = os.path.join(LBSA_DATA_DIR, "afinn/imm6010.zip")
            progressbar = DownloadProgressBar('Downloading AFINN lexicon')
            urlretrieve(LEXICON_URL, filepath, reporthook=progressbar.progress_hook)
            print('')
            with zipfile.ZipFile(filepath) as zf:
                zf.extractall(path=os.path.join(LBSA_DATA_DIR, "afinn"))
            # Remove zip archive
            os.remove(filepath)
        else:
            shutil.copyfile(path, os.path.join(LBSA_DATA_DIR, 'afinn/AFINN/AFINN-111.txt'))
    
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


def create_sa_lexicon(source='nrc', language='english', path=None):
    if source == 'nrc':
        nrc_all_languages, tag_names = load_nrc_lexicon(path=path)
        to_remove = ['positive', 'negative']
        nrc_all_languages.drop(to_remove, axis=1, inplace=True)
        for tag_name in to_remove:
            tag_names.remove(tag_name)
        lexicon = Lexicon(nrc_all_languages, tag_names, source, language=language)
    else:
        raise UnknownSource('Source %s does not provide any available sentiment analysis lexicon')
    return lexicon


def create_opinion_lexicon(source='nrc', language='english', path=None):
    if source == 'nrc':
        nrc_all_languages, tag_names = load_nrc_lexicon(path=path)
        to_remove = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        nrc_all_languages.drop(to_remove, axis=1, inplace=True)
        for tag_name in to_remove:
            tag_names.remove(tag_name)
        lexicon = Lexicon(nrc_all_languages, tag_names, source, language=language)
    elif source == 'afinn':
        ol = load_afinn_opinion_lexicon(path=path)
        lexicon = Lexicon(ol, ['positive', 'negative'], source, language=language)
    elif source == 'mpqa':
        ol = load_mpqa_sujectivity_lexicon(path=path)
        lexicon = Lexicon(ol, ['positive', 'negative', 'strong_subjectivty'], source, language=language)
    else:
        raise UnknownSource('Source %s does not provide any available opinion/subjectivity lexicon')
    return lexicon


def get_lexicon(lexicon_type, **kwargs):
    assert(lexicon_type in ['sa', 'opinion'])
    if lexicon_type == 'sa':
        return create_sa_lexicon(**kwargs)
    else:
        return create_opinion_lexicon(**kwargs)
