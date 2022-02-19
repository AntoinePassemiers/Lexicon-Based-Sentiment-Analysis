# -*- coding: utf-8 -*-
# lbsa.py: lexicon-based sentiment analysis
# author : Antoine Passemiers

import lbsa

import numpy as np
import matplotlib.pyplot as plt


def moving_average(sequence, n=1000):
    ma = np.cumsum(sequence, axis=0)
    ma[n:] = ma[n:] - ma[:-n]
    return ma[n - 1:] / n


# https://archive.org/stream/thusspokezarathu00nietuoft/thusspokezarathu00nietuoft_djvu.txt
with open('../../data/thus_spoke_zarathustra.txt', 'r') as f:
    text = f.read()
    lexicon = lbsa.get_lexicon('sa', language='english')
    features = lbsa.make_time_analysis(text, lexicon)

    block_size = 100

    tag_names = lexicon.get_tag_names()
    for feature_name in tag_names:
        feature = features[feature_name]
        new_length = len(feature) - (len(feature) % block_size)
        feature = np.mean(feature[:new_length].reshape(-1, block_size), axis=1)
        feature = moving_average(feature, n=100)

        plt.plot(feature, label=feature_name)

    plt.legend()
    plt.ylabel('Average counts', fontsize=15)
    plt.xlabel('Number of blocks (1 block = %i words)' % block_size, fontsize=15)
    plt.title('Sentiment analysis of "Thus spoke Zarathustra" over time', fontsize=15)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('zarathustra.png', dpi=100)
