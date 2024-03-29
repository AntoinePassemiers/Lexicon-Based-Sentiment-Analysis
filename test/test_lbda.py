# -*- coding: utf-8 -*-
# test-lbsa.py
# author : Antoine Passemiers

import lbsa


def test_lbsa():
    reviews = [
        'You should get this game, because in this game you can destroy other cars with really AWESOME guns like a acid thrower',
        'A great dev, but a mediocre app. You just tap the screen . In a word : BORING . Don\'t get this app.',
        'Even at free it was too expensive. Total waste of time and space. Save yourself the trouble of having to remove it by not downloading it in the first place.',
        'Works flawlessly with my favorite stations. I highly recommend this app as it makes finding a stream for your favorite local radio stations a breeze.'
    ]

    afinn_lexicon = lbsa.get_lexicon('opinion', language='english', source='afinn')
    mpqa_lexicon = lbsa.get_lexicon('opinion', language='english', source='mpqa')

    extractor = lbsa.FeatureExtractor(afinn_lexicon, mpqa_lexicon)
    extractor.process(reviews)
