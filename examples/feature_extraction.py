# -*- coding: utf-8 -*-
# tweets.py
# author : Antoine Passemiers

import lbsa

reviews = [
    'You should get this game, because in this game you can destroy other cars with really AWESOME guns like a acid thrower',
    'A great dev, but a mediocre app. You just tap the screen . In a word : BORING . Don\'t get this app.',
    'Even at free it was too expensive. Total waste of time and space. Save yourself the trouble of having to remove it by not downloading it in the first place.',
    'Works flawlessly with my favorite stations. I highly recommend this app as it makes finding a stream for your favorite local radio stations a breeze.'
]

afinn_lexicon = lbsa.get_lexicon('opinion', language='english', source='afinn')
nrc_lexicon = lbsa.get_lexicon('opinion', language='english', source='nrc')
nrc_sa_lexicon = lbsa.get_lexicon('sa', language='english', source='nrc')
mpqa_lexicon = lbsa.get_lexicon('opinion', language='english', source='mpqa')

extractor = lbsa.FeatureExtractor(afinn_lexicon, nrc_lexicon, nrc_sa_lexicon, mpqa_lexicon)

print('Feature names:')
print('{}\n'.format(extractor.feature_names))

print(extractor.process(reviews))