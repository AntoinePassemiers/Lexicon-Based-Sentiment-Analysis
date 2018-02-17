# -*- coding: utf-8 -*-
# tweets.py
# author : Antoine Passemiers

import lbsa

tweet = """
The Budget Agreement today is so important for our great Military.
It ends the danger sequester and gives Secretary Mattis what he needs to keep America Great.
Republicans and Democrats must support our troops and support this Bill!
"""

print('\nUse afinn lexicon')
lexicon = lbsa.create_opinion_lexicon(language='english', source='afinn')
print(lbsa.make_analysis(tweet, lexicon))

print('\nUse nrc lexicon')
lexicon = lbsa.create_opinion_lexicon(language='english', source='nrc')
print(lbsa.make_analysis(tweet, lexicon))

print('\nUse mpqa lexicon')
lexicon = lbsa.create_opinion_lexicon(language='english', source='mpqa')
print(lbsa.make_analysis(tweet, lexicon))
