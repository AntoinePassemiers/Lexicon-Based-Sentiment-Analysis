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
lexicon = lbsa.get_lexicon('opinion', language='english', source='afinn')
print(lexicon.process(tweet))

print('\nUse nrc opinion lexicon')
lexicon = lbsa.get_lexicon('opinion', language='english', source='nrc')
print(lexicon.process(tweet))

print('\nUse nrc sentiment analysis lexicon')
lexicon = lbsa.get_lexicon('sa', language='english', source='nrc')
print(lexicon.process(tweet))

print('\nUse mpqa lexicon')
lexicon = lbsa.get_lexicon('opinion', language='english', source='mpqa')
print(lexicon.process(tweet))
