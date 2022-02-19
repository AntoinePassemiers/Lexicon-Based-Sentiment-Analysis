# -*- coding: utf-8 -*-
# tweets.py
# author : Antoine Passemiers

import lbsa


tweet = """
The Budget Agreement today is so important for our great Military.
It ends the danger sequester and gives Secretary Mattis what he needs to keep America Great.
Republicans and Democrats must support our troops and support this Bill!
"""

print('\nUse NRC lexicon')
lexicon = lbsa.get_lexicon('opinion', language='english', source='nrc')
print(lexicon.process(tweet))

print('\nUse afinn lexicon')
lexicon = lbsa.get_lexicon('opinion', language='english', source='afinn')
print(lexicon.process(tweet))

print('\nUse mpqa lexicon')
lexicon = lbsa.get_lexicon('opinion', language='english', source='mpqa')
print(lexicon.process(tweet))

tweet2 = """
A la suite de la tempête #Eunice et à la demande du Président de la République,
l’Etat décrétera dans les meilleurs délais l’état de catastrophe naturelle partout
où cela s’avérera nécessaire.
"""
print('\nAuto-detect languages')
lexicon = lbsa.get_lexicon('sa', language='auto', source='nrc')
print(lexicon.process(tweet))
print(lexicon.process(tweet2))
