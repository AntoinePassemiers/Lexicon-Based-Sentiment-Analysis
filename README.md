[![Travis-CI Build Status](https://travis-ci.org/AntoinePassemiers/Lexicon-Based-Sentiment-Analysis.svg?branch=master)](https://travis-ci.org/AntoinePassemiers/Lexicon-Based-Sentiment-Analysis)
# LBSA - Lexicon-based Sentiment Analysis

## Installation

From the parent folder, install the library by typing the following command:

```sh
$ sudo python setup.py install
```

### Dependencies

* numpy >= 1.13.3
* pandas >= 0.21.0
* xlrd

## Features

### Sentiment analysis

```python
>>> import lbsa
>>> tweet = """
... The Budget Agreement today is so important for our great Military.
... It ends the dangerous sequester and gives Secretary Mattis what he needs to keep America Great.
... Republicans and Democrats must support our troops and support this Bill!
... """
>>> sa_lexicon = lbsa.get_lexicon('sa', language='english', source='nrc')
>>> sa_lexicon.process(tweet)
{'anger': 0, 'anticipation': 0, 'disgust': 0, 'fear': 2, 'joy': 0, 'sadness': 0, 
'surprise': 0, 'trust': 3}
```

### Opinion mining

```python
>>> op_lexicon = lbsa.get_lexicon('opinion', language='english', source='nrc')
>>> op_lexicon.process(tweet)
{'positive': 2, 'negative': 1}
```

### Feature extractor

```python
>>> extractor = lbsa.FeatureExtractor(sa_lexicon, op_lexicon)
>>> extractor.process(tweet)
array([0., 0., 0., 2., 0., 0., 0., 3., 2., 1.])
```

#### Example

Feature extractor:

[feature_extraction.py](https://github.com/AntoinePassemiers/Lexicon-Based-Sentiment-Analysis/blob/master/src/examples/feature_extraction.py)

![alt text](imgs/zarathustra.png)

Perform sentiment analysis over time on "Thus spoke Zarathustra":

[book.py](https://github.com/AntoinePassemiers/Lexicon-Based-Sentiment-Analysis/blob/master/src/examples/book.py)
