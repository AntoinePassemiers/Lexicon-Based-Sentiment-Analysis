[![Travis-CI Build Status](https://travis-ci.org/AntoinePassemiers/Lexicon-Based-Sentiment-Analysis.svg?branch=master)](https://travis-ci.org/AntoinePassemiers/Lexicon-Based-Sentiment-Analysis)
# LBSA - Lexicon-based Sentiment Analysis

http://www.saifmohammad.com/WebPages/lexicons.html

## Installation

From the parent folder, install the library by typing the following command:

```sh
$ sudo python setup.py install
```

### Dependencies

* numpy >= 1.13.3
* pandas >= 0.21.0
* matplotlib >= 2.0.2 (optional)
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
>>> lexicon = lbsa.create_sa_lexicon(language='english', source='nrc')
>>> lbsa.make_analysis(tweet, lexicon)
{'anger': 0, 'anticipation': 0, 'disgust': 0, 'fear': 2, 'joy': 0, 'sadness': 0, 
'surprise': 0, 'trust': 3}
```

### Opinion mining

```python
>>> lexicon = lbsa.create_opinion_lexicon(language='english', source='nrc')
>>> lbsa.make_analysis(tweet, lexicon)
{'positive': 2, 'negative': 1}
```

### Sentiment analysis over time

This should be used with large texts, such as blogs, books, articles, etc.

#### Example

![alt text](imgs/zarathustra.png)

Check the full example:

[Full example](https://github.com/AntoinePassemiers/Lexicon-Based-Sentiment-Analysis/blob/master/src/examples/example.py)