#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#


# import libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob

import eng_spacysentiment


# run this once
nltk.download('vader_lexicon')

def get_sentiment(text):
    '''
    Returns sentiment score of input text in dict(object) form with
    scores ranging in value from -1.0 to 1.0.
    '''
    # initialize sentiment detector
    sd = SentimentIntensityAnalyzer()
    # create column to capture sentiments from comments
    scores_dict = sd.polarity_scores(text)
    sentiment_score = 1*scores_dict['pos'] + (-1)*scores_dict['neg']
    return sentiment_score

def get_positive_sentiment(text):
    '''
    Returns sentiment score of input text in dict(object) form with
    scores ranging in value from -1.0 to 1.0.
    '''
    # initialize sentiment detector
    sd = SentimentIntensityAnalyzer()
    # create column to capture sentiments from comments
    scores_dict = sd.polarity_scores(text)
    sentiment_score = 1*scores_dict['pos']
    return sentiment_score

def get_negative_sentiment(text):
    '''
    Returns sentiment score of input text in dict(object) form with
    scores ranging in value from -1.0 to 1.0.
    '''
    # initialize sentiment detector
    sd = SentimentIntensityAnalyzer()
    # create column to capture sentiments from comments
    scores_dict = sd.polarity_scores(text)
    sentiment_score = (-1)*scores_dict['neg']
    return sentiment_score


def sentiment_polarity(text):
    """
    Returns Sentiment Polarity score using SpacyTextBlob
    """
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("spacytextblob")
    doc = nlp(text)
    return doc._.blob.polarity

def sentiment_polarity2(text):
    """
    Returns Sentiment Polarity with TextBlob
    """
    blob = TextBlob(text)
    return blob.sentiment_assessments.polarity

def subjectivity_score(text):
    blob = TextBlob(text)
    return blob.sentiment_assessments.subjectivity

def simple_sentiment_score(text):
    """
    Uses eng_spacysentiment to get sentiment score
    Caution: this needs spacy<3.3.0. It may not work with upgraded spacy library
    """
    nlp = eng_spacysentiment
    doc = nlp(text)
    scores_dict = doc.cats
    sentiment_score = 1*scores_dict['positive'] + (-1)*scores_dict['negative']
    return sentiment_score
