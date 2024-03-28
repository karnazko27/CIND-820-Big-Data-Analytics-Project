#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Includes Named Entity Recognition Functions and Custom Text Filters

"""Named Entity Recognition and Custom Text Filters

This script includes functions that will allow text processing to be more informative
by customizing according to needs of user.

"""
import spacy
from spacy.lang.en.stop_words import STOP_WORDS



def check_ticker_alnum(ticker):
    if ticker.isalnum():
        return True
    else:
        return False

def filter_newline(sent):
    filtered_text = text_string.replace('\n', '')
    return filtered_text

def filter_stopwords(sent):
    nlp = spacy.load('en_core_web_sm')
    # instantiate doc object
    doc = nlp(sent)
    tokens = [token.text for token in doc]
    tokens_set = set(tokens)
    final_tokens_without_stopwords = set(tokens) - STOP_WORDS
    return final_tokens_without_stopwords

def add_custom_stopwords(list_of_custom_stopwords):
    # instantiate language model
    nlp = spacy.load('en_core_web_sm')

    for stopword in list_of_custom_stopwords:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

def filter_stopwords2(sent):
    '''Alternative to filter stopwords function from above. Slower alternative'''
    nlp = spacy.load('en_core_web_sm')
    # instantiate doc object
    doc = nlp(sent)

    new_sent = []

    for w in doc:
        if not w.is_stop:
            new_sent.append(w.text)
    return new_sent

def get_proper_nouns(sent):
    # load data and models for the English language as first step
    # language models must be downloaded on server before use
    try:
        nlp = spacy.load('en_core_web_sm')
        # instantiate doc object
        doc = nlp(sent)
        return [(token, token.pos_) for token in doc if token.pos_=="PROPN"]
    except ValueError:
        raise ValueError('Must enter string')

def find_word(text, substring):
    if text.find(substring) == -1:
        return False
    else:
        return True

def get_simple_named_entity(sent):
    # load data and models for the English language as first step
    # language models must be downloaded on server before use
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sent)
        return [token.text for token in doc
                    if token.ent_type != 0]
    except ValueError:
        raise ValueError('Must enter string')

