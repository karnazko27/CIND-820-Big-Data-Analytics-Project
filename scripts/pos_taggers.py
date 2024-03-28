#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import spacy

def get_tokens(sent):
    # load data and models for the English language as first step
    # language models must be downloaded on server before use
    try:
        nlp = spacy.load('en_core_web_sm')
        # instantiate doc object
        doc = nlp(sent)
        return [token.text for token in doc]
    except ValueError:
        raise ValueError('Must enter string')

def get_lemmas(sent):
    # load data and models for the English language as first step
    # language models must be downloaded on server before use
    try:
        nlp = spacy.load('en_core_web_sm')
        # instantiate doc object
        doc = nlp(sent)
        return [token.lemma_ for token in doc]
    except ValueError:
        raise ValueError('Must enter string')

def get_coarse_tags(sent):
    # load data and models for the English language as first step
    # language models must be downloaded on server before use
    try:
        nlp = spacy.load('en_core_web_sm')
        # instantiate doc object
        doc = nlp(sent)
        return [token.pos_ for token in doc]
    except ValueError:
        raise ValueError('Must enter string')

def get_fine_tags(sent):
    # load data and models for the English language as first step
    # language models must be downloaded on server before use
    try:
        nlp = spacy.load('en_core_web_sm')
        # instantiate doc object
        doc = nlp(sent)
        return [token.tag_ for token in doc]
    except ValueError:
        raise ValueError('Must enter string')

def find_specific_tag(sent, pos_tag):
    # load data and models for the English language as first step
    # language models must be downloaded on server before use
    '''
        pos_tag: must be fine-grained tag such as NNP, VB, JJ, etc.
    '''
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sent)
        return [word for word in doc
                if word.tag_.startswith(pos_tag)]
    except ValueError:
        raise ValueError('Must enter string')

def find_specific_tag_coarse(sent, pos_tag):
    # load data and models for the English language as first step
    # language models must be downloaded on server before use
    '''
        pos_tag: must be coarse-grained tag such as NOUN, VERB, PROPN, etc.
    '''
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sent)
        return [word for word in doc
                if word.pos_.startswith(pos_tag)]
    except ValueError:
        raise ValueError('Must enter string')

