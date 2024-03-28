#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

"""
Steps for Event extraction using ngrams

Step 1: convert list of string of comments or threads to ngrams using get_all_ngrams()
Step 2: create mapping of Ngram to Frequency
"""

from nltk import ngrams
from gensim import corpora
from gensim import models

import spacy
from spacy.lang.en.stop_words import STOP_WORDS


def get_all_ngrams(list_of_strings, n=3):
    """
    Param: list_of_strings: comments or thread titles
    """
    all_ngrams = [list(ngrams(c.split(), n=n)) for c in list_of_strings]
    joined_ngrams_all = []

    for each_sent in all_ngrams:
        joined_ngrams_sent = []
        for x in each_sent:
            joined_ngram = ' '.join(x)
            joined_ngrams_sent.append(joined_ngram)
        joined_ngrams_all.append(joined_ngrams_sent)

    return joined_ngrams_all

def bag_of_words_ngram_to_frequency_mapping(joined_ngrams_all):
    gensim_dict_object = corpora.Dictionary(joined_ngrams_all)
    # mapping word to index
    words2index = {w: index for (w, index) in gensim_dict_object.token2id.items()}

    # mapping index to frequency
    corpus = [gensim_dict_object.doc2bow(text) for text in joined_ngrams_all]

    ary_index_to_freq = [{index: frequency for (index, frequency) in corp} for corp in corpus]
    flattened_dict_of_index_to_freq = {}
    for item in ary_index_to_freq:
        # take key, value in its own dict and append it to flattened_ary
        if item:
            for (i, v) in item.items():
                flattened_dict_of_index_to_freq[i] = v

    trigram_to_frequency = {}
    for trigram, trigram_index in words2index.items():
        trigram_to_frequency[trigram] = flattened_dict_of_index_to_freq[trigram_index]

    return trigram_to_frequency

def tfidf_ngram_to_frequency_mapping(joined_ngrams_all):
    gensim_dict_object = corpora.Dictionary(joined_ngrams_all)
    # mapping word to index
    words2index = {w: index for (w, index) in gensim_dict_object.token2id.items()}

    # mapping index to frequency
    corpus = [gensim_dict_object.doc2bow(text) for text in joined_ngrams_all]

    tfidf_model = models.TfidfModel(corpus)
    tfidf_docs = [doc for doc in tfidf_model[corpus]]

    ary_index_to_freq = [{index: frequency for (index, frequency) in corp} for corp in tfidf_docs]
    flattened_dict_of_index_to_freq = {}
    for item in ary_index_to_freq:
        # take key, value in its own dict and append it to flattened_ary
        if item:
            for (i, v) in item.items():
                flattened_dict_of_index_to_freq[i] = v

    trigram_to_frequency = {}
    for trigram, trigram_index in words2index.items():
        trigram_to_frequency[trigram] = flattened_dict_of_index_to_freq[trigram_index]

    return trigram_to_frequency

def topics_from_LDA_model(joined_ngrams_all, n_topics=10):
    """
    To get document-topic proportions: ldamodel[document]
    - this document can be an unseen document as long as the words are in the vocab of the model

    """
    gensim_dict_object = corpora.Dictionary(joined_ngrams_all)
    # mapping word to index
    words2index = {w: index for (w, index) in gensim_dict_object.token2id.items()}

    # mapping index to frequency
    corpus = [gensim_dict_object.doc2bow(text) for text in joined_ngrams_all]
    ldamodel = models.LdaModel(corpus=corpus,
                               num_topics=n_topics,
                               id2word=gensim_dict_object)

    return ldamodel.show_topics()

def topics_from_LSI_model(joined_ngrams_all, n_topics=10):
    gensim_dict_object = corpora.Dictionary(joined_ngrams_all)
    # mapping word to index
    words2index = {w: index for (w, index) in gensim_dict_object.token2id.items()}

    # mapping index to frequency
    corpus = [gensim_dict_object.doc2bow(text) for text in joined_ngrams_all]
    lsimodel = models.LdaModel(corpus=corpus,
                               num_topics=n_topics,
                               id2word=gensim_dict_object)

    return lsimodel.show_topics()

def topics_from_HDP_model(joined_ngrams_all):
    gensim_dict_object = corpora.Dictionary(joined_ngrams_all)
    # mapping index to frequency
    corpus = [gensim_dict_object.doc2bow(text) for text in joined_ngrams_all]

    # HDP model
    hdpmodel = models.HdpModel(corpus=corpus, id2word=gensim_dict_object)
    return hdpmodel.show_topics()

def remove_stop_words():
    # list of custom stopwords to remove
    my_stopwords = ['lol', 'shit', 'fuck', 'dick']

    nlp = spacy.load("en_core_web_sm")

    # add custom stopwords to spacy stopwords list
    for stopword in my_stopwords:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

def make_topics_bow(topics):
    """
    Pass results of functions: topics_from_HDP_model, topics_from_LSI_model, topics_from_LDA_model
    as arguments to get topics and probability of that topic appearing in document


    """
    topic = topics.split('+')
    # list to store topic bag-of-words
    topic_bow = []
    for word in topic:
        # split probability and word
        prob, word = word.split("*")
        word = word.strip()
        word = word.replace("\\", "")
        word = word.replace('"', '')
        topic_bow.append((word, float(prob)))
    return topic_bow