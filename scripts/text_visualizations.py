#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

# import required libraries
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize

# some Data Analysis Libraries
import matplotlib.pyplot as plt

# other required libraries
from PIL import Image


# generate simple wordcloud
def generate_simple_wordcloud1(text):
    text_cloud = WordCloud().generate(text)
    # display wordcloud
    plt.imshow(text_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# generate simple wordcloud
def generate_simple_wordcloud2(text, stopwords_keep=False,
                               max_words=1000):
    stopwords_list_en = stopwords.words('english')
    # remove stopwords if True
    if stopwords_keep==True:
        text_wo_stopwords = ' '.join([w for w in word_tokenize(text)
                                      if w.lower()
                                      not in stopwords_list_en])
        text_cloud = WordCloud(background_color='white',
                                max_words=max_words).generate(text_wo_stopwords)
    else:
        text_cloud = WordCloud(background_color='white',
                                max_words=max_words).generate(text)
    # display wordcloud
    plt.imshow(text_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


