#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heading
*******
About this module...
Contents
********

"""
#################### CORE Filehandling #####################
import nltk
import os
import pandas as pd
import numpy as np

def collect_files_from_dir(corpus_folder, file_ending):

    # create single-column pandas dataframe containing filenames from dir with certain ending

    files = list()
    for path, subdirs, filenames in os.walk(corpus_folder):
        for filename in filenames:
            if filename.endswith(file_ending):
                files.append(os.path.join(path, filename))
    files.sort()
    corpus_df = pd.DataFrame(files, columns=['filepath'])

    return corpus_df

def add_categories(corpus_df):

    # map directories as descriptors for categories
    categories = corpus_df.applymap(lambda x: x.split(os.sep)[-2])
    corpus_df["Categories"] = categories

    return corpus_df

def add_text(corpus_df):

    corpus_df["text"] = corpus_df["filepath"].apply(lambda x:read_save(x))

    return corpus_df

def read_save(f_path):


    with open(f_path,"r") as file:
        text = file.read()

    return text


################ Preprocessing text #################
def to_sentences(corpus_df):

    sent_detector = nltk.data.load('tokenizers/punkt/german.pickle')
    corpus_df["text"] = corpus_df["text"].apply(lambda x: sent_detector.tokenize(x))

    return corpus_df

def to_tokens(x):


    return x

def to_chunks(x,chunk_size):


    return x

def to_chars(corpus_df):

    corpus_df["text"]=corpus_df["text"].apply(lambda x: list(x))

    return corpus_df

def to_ngrams(corpus_df,gram_size):

    corpus_df["text"] = corpus_df["text"].apply(lambda x: nltk.ngrams(x,gram_size))

    return corpus_df









################ Tests
corpus = collect_files_from_dir("/home/leo/Documents/schemaliteratur_DS/test_HA","xml")
corpus = add_categories(corpus)
corpus = add_text(corpus)
corpus = to_tokens(corpus)
corpus = to_ngrams(corpus,3)
gram = corpus["text"][1]
for g in gram:
    print(g)