#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heading
*******
About this module...
Contents
********

"""
# CORE File handling
import os
import numpy as np
import spacy as sp
import pandas as pd
from itertools import islice, tee

from keras.preprocessing.text import Tokenizer

def collect_files_from_dir(corpus_folder, file_ending):
    """
    Create single-column pandas DataFrame containing Paths from dir with certain ending
    :param corpus_folder: path to folder containing files or dirs with files
    :param file_ending: ending of files to collect in your corpus DataFrame
    :return: DataFrame with column "file_path" to store Paths of corpus files
    """

    files = list()
    for path, sub_dirs, file_names in os.walk(corpus_folder):
        for filename in file_names:
            if filename.endswith(file_ending):
                files.append(os.path.join(path, filename))
    files.sort()
    corpus_df = pd.DataFrame(files, columns=['file_path'])

    return corpus_df


def add_categories(corpus_df):
    """
    Adds Categories for Classification task by using subdirs the corpus folder as label
    :param corpus_df: DataFrame containing at least a column for "file_path"
    :return: DataFrame containing new column "Categories" for classification based on subdirs
    """
    categories = corpus_df.applymap(lambda x: x.split(os.sep)[-2])
    corpus_df["Categories"] = categories

    return corpus_df


def add_text(corpus_df):
    """
    Reads all paths in DataFrame["file_path"] and stores resulting string in DataFrame
    :param corpus_df: DataFrame containing at least a column for "file_path"
    :return: DataFrame with new column "text" with string
    """
    corpus_df["text"] = corpus_df["file_path"].apply(lambda x: read_save(x))

    return corpus_df


def read_save(f_path):
    """
    Reads file and closes it afterwards
    :param f_path: a valid path as string
    :return: text inside the file
    """
    with open(f_path, "r") as file:
        text = file.read()

    return text


# preprocessing text
def to_sentences(corpus_df):
    """
    Splits list of tokens into list of sentences containing tokens
    :param corpus_df: DataFrame containing at least column "NLP"
    :return: DataFrame with new column "sentences"
    """
    corpus_df["sentences"] = corpus_df["NLP"].apply(lambda x: [x for x in x.sents])

    return corpus_df


def to_chunks(corpus_df, scope, chunk_size):
    """
    Splits list of items into sublists of size n
    :param corpus_df: DataFrame at least containing column with items to chunk
    :param scope: name of column where items will be chunked
    :param chunk_size: size of the chunks
    :return:
    """
    corpus_df[scope+"_chunks"] = corpus_df[scope].apply(
        lambda x: [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)])

    return corpus_df


def to_chars(corpus_df):
    """
    Splitting strings to list of characters
    :param corpus_df: DataFrame at least containing column "text" with strings inside
    :return: DataFrame with new column "characters" containing list of characters
    """
    corpus_df["characters"] = corpus_df["text"].apply(lambda x: list(x))

    return corpus_df


def to_ngrams(corpus_df, scope, gram_size):
    """
    Generates ngrams from sentences, tokens or chars
    :param corpus_df: corpus_df: DataFrame at least containing column with something to transform to ngrams
    :param scope: column name of corpus_df containing items to transform
    :param gram_size: size of the resulting ngrams
    :return:
    """
    corpus_df[scope+"_ngrams"] = corpus_df[scope].apply(lambda x: list(
        zip(*(islice(seq, index, None) for index, seq in enumerate(tee(x, gram_size))))))

    return corpus_df


def initial_spacy_step(corpus_df, lang):

    nlp = sp.load(lang)
    corpus_df["NLP"] = corpus_df["text"].apply(lambda x: nlp(x))

    return corpus_df


def generate_one_hot_matrix(corpus_df,scope,num_words):

    samples = np.array(corpus_df[scope]).flatten("A")
    samples = [str(item) for sublist in samples for item in sublist]
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(samples)

    corpus_df["one_hot"] = corpus_df[scope].apply(lambda x: tokenizer.texts_to_matrix(
        [str(item) for sublist in x for item in sublist], mode="binary"))

    return corpus_df

def generate_sequences(corpus_df,scope,num_words):

    samples = np.array(corpus_df[scope]).flatten("A")
    samples = [str(item) for sublist in samples for item in sublist]
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(samples)

    corpus_df["sequneces"] = corpus_df[scope].apply(lambda x: tokenizer.texts_to_sequences(
        [str(x) for x in x]))

    return corpus_df
# Tests

corpus = collect_files_from_dir("./../testing/class_test", "txt")
corpus = add_categories(corpus)
corpus = add_text(corpus)
corpus = initial_spacy_step(corpus, "de")
#corpus = to_ngrams(corpus, "NLP", 3)
corpus = to_sentences(corpus)
#corpus = to_chunks(corpus, "NLP", 4)




corpus = generate_sequences(corpus, "sentences", 50)

print(len(corpus["sentences"][0]))
print(len([x for x in corpus["sequneces"][0]]))

