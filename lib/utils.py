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
import spacy as sp
import pandas as pd


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

    return corpus_df


def to_tokens(x):

    return x


def to_chunks(x, chunk_size):

    chunk_size += 1

    return x


def to_chars(corpus_df):

    corpus_df["text"] = corpus_df["text"].apply(lambda x: list(x))

    return corpus_df


def to_ngrams(corpus_df, gram_size):

    gram_size += 1

    return corpus_df


def initial_spacy_step(corpus_df, lang):

    nlp = sp.load(lang)
    corpus_df["NLP"] = corpus_df["text"].apply(lambda x: nlp(x))

    return corpus_df


# Tests

corpus = collect_files_from_dir("./../testing/class_test", "txt")
corpus = add_categories(corpus)
corpus = add_text(corpus)
corpus = initial_spacy_step(corpus, "de")
print(corpus["NLP"].iloc[1])
