#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heading
*******
About this module...
Contents
********

"""
import os
import re
import numpy as np
import spacy as sp
import pandas as pd
from itertools import islice, tee
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def read_save(f_path):
    """
    Reads file and closes it afterwards
    :param f_path: a valid path as string
    :return: text inside the file
    """
    with open(f_path, "r") as file:
        text = file.read()

    return text


class TrainingData(object):

    def __init__(self, folder):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.corpus_df = pd.DataFrame()
        self.corpus_folder = folder
        self.X = None
        self.Y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def collect_files_from_dir(self, file_ending):
        """
        Create single-column pandas DataFrame containing Paths from dir with certain ending

        :param file_ending: ending of files to collect in your corpus DataFrame
        :return: DataFrame with column "file_path" to store Paths of corpus files
        """

        files = list()
        for path, sub_dirs, file_names in os.walk(self.corpus_folder):
            for filename in file_names:
                if filename.endswith(file_ending):
                    files.append(os.path.join(path, filename))
        files.sort()
        self.corpus_df = pd.DataFrame(files, columns=['file_path'])

        return self.corpus_df

    def add_categories(self):
        """
        Adds Categories for Classification task by using subdirs the corpus folder as label

        :return: DataFrame containing new column "Categories" for classification based on subdirs
        """
        categories = self.corpus_df.applymap(lambda x: x.split(os.sep)[-2])
        self.corpus_df["Categories"] = categories

        return self.corpus_df

    def add_text(self, lang):
        """
        Reads all paths in DataFrame["file_path"] and stores resulting string in DataFrame

        :return: DataFrame with new column "text" with string
        """
        nlp = sp.load(lang)
        self.corpus_df["text"] = self.corpus_df["file_path"].apply(lambda x: read_save(x))
        self.corpus_df["tokens"] = self.corpus_df["text"].apply(lambda x: nlp(x))
        return self.corpus_df

    def to_sentences(self):
        """
        Splits list of tokens into list of sentences containing tokens

        :return: DataFrame with new column "sentences"
        """
        self.corpus_df["sentences"] = self.corpus_df["tokens"].apply(lambda x: [x for x in x.sents])

        return self.corpus_df

    def to_chunks(self, scope, chunk_size):
        """
        Splits list of items into sublists of size n

        :param scope: name of column where items will be chunked
        :param chunk_size: size of the chunks
        :return:
        """
        self.corpus_df[scope+"_chunks"] = self.corpus_df[scope].apply(
            lambda x: [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)])

        return self.corpus_df

    def to_chars(self):
        """
        Splitting strings to list of characters

        :return: DataFrame with new column "characters" containing list of characters
        """
        self.corpus_df["characters"] = self.corpus_df["text"].apply(lambda x: list(x))

        return self.corpus_df

    def to_ngrams(self, scope, gram_size):
        """
        Generates ngrams from sentences, tokens or chars

        :param scope: column name of corpus_df containing items to transform
        :param gram_size: size of the resulting ngrams
        :return:
        """
        self.corpus_df[scope+"_ngrams"] = self.corpus_df[scope].apply(lambda x: list(
            zip(*(islice(seq, index, None) for index, seq in enumerate(tee(x, gram_size))))))

        return self.corpus_df

    def generate_one_hot_matrix(self, scope, num_words):
        """

        :param scope:
        :param num_words:
        :return:
        """
        samples = np.array(self.corpus_df[scope]).flatten("A")
        samples = [str(item) for sublist in samples for item in sublist]
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(samples)

        self.corpus_df["one_hot"] = self.corpus_df[scope].apply(lambda x: tokenizer.texts_to_matrix(
            [str(item) for sublist in x for item in sublist], mode="binary"))

        return self.corpus_df

    def generate_sequences(self, scope, num_words):
        """
        :param scope:
        :param num_words:
        :return:
        """
        samples = np.array(self.corpus_df[scope]).flatten("A")
        samples = [str(item) for sublist in samples for item in sublist]
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(samples)

        self.corpus_df["sequences"] = self.corpus_df[scope].apply(lambda x: tokenizer.texts_to_sequences(
            [str(x) for x in x]))

        return self.corpus_df

    def padding_sequences(self, maxlen):
        """


        :param maxlen:
        :return:
        """
        self.corpus_df["sequences"] = self.corpus_df["sequences"].apply(lambda x: pad_sequences(x, maxlen))

        return self.corpus_df

    def to_categorical_trainingdata(self, scope):
        """

        :param scope:
        :return:
        """
        categories = np.unique(self.corpus_df["Categories"])
        i = 0
        for category in categories:
            self.corpus_df["Categories"] = self.corpus_df["Categories"].apply(
                lambda x: re.sub("^"+re.escape(category)+"$", str(i), x))
            i += 1

        self.X = list()
        self.Y = list()

        for index, row in self.corpus_df.iterrows():

            self.X.append(row[scope])
            self.Y.append([int(row["Categories"])] * len(row[scope]))

        self.X = [x for sublist in self.X for x in sublist]
        self.Y = [y for sublist in self.Y for y in sublist]
        self.X = np.array(self.X)
        self.Y = np.array(self.Y).flatten()

        return self.X, self.Y

    def split_training_data(self, ratio):
        """


        :param ratio:
        :return:
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=ratio)

        return self.x_train, self.x_test, self.y_train, self.y_test

    def load_sequential_data(self):

        self.corpus_df["text"] = self.corpus_df["file_path"].apply(lambda x: read_save(x))
        self.corpus_df["sequence_label"] = self.corpus_df["text"].apply(lambda x: re.sub(".*\t", "", x).split("\n"))
        self.corpus_df["text"] = self.corpus_df["text"].apply(lambda x: re.sub("\t.*", "", x).split("\n"))

        return self.corpus_df

    def add_sequential_context(self, windowsize):

        self.corpus_df["sequence_training"] = self.corpus_df["text"].apply(lambda x: context_grabber(x, windowsize))


def context_grabber(sequence, windowsize):

    padding = list(np.array([["padder"] * windowsize]).flatten())

    expanded_sequence = sequence+padding
    expanded_sequence = padding+expanded_sequence

    single_sequences = [
        " ".join(expanded_sequence[ind - windowsize:ind]) +
        " " + x + " " +
        " ".join(expanded_sequence[ind+1:ind + windowsize])
        for ind, x in enumerate(expanded_sequence) if not x.startswith("padder")]

    return single_sequences
