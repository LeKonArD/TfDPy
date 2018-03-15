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

    def __init__(self, scope, num_words, file_ending, folder, windowsize, ratio, maxlen):
        self.corpus_df = pd.DataFrame()
        self.scope = scope
        self.num_words = num_words
        self.file_ending = file_ending
        self.folder = folder
        self.windowsize = windowsize
        self.ratio = ratio
        self.maxlen = maxlen
        #self.lang = lang
        #self.chunk_size = chunk_size
        #self.ngrams = ngrams
        self.X = None
        self.Y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def collect_files_from_dir(self):

        files = list()
        for path, sub_dirs, file_names in os.walk(self.folder):
            for filename in file_names:
                if filename.endswith(self.file_ending):
                    files.append(os.path.join(path, filename))
        files.sort()
        self.corpus_df = pd.DataFrame(files, columns=['file_path'])

        return self.corpus_df

    def add_categories(self):

        categories = self.corpus_df.applymap(lambda x: x.split(os.sep)[-2])
        self.corpus_df["Categories"] = categories

        return self.corpus_df

    def add_text(self):

        nlp = sp.load(self.lang)
        self.corpus_df["text"] = self.corpus_df["file_path"].apply(lambda x: read_save(x))
        self.corpus_df["tokens"] = self.corpus_df["text"].apply(lambda x: nlp(x))
        return self.corpus_df

    def to_sentences(self):

        self.corpus_df["sentences"] = self.corpus_df["tokens"].apply(lambda x: [x for x in x.sents])

        return self.corpus_df

    def to_chunks(self):

        self.corpus_df[self.scope+"_chunks"] = self.corpus_df[self.scope].apply(
            lambda x: [x[i:i+self.chunk_size] for i in range(0, len(x), self.chunk_size)])

        return self.corpus_df

    def to_chars(self):

        self.corpus_df["characters"] = self.corpus_df["text"].apply(lambda x: list(x))

        return self.corpus_df

    def to_ngrams(self):

        self.corpus_df[self.scope+"_ngrams"] = self.corpus_df[self.scope].apply(lambda x: list(
            zip(*(islice(seq, index, None) for index, seq in enumerate(tee(x, self.gram_size))))))

        return self.corpus_df

    def generate_one_hot_matrix(self):

        samples = np.array(self.corpus_df[scope]).flatten("A")
        samples = [str(item) for sublist in samples for item in sublist]
        tokenizer = Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(samples)

        self.corpus_df["one_hot"] = self.corpus_df[self.scope].apply(lambda x: tokenizer.texts_to_matrix(
            [str(item) for sublist in x for item in sublist], mode="binary"))

        return self.corpus_df

    def generate_sequences(self):

        samples = np.array(self.corpus_df[self.scope]).flatten("A")
        samples = [str(item) for sublist in samples for item in sublist]
        tokenizer = Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(samples)

        self.corpus_df["sequences"] = self.corpus_df[self.scope].apply(lambda x: tokenizer.texts_to_sequences(
            [str(x) for x in x]))

        return self.corpus_df

    def padding_sequences(self):

        self.corpus_df["sequences"] = self.corpus_df["sequences"].apply(lambda x: pad_sequences(x, self.maxlen))

        return self.corpus_df

    def to_categorical_trainingdata(self):

        categories = np.unique(self.corpus_df["Categories"])
        i = 0
        for category in categories:
            self.corpus_df["Categories"] = self.corpus_df["Categories"].apply(
                lambda x: re.sub("^"+re.escape(category)+"$", str(i), x))
            i += 1

        self.X = list()
        self.Y = list()

        for index, row in self.corpus_df.iterrows():

            self.X.append(row[self.scope])
            self.Y.append([int(row["Categories"])] * len(row[self.scope]))

        self.X = [x for sublist in self.X for x in sublist]
        self.Y = [y for sublist in self.Y for y in sublist]
        self.X = np.array(self.X)
        self.Y = np.array(self.Y).flatten()

        return self.X, self.Y

    def split_training_data(self):

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.ratio)

        return self.x_train, self.x_test, self.y_train, self.y_test

    def load_sequential_data(self):

        self.corpus_df["text"] = self.corpus_df["file_path"].apply(
            lambda x: read_save(x))
        self.corpus_df["sequence_label"] = self.corpus_df["text"].apply(
            lambda x: re.sub(".*\t", "", x).split("\n")[0:-1])
        self.corpus_df["text"] = self.corpus_df["text"].apply(
            lambda x: re.sub("\t.*", "", x).split("\n")[0:-1])

        return self.corpus_df

    def add_sequential_context(self):

        self.corpus_df["sequence_training"] = self.corpus_df["text"].apply(
            lambda x: context_grabber(x, self.windowsize))

        return self.corpus_df

    def to_sequential_trainingdata(self):

        self.X = list(self.corpus_df["sequences"])
        self.X = np.array([np.asarray(x, dtype=int) for sublist in self.X for x in sublist])

        self.Y = list(self.corpus_df["sequence_label"])
        self.Y = [int(x) for sublist in self.Y for x in sublist]
        self.Y = np.array(self.Y)

        return self.X, self.Y


def context_grabber(sequence, windowsize):

    padding = list(np.array([["padder"] * (windowsize+1)]).flatten())

    expanded_sequence = sequence+padding
    expanded_sequence = padding+expanded_sequence

    single_sequences = [
        " ".join(expanded_sequence[ind - windowsize:ind]) +
        " " + x + " " +
        " ".join(expanded_sequence[ind+1:ind + windowsize+1])
        for ind, x in enumerate(expanded_sequence) if not x.startswith("padder")]

    return single_sequences


def TD_paramsearch(pipeline, parameters, classifier):

    for scope_single in parameters["scope"]:
        for num_words_single in parameters["num_words"]:
            for file_ending_single in parameters["file_ending"]:
                for folder_single in parameters["folder"]:
                    for windowsize_single in parameters["windowsize"]:
                        for ratio_single in parameters["ratio"]:
                            for maxlen_single in parameters["maxlen"]:
                                this_trainingdata = TrainingData(scope=scope_single,
                                                                 num_words=num_words_single,
                                                                 file_ending=file_ending_single,
                                                                 folder=folder_single,
                                                                 windowsize=windowsize_single,
                                                                 ratio=ratio_single,
                                                                 maxlen=maxlen_single)

                                for function_call in pipeline:

                                    if function_call.__name__ == "to_sequential_trainingdata":
                                        this_trainingdata.X, this_trainingdata.Y = getattr(this_trainingdata,
                                                                                           function_call.__name__)()
                                        continue

                                    if function_call.__name__ == "split_training_data":
                                        this_trainingdata.x_train, this_trainingdata.x_test, this_trainingdata.y_train, this_trainingdata.y_test = getattr(this_trainingdata,
                                                                                                                                                           function_call.__name__)()
                                        continue

                                    this_trainingdata.corpus_df = getattr(this_trainingdata, function_call.__name__)()



    return this_trainingdata
