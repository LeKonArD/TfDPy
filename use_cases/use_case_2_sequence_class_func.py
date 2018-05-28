#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib import utils
from sklearn.svm import SVC

# create an instance of Trainingclass

myData = utils.TrainingData()

# get filepaths

myData.folder = "/media/konle/3d665f71-096f-4974-8ef9-b365f5f16389/software/TfDPy/testing/sequence_test"
myData.file_ending = "tsv"
myData.corpus_df = myData.collect_files_from_dir()


# load training data

myData.corpus_df = myData.load_sequential_data()


# add some context
myData.windowsize = 3
myData.corpus_df = myData.add_sequential_context()
print(myData.corpus_df)

# tokenize text

myData.lang = "de"
myData.nlp_scope = "sequence_training"
myData.corpus_df = myData.nlp_text()

# indexing

myData.num_words = 300
myData.sequence_scope = "sequence_training"
myData.corpus_df = myData.generate_sequences()


# create Trainingdata

myData.X, myData.Y = myData.to_sequential_trainingdata()


# split in train and testdata

myData.ratio = 0.5
myData.x_train, myData.x_test, myData.y_train, myData.y_test = myData.split_training_data()

clf = SVC()

clf.fit(myData.x_train, myData.y_train)
print(clf.score(myData.x_test, myData.y_test))