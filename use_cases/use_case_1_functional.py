#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib import utils
from sklearn.svm import SVC

# create an instance of Trainingclass

myData = utils.TrainingData()

# load Trainingdata

myData.folder = "/media/konle/3d665f71-096f-4974-8ef9-b365f5f16389/classification_test"
myData.file_ending = "txt"
myData.corpus_df = utils.TrainingData.collect_files_from_dir(myData)

# add categories based on filepaths

myData.corpus_df = utils.TrainingData.add_categories(myData)

# read the files

myData.lang = "de"
myData.corpus_df = utils.TrainingData.add_text(myData)

# chunking

myData.chunk_scope = "tokens"
myData.chunk_size = 1000
myData.corpus_df = utils.TrainingData.to_chunks(myData)

# Indexing
myData.num_words = 1000
myData.sequence_scope = "tokens"
myData.corpus_df = utils.TrainingData.generate_sequences(myData)

# Padding
myData.maxlen = 4000
myData.corpus_df = utils.TrainingData.padding_sequences(myData)

# To Training Data
myData.categorical_scope = "sequences"
myData.X, myData.Y = utils.TrainingData.to_categorical_trainingdata(myData)

# Train Test Split
myData.ratio = 0.5
myData.x_train, myData.x_test, myData.y_train, myData.y_test = utils.TrainingData.split_training_data(myData)

clf = SVC()
clf.fit(myData.x_train, myData.y_train)
print(clf.score(myData.x_test, myData.y_test))
