#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib import utils
from sklearn.neighbors import KNeighborsClassifier
from keras import Sequential
from keras.layers import Dense,Flatten,Embedding


myData = utils.TrainingData()


# get filepaths

myData.folder = "/home/konle/Documents/data_jose/casa-citas-private/data/"
myData.file_ending = "csv"
myData.corpus_df = myData.collect_files_from_dir()

# read tsv
myData.corpus_df = myData.read_single_tsv()

# nlp text
myData.lang = "es"
myData.nlp_scope = "text"
myData.corpus_df = myData.nlp_text()


# generate sequences

myData.sequence_scope = "sequence_training"
myData.num_words = 1000
myData.corpus_df = myData.generate_sequences()

# padding

myData.maxlen = 3
myData.corpus_df = myData.padding_sequences()

# to multilabel training data

myData.labels = ["1","2"]
myData.X, myData.Y = myData.to_multilabel_trainingdata()

myData.ratio = 0.1
myData.x_train, myData.x_test, myData.y_train, myData.y_test = myData.split_training_data()

clf = KNeighborsClassifier()
clf.fit(myData.x_train, myData.y_train)
print(clf.score(myData.x_test, myData.y_test))

model = Sequential()
model.add(Embedding(300, myData.maxlen, input_length=3))
model.add(Flatten())
model.add(Dense(len(myData.labels), activation='sigmoid'))
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])

model.fit(myData.x_train, myData.y_train)
model.predict(myData.x_test)
