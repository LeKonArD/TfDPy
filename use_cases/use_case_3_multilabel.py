#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib import utils as TfDPy
from sklearn.neighbors import KNeighborsClassifier
from keras import Sequential
from keras.layers import Dense,Flatten,Embedding


pipeline = [TfDPy.TrainingData.collect_files_from_dir,
            TfDPy.TrainingData.read_single_tsv,
            TfDPy.TrainingData.nlp_text,
            TfDPy.TrainingData.generate_sequences,
            TfDPy.TrainingData.padding_sequences,
            TfDPy.TrainingData.to_multilabel_trainingdata,
            TfDPy.TrainingData.split_training_data]

parameters = {"folder": ["/home/konle/Documents/data_jose/casa-citas-private/data/"],
             "file_ending": ["csv"],
             "lang": ["es"],
             "nlp_scope": ["text"],
             "sequence_scope": ["sequence_training"],
             "num_words": [100],
             "maxlen": [3],
             "labels": [[1], [2], [3]],
             "ratio": [0.1],
             "categorical_scope": ["sequences"]}

clf = KNeighborsClassifier()

model = Sequential()
model.add(Embedding(300, 3, input_length=3))
model.add(Flatten())
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])

classifier = [clf, model]

values = TfDPy.td_paramsearch(pipeline, parameters, classifier)

model2 = Sequential()
model2.add(Embedding(300, 3, input_length=3))
model2.add(Flatten())
model2.add(Dense(units=3, activation='sigmoid'))
model2.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])

parameters["labels"] = [[1, 2, 3]]
classifier = [clf, model2]

values2 = TfDPy.td_paramsearch(pipeline, parameters, classifier)
print(values)
print(values2)
