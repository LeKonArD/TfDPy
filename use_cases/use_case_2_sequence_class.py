#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib import utils
from sklearn.svm import SVC
from keras import Sequential
from keras.layers import Dense,Flatten,Embedding


# This is an example for using TfDPy to classify collections of documents

# First step: Build the Preprocessing Pipeline

pipeline = [utils.TrainingData.collect_files_from_dir,
            utils.TrainingData.load_sequential_data,
            utils.TrainingData.add_sequential_context,
            utils.TrainingData.nlp_text,
            utils.TrainingData.generate_sequences,
            utils.TrainingData.to_sequential_trainingdata,
            utils.TrainingData.split_training_data
            ]


# Second step: add parameters for the pipeline


parameters = {"folder": ["/media/konle/3d665f71-096f-4974-8ef9-b365f5f16389/software/TfDPy/testing/sequence_test"],
              "file_ending": ["tsv"],
              "lang": ["de"],
              "windowsize": [3],
              "num_words": [300],
              "sequence_scope": ["sequence_training"],
              "ratio": [0.3]}

# Third step: define classifiers

# A neural network
model = Sequential()
model.add(Embedding(1000, 4, input_length=7))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])

# And some linear regressions
classifier = [SVC(kernel="linear", C=0.025), SVC(kernel="linear", C=0.1),model]

# Run the pipeline
values = utils.td_paramsearch(pipeline, parameters, classifier)
print(values)


