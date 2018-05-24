#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lib import utils
from sklearn.svm import SVC
from keras import Sequential
from keras.layers import Dense,Flatten,Embedding


# This is an example for using TfDPy to classify collections of documents

# First step: Build the Preprocessing Pipeline

pipeline = [utils.TrainingData.collect_files_from_dir,
            utils.TrainingData.add_categories,
            utils.TrainingData.add_text,
            utils.TrainingData.to_chunks,
            utils.TrainingData.generate_sequences,
            utils.TrainingData.padding_sequences,
            utils.TrainingData.to_categorical_trainingdata,
            utils.TrainingData.split_training_data
            ]


# Second step: add parameters for the pipeline


parameters = {"folder": ["/media/konle/3d665f71-096f-4974-8ef9-b365f5f16389/classification_test"],
              "file_ending": ["txt"],
              "lang": ["de"],
              "chunk_scope": ["tokens"],
              "chunk_size": [1000],
              "sequence_scope": ["tokens_chunks"],
              "maxlen": [1000],
              "num_words": [100],
              "categorical_scope": ["sequences"],
              "ratio": [0.3]}

# Third step: define classifiers

# A neural network
model = Sequential()
model.add(Embedding(1000, 4, input_length=1000))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['acc'])

# And some linear regressions
classifier = [model, SVC(kernel="linear", C=0.025), SVC(kernel="linear", C=0.1)]

# Run the pipeline
values = utils.td_paramsearch(pipeline, parameters, classifier)
print(values)