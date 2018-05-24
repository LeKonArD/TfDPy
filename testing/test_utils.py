#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unittest.mock import patch, mock_open
from lib import utils
import unittest
import pandas as pd
import collections


class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_read_save(self):

        with patch("builtins.open", mock_open(read_data="data")) as mock_file:

            file_input = utils.read_save(mock_file)
            self.assertTrue(file_input == "data")

    def test_fill_parameters(self):

        out_parameters = utils.fill_parameters(parameter_raw=dict())
        self.assertTrue(set(out_parameters.keys()) == {"scope", "num_words", "file_ending", "folder",
                      "windowsize", "ratio", "maxlen", "gram_size", "chunk_size", "lang",
                      "categorical_scope", "chunk_scope", "sequence_scope"})

    def test_context_grabber(self):

        context = utils.context_grabber(["a", "b", "c"], 3)
        self.assertTrue(context == ['padder padder padder a b c padder',
                                    'padder padder a b c padder padder',
                                    'padder a b c padder padder padder'])

    def test_TrainingData_init(self):

        test_object = utils.TrainingData(num_words=100, file_ending=".test", folder="foo/bar", windowsize=0,
                 ratio=0.1, maxlen=0, chunk_size=0, gram_size=0, lang="de", chunk_scope="scope", sequence_scope="scope",
                                         categorical_scope="scope")

        self.assertTrue(test_object.categorical_scope == "scope" and
                        test_object.chunk_scope == "scope" and
                        test_object.sequence_scope == "scope" and
                        test_object.num_words == 100 and
                        test_object.file_ending == ".test" and
                        test_object.folder == "foo/bar" and
                        test_object.windowsize == 0 and
                        test_object.ratio == 0.1 and
                        test_object.maxlen == 0 and
                        test_object.chunk_size == 0 and
                        test_object.gram_size == 0 and
                        test_object.lang == "de" and
                        test_object.X == None and
                        test_object.Y == None and
                        test_object.x_train == None and
                        test_object.x_test == None and
                        test_object.y_train == None and
                        test_object.y_test == None and
                        isinstance(test_object.corpus_df, type(pd.DataFrame()))
                        )

    @unittest.mock.patch('os.walk')
    def test_collect_files_from_dir(self, mock_walk):

        mock_walk.return_value = [("foo", "bar", ["Training.test"]), ("foo", "bar", ["Training.txt"])]
        mock = unittest.mock.Mock()
        mock.folder = ""
        mock.file_ending = ".test"
        mock.DataFrame = pd.DataFrame()
        output = utils.TrainingData.collect_files_from_dir(mock)
        self.assertTrue(list(output["file_path"]) == ["foo/Training.test"])

    def test_add_categories(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.corpus_df = pd.DataFrame({"file_path": ["foo/a/Training.test",
                                                         "foo/b/Training.test",
                                                         "foo/b/Training.test",
                                                         "foo/c/Training.test"]})

        output = utils.TrainingData.add_categories(mock_obj)
        self.assertTrue(list(output["Categories"] == ["a", "b", "b", "c"]))

    @unittest.mock.patch("lib.utils.read_save")
    def test_add_text(self, mock_read_save):

        mock_read_save.return_value = "Das ist ein Test"
        mock_read_save.nlp = ("Das", "ist", "ein", "Test")
        mock_obj = unittest.mock.Mock()
        mock_obj.lang = "de"
        mock_obj.corpus_df = pd.DataFrame({"file_path": ["foo/a/Training.test",
                                                         "foo/b/Training.test"]})
        output = utils.TrainingData.add_text(mock_obj)
        self.assertTrue(list(output["text"]) == ["Das ist ein Test", "Das ist ein Test"])

    def test_to_sentences(self):

        mock_obj = unittest.mock.Mock()
        a = collections.namedtuple("doc","sents")
        b = a(sents=[["Das", "ist", "ein", "Test", "."], ["Das", "auch", "."]])


        mock_obj.corpus_df = pd.DataFrame({"tokens": [b]})

        output = utils.TrainingData.to_sentences(mock_obj)
        self.assertTrue(list(output["sentences"]) == [[["Das", "ist", "ein", "Test", "."], ["Das", "auch", "."]]])

    def test_to_chunks(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.chunk_scope = "test"
        mock_obj.chunk_size = 2
        mock_obj.corpus_df = pd.DataFrame({"test": [[None] * 6]})

        output = utils.TrainingData.to_chunks(mock_obj)
        self.assertTrue(list(output["test_chunks"]) == [[[None, None], [None, None], [None, None]]])

    def test_to_chars(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.corpus_df = pd.DataFrame({"text": ["test", "tset"]})
        output = utils.TrainingData.to_chars(mock_obj)
        self.assertTrue(list(output["characters"]) == [["t", "e", "s", "t"], ["t", "s", "e", "t"]])

    def test_to_ngrams(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.scope = "test"
        mock_obj.gram_size = 3
        mock_obj.corpus_df = pd.DataFrame({"test": [["t", "e", "s", "t"], ["t", "s", "e", "t"]]})
        output = utils.TrainingData.to_ngrams(mock_obj)
        self.assertTrue(list(output["test_ngrams"]) == [[("t", "e", "s"), ("e", "s", "t")],
                                                        [("t", "s", "e"), ("s", "e", "t")]])

    def test_generate_one_hot_matrix(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.scope = "test"
        mock_obj.num_words = 3
        mock_obj.corpus_df = pd.DataFrame({"test": ["Ein Test"]})

        output = utils.TrainingData.generate_one_hot_matrix(mock_obj)
        self.assertTrue(list(output["one_hot"][0].flatten()) == [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                                 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
                                                                 0., 0., 0., 1.])

    def test_generate_sequences(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.sequence_scope = "test"
        mock_obj.num_words = 5

        mock_obj2 = unittest.mock.Mock()
        mock_obj2.text = "Ein Text"

        mock_obj.corpus_df = pd.DataFrame({"test": [[mock_obj2]]})

        output = utils.TrainingData.generate_sequences(mock_obj)
        self.assertTrue(list(output["sequences"]) == [[[1, 2]]])

    def test_padding_sequences(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.maxlen = 8
        mock_obj.corpus_df = pd.DataFrame({"sequences": [[[1, 3, 4, 4, 1, 2],
                                                         [1, 3, 4, 2, 1, 2]]]})
        output = utils.TrainingData.padding_sequences(mock_obj)
        self.assertTrue(list(output["sequences"][0].flatten()) == [1, 3, 4, 4, 1, 2, 0, 0, 1, 3, 4, 2, 1, 2, 0, 0])

    def test_to_categorical_trainingdata(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.categorical_scope = "test"
        mock_obj.corpus_df = pd.DataFrame({"test": [[1, 1, 4], [2, 4, 3], [3, 4, 5], [4, 6, 7]],
                                           "Categories": ["a", "b", "a", "b"]})

        x, y = utils.TrainingData.to_categorical_trainingdata(mock_obj)

        self.assertTrue(list(x) == [1, 1, 4, 2, 4, 3, 3, 4, 5, 4, 6, 7] and
                        list(y) == [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])

    def test_split_training_data(self):

        mock_obj = unittest.mock.Mock()
        mock_obj.X = [[0], [0], [1], [1]]
        mock_obj.Y = [[1], [1], [2], [2]]
        mock_obj.ratio = 0.25
        x_train, x_test, y_train, y_test = utils.TrainingData.split_training_data(mock_obj)

        self.assertTrue(len(x_train) == 3 and
                        len(x_test) == 1 and
                        len(y_train) == 3 and
                        len(y_test) == 1)

    @unittest.mock.patch("lib.utils.read_save")
    def test_load_sequential_data(self, mock_read_save):

        mock_read_save.return_value = "Das\t0\nist\t0\nein\t1\nTest\t1\n"

        mock_obj = unittest.mock.Mock()
        mock_obj.corpus_df = pd.DataFrame({"file_path": ["foo/a/Training.test",
                                                         "foo/b/Training.test"]})
        output = utils.TrainingData.load_sequential_data(mock_obj)

        self.assertTrue(output["text"][0] == ["Das", "ist", "ein", "Test"] and
                        output["sequence_label"][1] == ['0', '0', '1', '1'])

    @unittest.mock.patch("lib.utils.context_grabber")
    def test_add_sequential_context(self, mock_context_grabber):

        mock_context_grabber.return_value = ['padder padder padder a b c padder',
                                             'padder padder a b c padder padder',
                                             'padder a b c padder padder padder']

        mock_obj = unittest.mock.Mock()
        mock_obj.corpus_df = pd.DataFrame({"text": ["test"]})


        output = utils.TrainingData.add_sequential_context(mock_obj)
        self.assertTrue(output["sequence_training"][0] == ['padder padder padder a b c padder',
                                                           'padder padder a b c padder padder',
                                                           'padder a b c padder padder padder'])

    @unittest.mock.patch("lib.utils.fill_parameters")
    @unittest.mock.patch("lib.utils.single_run_paramsearch")
    def test_td_paramsearch(self, mock_single_run_paramsearch, mock_fill_parameters):

        mock_single_run_paramsearch.return_value = {"score": 1}

        mock_fill_parameters.return_value = {"chunk_scope": ["test"], "num_words": [3], "file_ending": ["txt"],
                                             "folder": ["test"],
                                             "windowsize": [4, 8], "ratio": [0.1], "maxlen": [6], "gram_size": [3],
                                             "chunk_size": [0], "lang": ["de"], "sequence_scope": ["test"],
                                             "categorical_scope": ["test"]}

        output = utils.td_paramsearch([], [], [])
        self.assertTrue(any(output == [{'score': 1}, {'score': 1}]))

    @unittest.mock.patch("lib.utils.TrainingData")
    def test_single_run_paramsearch(self, mock_init):

        mock_obj = unittest.mock.Mock()
        mock_obj.x_train = [0]

        mock_init.return_value = mock_obj

        mock_obj2 = unittest.mock.Mock()
        mock_obj2.score = lambda x, y: 1
        mock_obj2.fit = lambda x, y: 1

        output = utils.single_run_paramsearch([], [mock_obj2], [], [], [], [], [], [], [], [], [], [], [], [])
        self.assertTrue(set(output[0].keys()) == set(['maxlen', 'lang', 'ratio', 'file_ending', 'classifier',
                                                      'chunk_size',
                                                      'score', 'num_words', 'sequence_scope', 'categorical_scope',
                                                      'chunk_scope', 'folder', 'windowsize', 'gram_size']))






