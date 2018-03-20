#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unittest.mock import patch, mock_open
from lib import utils
import unittest
import pandas as pd


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
                                                       "windowsize", "ratio", "maxlen", "gram_size",
                                                       "chunk_size", "lang"})

    def test_context_grabber(self):

        context = utils.context_grabber(["a", "b", "c"], 3)
        self.assertTrue(context == ['padder padder padder a b c padder',
                                    'padder padder a b c padder padder',
                                    'padder a b c padder padder padder'])

    def test_TrainingData_init(self):

        test_object = utils.TrainingData(scope="test", num_words=0, file_ending=".test", folder="/foo/bar",
                                         windowsize=0, ratio=0.1, maxlen=0, chunk_size=0, gram_size=0, lang="")

        self.assertTrue(test_object.scope == "test" and
                        test_object.num_words == 0 and
                        test_object.file_ending == ".test" and
                        test_object.folder == "/foo/bar" and
                        test_object.windowsize == 0 and
                        test_object.ratio == 0.1 and
                        test_object.maxlen == 0 and
                        test_object.chunk_size == 0 and
                        test_object.gram_size == 0 and
                        test_object.lang == "" and
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


