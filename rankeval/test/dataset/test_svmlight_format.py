import logging
import os
import unittest

import numpy as np
from nose.tools import raises
from numpy.testing import assert_equal, assert_array_equal, assert_array_almost_equal

try:
    from sklearn.datasets import load_svmlight_file as sk_load_svmlight_file
    scikit_missing = False
except ImportError:
    scikit_missing = True

from rankeval.dataset.svmlight_format import load_svmlight_file, \
    load_svmlight_files, dump_svmlight_file
from rankeval.test.base import data_dir

datafile = os.path.join(data_dir, "svmlight_classification.txt")
invalidfile = os.path.join(data_dir, "svmlight_invalid.txt")
qid_datafile = os.path.join(data_dir, "svmlight_classification_qid.txt")


class SVMLightLoaderTestCase(unittest.TestCase):

    def test_load_svmlight_qid_file(self):
        X, y, q = load_svmlight_file(qid_datafile, query_id=True)

        # test X's shape
        assert_array_equal(X.shape, (4, 33))
        #print X

        # test X's non-zero values
        # tests X's zero values
        # test can change X's values

        # test y
        assert_array_equal(y, [1, 2, 0, 3])

        # test q
        # print q
        assert_array_equal(q, [1, 37, 37, 12])

    def test_load_svmlight_file_empty_qid(self):
        X, y, q = load_svmlight_file(datafile, query_id=True)

        # test X's shape
        assert_array_equal(X.shape, (3, 20))

        # test X's non-zero values
        # tests X's zero values
        # test can change X's values

        # test y
        assert_array_equal(y, [1, 2, 3])

        # test q
        assert_equal(q.shape[0], 0)

    def test_load_svmlight_file(self):
        X, y = load_svmlight_file(datafile)

        # test X's shape
        assert_array_equal(X.shape, (3, 20))

        # test X's non-zero values
        # tests X's zero values
        # test can change X's values

        # test y
        assert_array_equal(y, [1, 2, 3])

    def test_load_svmlight_files_comment_qid(self):
        X_train, y_train, q_train, X_test, y_test, q_test = \
            load_svmlight_files([datafile] * 2,  query_id=True)
        assert_array_equal(X_train, X_test)
        assert_array_equal(y_train, y_test)
        assert_equal(X_train.dtype, np.float32)
        assert_equal(X_test.dtype, np.float32)

        X1, y1, q1, X2, y2, q2, X3, y3, q3 = load_svmlight_files([datafile] * 3, query_id=True)
        assert_equal(X1.dtype, X2.dtype)
        assert_equal(X2.dtype, X3.dtype)
        assert_equal(X3.dtype, np.float32)

    def test_load_svmlight_files(self):
        # print load_svmlight_files([datafile] * 2)
        X_train, y_train, X_test, y_test = load_svmlight_files([datafile] * 2, query_id=False)
        assert_array_equal(X_train, X_test)
        assert_array_equal(y_train, y_test)
        assert_equal(X_train.dtype, np.float32)
        assert_equal(X_test.dtype, np.float32)

        X1, y1, X2, y2, X3, y3 = load_svmlight_files([datafile] * 3, query_id=False)
        assert_equal(X1.dtype, X2.dtype)
        assert_equal(X2.dtype, X3.dtype)
        assert_equal(X3.dtype, np.float32)

    def test_load_invalid_file(self):
        try:
            load_svmlight_file(invalidfile)
            assert False
        except RuntimeError:
            pass

    def test_load_invalid_file2(self):
        try:
            load_svmlight_files([datafile, invalidfile, datafile])
            assert False
        except RuntimeError:
            pass

    @raises(TypeError)
    def test_not_a_filename(self):
        load_svmlight_file(1)

    @raises(IOError)
    def test_invalid_filename(self):
        load_svmlight_file("trou pic nic douille")

    @unittest.skipIf(scikit_missing, "Scikit-Learn package missing")
    def test_dump(self):
        tmpfile = "tmp_dump.txt"
        try:
            # loads from file
            Xs, y = load_svmlight_file(datafile)

            # dumps to file
            dump_svmlight_file(Xs, y, tmpfile, zero_based=False)

            # loads them as CSR MATRIX
            X2, y2 = sk_load_svmlight_file(tmpfile)

            X3 = np.ndarray(shape=X2.shape, dtype=X2.dtype)
            X2.toarray(out=X3)

            # check assertions
            assert_array_almost_equal(Xs, X3)
            assert_array_almost_equal(y, y2)
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)

    @unittest.skipIf(scikit_missing, "Scikit-Learn package missing")
    def test_dump_qid(self):
        tmpfile = "/tmp/tmp_dump.txt"
        try:
            # loads from file
            Xs, y, q = load_svmlight_file(qid_datafile, query_id=True)

            # dumps to file
            dump_svmlight_file(Xs, y, tmpfile, query_id=list(q), zero_based=False)

            # loads them as CSR MATRIX with scikit-learn
            X2, y2, q2 = sk_load_svmlight_file(tmpfile, query_id=True)

            X3 = np.ndarray(shape=X2.shape, dtype=X2.dtype)
            X2.toarray(out=X3)

            # check assertions
            assert_array_almost_equal(Xs, X3)
            assert_array_almost_equal(y, y2)
            assert_array_equal(q, q2)
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
