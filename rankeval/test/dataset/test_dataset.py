import logging
import os
import unittest

from rankeval.dataset import Dataset
from ..base import data_dir

datafile = os.path.join(data_dir, "msn1.fold1.test.5k.txt")


class SVMLightLoaderTestCase(unittest.TestCase):

    def test_svmlight_dataset(self):
        try:
            dataset = Dataset.load(datafile, format="svmlight")
        except TypeError:
            assert False

    def test_not_supported_dataset(self):
        try:
            Dataset.load(datafile, format="unsupported")
            # if we reach the code below, it means the constructor has not failed...raise error!
            assert False
        except TypeError:
            pass

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
