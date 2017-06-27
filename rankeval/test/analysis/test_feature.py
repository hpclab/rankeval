import os
import unittest
import logging

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from rankeval.analysis.feature import feature_importance

from rankeval.test.base import data_dir

from rankeval.core.model import RTEnsemble
from rankeval.core.dataset import Dataset


class FeatureImportanceTestCase(unittest.TestCase):

    def setUp(self):
        self.model = RTEnsemble(os.path.join(data_dir, "quickrank.model.xml"), format="quickrank")
        self.model = RTEnsemble(os.path.join(data_dir, "quickrank.model.xml"), format="quickrank")
        self.dataset = Dataset.load(os.path.join(data_dir, "msn1.fold1.train.5k.txt"), format="svmlight")

    def tearDown(self):
        del self.model
        self.model = None
        del self.dataset
        self.dataset = None
 
    def test_feature_importance(self):
        feature_imp = feature_importance(self.dataset, self.model)

        print(feature_imp)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
