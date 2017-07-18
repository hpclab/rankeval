import logging
import os
import unittest

from rankeval.model import RTEnsemble
from rankeval.test.base import data_dir

model_file = os.path.join(data_dir, "quickrank.model.xml")


class ProxyModelTestCase(unittest.TestCase):

    def test_not_supported_model(self):
        try:
            RTEnsemble(model_file, format="unsupported")
            # if we reach the code below, it means the constructor
            # has not failed...raise error!
            assert False
        except TypeError:
            pass

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    unittest.main()
