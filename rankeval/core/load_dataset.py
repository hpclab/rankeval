"""
     Here we test the svmlight_loader utility.
"""

# rankeval : load_dataset
# Created by muntean on 4/20/17

import argparse
from memory_profiler import profile
from svmlight_loader import load_svmlight_file

parser = argparse.ArgumentParser(description='Import a dataset in light-svm format.')
parser.add_argument('-i', '--input', type=str, help='the input file')

@profile
def loadDataset(filename):
    """
    Tries to load a dataset in svm light format and does memory profiling.
    :param filename: the input datatset file
    :return: 
    """
    # X, y, q = load_svmlight_file(filename, comment=False, query_id=True)
    X, y, q = load_svmlight_file(filename, query_id=True)
    print X.shape
    print y.shape
    print q.shape

if __name__ == '__main__':
    args = parser.parse_args()
    print args
    loadDataset(args.input)
