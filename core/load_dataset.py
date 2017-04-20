"""
     Here we test the svmlight_loader utility
"""

# rankeval : load_dataset
# Created by muntean on 4/20/17

import argparse

from svmlight_loader import load_svmlight_file

parser = argparse.ArgumentParser(description='Import a dataset in light-svm format.')
parser.add_argument('-i', '--input', type=str, help='the input file')


def loadDataset(filename):
    X, y, c, q = load_svmlight_file(filename, comment=True, query_id=True)
    print X.shape
    print y.shape
    print len(c)
    print q.shape

if __name__ == '__main__':

    args = parser.parse_args()
    print args

    loadDataset(args.input)
