# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Franco Maria Nardini <francomaria.nardini@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import json

"""
This is a simple python script that generates the dataset catalogue. It is then
dumped in JSON for simple and easy handling.
"""

output_file = "dataset_dictionary.json"

def __add_istella_full__(data):
    item = {'TRAIN_FILE': 'full/train.txt',
            'TEST_FILE': 'full/test.txt',
            'VALIDATION_FILE': 'None',
            'LICENSE_FILE': 'istella-letor-LA.txt',
            'DATASET_ARCHIVE_NAME': 'istella-letor.tar.gz',
            'MODELS_ARCHIVE_NAME': 'istella-letor-models.tar.gz',
            # DATASET_URL = ("http://library.istella.it/" "dataset/istella-letor.tar.gz")
            'DATASET_URL': ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-letor/dataset/istella-letor.tar.gz"),
            'MODELS_URL' : ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-letor/models/istella-letor-models.tar.gz"),
            'BLOG_POST_URL': 'http://blog.istella.it/istella-learning-to-rank-dataset/',
            'DATASET_NAME': 'istella-full',
            'DATASET_DESCRIPTION': 'The istella LETOR full dataset',
            'DATASET_FORMAT': 'svmlight'}
    data[item['DATASET_NAME']] = item
    return data

def __add_istella_sample__(data):
    item = {'TRAIN_FILE': 'sample/train.txt',
            'TEST_FILE': 'sample/test.txt',
            'VALIDATION_FILE': 'sample/vali.txt',
            'LICENSE_FILE': 'istella-letor-LA.txt',
            'DATASET_ARCHIVE_NAME': 'istella-s-letor.tar.gz',
            'MODELS_ARCHIVE_NAME': 'istella-s-letor-models.tar.gz',
            # DATASET_URL = ("http://library.istella.it/" "dataset/istella-letor.tar.gz")
            'DATASET_URL': ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-s-letor/dataset/istella-s-letor.tar.gz"),
            'MODELS_URL' : ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-s-letor/models/istella-s-letor-models.tar.gz"),
            'BLOG_POST_URL': 'http://blog.istella.it/istella-learning-to-rank-dataset/',
            'DATASET_NAME': 'istella-sample',
            'DATASET_DESCRIPTION': 'The istella LETOR sample dataset',
            'DATASET_FORMAT': 'svmlight'}
    data[item['DATASET_NAME']] = item
    return data

def __add_msn10k__(data):
    item = {'COMMON_SUBFOLDER_NAME': 'Fold',
            'TRAIN_FILE': 'train.txt',
            'TEST_FILE': 'test.txt',
            'VALIDATION_FILE': 'vali.txt',
            'DATASET_ARCHIVE_NAME': 'msn10k.tar.gz',
            'MODELS_ARCHIVE_NAME': 'msn10k-models.tar.gz',
            'DATASET_URL': ("http://rankeval.isti.cnr.it/" "rankeval-datasets/msn10k/dataset/msn10k.tar.gz"),
            'MODELS_URL' : ("http://rankeval.isti.cnr.it/" "rankeval-datasets/msn10k/models/msn10k-models.tar.gz"),
            'BLOG_POST_URL': 'https://www.microsoft.com/en-us/research/project/mslr/',
            'DATASET_NAME': 'msn10k',
            'DATASET_DESCRIPTION': 'Microsoft Learning to Rank Datasets (WEB10K)',
            'DATASET_FORMAT': 'svmlight'}
    data[item['DATASET_NAME']] = item
    return data

def __add_msn30k__(data):
    item = {'COMMON_SUBFOLDER_NAME': 'Fold',
            'TRAIN_FILE': 'train.txt',
            'TEST_FILE': 'test.txt',
            'VALIDATION_FILE': 'vali.txt',
            'DATASET_ARCHIVE_NAME': 'msn30k.tar.gz',
            'MODELS_ARCHIVE_NAME': 'msn30k-models.tar.gz',
            'DATASET_URL': ("http://rankeval.isti.cnr.it/" "rankeval-datasets/msn30k/dataset/msn30k.tar.gz"),
            'MODELS_URL' : ("http://rankeval.isti.cnr.it/" "rankeval-datasets/msn30k/models/msn30k-models.tar.gz"),
            'BLOG_POST_URL': 'https://www.microsoft.com/en-us/research/project/mslr/',
            'DATASET_NAME': 'msn30k',
            'DATASET_DESCRIPTION': 'Microsoft Learning to Rank Datasets (WEB30K)',
            'DATASET_FORMAT': 'svmlight'}
    data[item['DATASET_NAME']] = item
    return data

def main():
    data = dict()
    __add_istella_full__(data)
    __add_istella_sample__(data)
    __add_msn10k__(data)
    __add_msn30k__(data)
    with open(output_file, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()
