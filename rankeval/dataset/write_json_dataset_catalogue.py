import json

output_file = "dataset_dictionary.json"

def __add_istella_full__(data):
    item =  {'TRAIN_FILE': 'full/train.txt',
             'TEST_FILE': 'full/test.txt',
             'VALIDATION_FILE': 'None',
             'LICENSE_FILE': 'istella-letor-LA.txt',
             'DATASET_ARCHIVE_NAME': 'istella-letor.tar.gz',
             'MODELS_ARCHIVE_NAME': 'istella-letor-models.tar.gz',
             # DATASET_URL = ("http://library.istella.it/" "dataset/istella-letor.tar.gz")
             'DATASET_URL': ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-letor/dataset/istella-letor.tar.gz"),
             'MODELS_URL' : ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-letor/models/istella-letor-models.tar.gz"),
             'BLOG_POST_URL': 'http://library.istella.it/dataset/istella-letor.tar.gz',
             'DATASET_NAME': 'istella-full',
             'DATASET_DESCRIPTION': 'The istella LETOR full dataset',
             'DATASET_FORMAT': 'svmlight'}
    data[item['DATASET_NAME']] = item
    return data

def __add_istella_sample__(data):
    item =  {'TRAIN_FILE': 'sample/train.txt',
             'TEST_FILE': 'sample/test.txt',
             'VALIDATION_FILE': 'sample/vali.txt',
             'LICENSE_FILE': 'istella-letor-LA.txt',
             'DATASET_ARCHIVE_NAME': 'istella-s-letor.tar.gz',
             'MODELS_ARCHIVE_NAME': 'istella-s-letor-models.tar.gz',
             # DATASET_URL = ("http://library.istella.it/" "dataset/istella-letor.tar.gz")
             'DATASET_URL': ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-s-letor/dataset/istella-s-letor.tar.gz"),
             'MODELS_URL' : ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-s-letor/models/istella-s-letor-models.tar.gz"),
             'BLOG_POST_URL': 'http://library.istella.it/dataset/istella-letor.tar.gz',
             'DATASET_NAME': 'istella-sample',
             'DATASET_DESCRIPTION': 'The istella LETOR sample dataset',
             'DATASET_FORMAT': 'svmlight'}
    data[item['DATASET_NAME']] = item
    return data

def main():
    data = dict()
    __add_istella_full__(data)
    __add_istella_sample__(data)
    with open(output_file, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()