import os
import sys
import six
import tarfile
from os import environ
from os import makedirs
from os.path import join
from os.path import exists
from os.path import expanduser

from rankeval.core.dataset import DatasetContainer
from rankeval.core.dataset import Dataset

# datasetcontainer per contenere tutti i dataset caricati dalla load: la load rende il container

# dizionario completo per i dataset
# - istella tutti e due, MSN, yahoo. (controllare licenses).

if six.PY3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen

def __dataset_catalogue():
    # TODO: add JSON wrapping...
    # pensavo che il json potrebbe stare proprio nella lib
    # in alternativa su un host remoto in modo che cambiamo quello e via ma lo trovo piu' pericoloso.
    main = dict()
    r = {'TRAIN_FILE': 'full/train.txt',
         'TEST_FILE': 'full/test.txt',
         'LICENSE_FILE': 'istella-letor-LA.txt',
         'ARCHIVE_NAME': 'istella-letor.tar.gz',
         #DATASET_URL = ("http://library.istella.it/" "dataset/istella-letor.tar.gz")
         'DATASET_URL' : ("http://rankeval.isti.cnr.it/" "rankeval-datasets/istella-letor/dataset/istella-letor.tar.gz"),
         'DATASET_NAME' : 'istella-full',
         'DATASET_DESCRIPTION' : 'The istella LETOR full dataset',
         'DATASET_FORMAT' : 'svmlight'}
    main[r['DATASET_NAME']] = r

    return main

def __get_data_home(data_home=None):
    """Return the path of the scikit-learn data dir.
    This folder is used by some large dataset loaders to avoid
    downloading the data several times.
    By default the data dir is set to a folder named 'scikit_learn_data'
    in the user home folder.
    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = environ.get('RANKEVAL_DATA',
                                join('~', 'rankeval_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home

def fetch_dataset(dataset_name, dataset_dictionary, data_home=None, subset='all', download_if_missing=True):
    """ Download the istella letor dataset.
    Blog post: http://blog.istella.it/istella-learning-to-rank-dataset/
    Direct HTTP URL: http://library.istella.it/dataset/istella-letor.tar.gz

    Parameters
    ----------
    subset : 'train' or 'test' or 'all', optional
        Select the dataset to download:
          'train' for the training set,
          'test' for the test set,
          'all' for both.
    data_home : optional, default: None
        Specify a data folder for the datasets. If None,
        all data is stored in the '~/rankeval_data' subfolder.
    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.
    """

    data_home = __get_data_home(data_home=data_home)
    istella_letor_home = os.path.join(data_home, "istella_letor_home")

    if (not download_if_missing) and (not os.path.exists(istella_letor_home)):
        raise IOError('istella-letor dataset not found')

    print "Downloading dataset. This may take a few minutes."
    archive_path = os.path.join(istella_letor_home, dataset_dictionary['ARCHIVE_NAME'])
    train_path = os.path.join(istella_letor_home, dataset_dictionary['TRAIN_FILE'])
    test_path = os.path.join(istella_letor_home, dataset_dictionary['TEST_FILE'])

    data = dict()
    data['train'] = train_path
    data['test'] = test_path

    if not os.path.exists(istella_letor_home):
        os.makedirs(istella_letor_home)

    if os.path.exists(archive_path):
        # Download is not complete as the .tar.gz file is removed after download.
        print "Download was incomplete, downloading again."
        os.remove(archive_path)

    print ("Downloading dataset from %s ", dataset_dictionary['URL'])
    opener = urlopen(dataset_dictionary['URL'])
    with open(archive_path, 'wb') as f:
        f.write(opener.read())

    print "Decompressing %s", archive_path
    tarfile.open(archive_path, "r:gz").extractall(path=istella_letor_home)
    os.remove(archive_path)

    if subset in ('train', 'test'):
        filter = dict()
        filter[subset] = data[subset]
        data = filter
    elif subset == 'all':
        pass
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    license_agreement = ""
    for line in open(os.path.join(istella_letor_home, dataset_dictionary['LICENSE_FILE']), 'r'):
        license_agreement += line
    data['license_agreement'] = license_agreement

    return data

def load_dataset(dataset_name, download_if_missing=True):
    dataset_catalogue = __dataset_catalogue()
    dataset_dictionary = dataset_catalogue.get(dataset_name)
    if dataset_dictionary == None:
        return None

    data = fetch_dataset(dataset_name, dataset_dictionary, download_if_missing)

    container = DatasetContainer()
    train_dataset = Dataset.load(data['train'], name=dataset_dictionary['DATASET_NAME'], format=dataset_dictionary['DATASET_FORMAT'])
    test_dataset = Dataset.load(data['test'], name=dataset_dictionary['DATASET_NAME'], format=dataset_dictionary['DATASET_FORMAT'])

    if (dataset_dictionary.get('VALIDATION_FILE') is not None):
        validation_dataset = Dataset.load(data['validation'], name=dataset_dictionary['DATASET_NAME'], format=dataset_dictionary['DATASET_FORMAT'])

def main():
    dataset = fetch_dataset(name='istella-full')
    print dataset

if __name__ == "__main__":
    main()