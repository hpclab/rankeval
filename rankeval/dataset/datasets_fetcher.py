import os
import six
import json
import shutil
import tarfile
import fnmatch
from os import environ
from os import makedirs
from os.path import exists
from os.path import expanduser
from os.path import join

from .dataset import Dataset
from .dataset_container import DatasetContainer

if six.PY3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen


def __dataset_catalogue__():
    resource_path = "http://rankeval.isti.cnr.it/rankeval-datasets/dataset_dictionary.json"
    json_file = urlopen(resource_path)
    data = json.load(json_file)
    return data


def __get_data_home__(data_home=None):
    """
    Return the path of the rankeval data dir.
    This folder is used by some large dataset loaders to avoid
    downloading the data several times.
    By default the data dir is set to a folder named 'rankeval_data'
    in the user home folder.
    Alternatively, it can be set by the 'RANKEVAL_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = environ.get('RANKEVAL_DATA', join('~', 'rankeval_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def __fetch_dataset_and_models__(dataset_dictionary, data_home=None,
                                 download_if_missing=True, force_download=True,
                                 with_models=True):
    """ Fetch and download a given dataset.

    Parameters
    ----------
    data_home : optional, default: None
        Specify a data folder for the datasets. If None,
        all data is stored in the '~/rankeval_data' subfolder.
    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.
    with_models : optional, True by default
        When True, the method downloads the models generated with different
        tools (QuickRank, LightGBM, XGBoost, etc.) to ease the comparison.
    """
    data_home = os.path.join(data_home, dataset_dictionary['DATASET_NAME'])
    dataset_home = os.path.join(data_home, "dataset")
    models_home = os.path.join(data_home, "models")

    # DATASET
    if not download_if_missing and not os.path.exists(data_home):
        raise IOError('dataset not found')

    # delete data_home if force_download is True, then re-create data_home dir
    if force_download:
        if os.path.exists(data_home):
            shutil.rmtree(data_home)
            os.makedirs(data_home)

    # preparing file names...
    archive_name = os.path.join(dataset_home,
                                dataset_dictionary['DATASET_ARCHIVE_NAME'])
    models_archive_name = os.path.join(models_home, dataset_dictionary[
        'MODELS_ARCHIVE_NAME'])
    train_file_path = os.path.join(dataset_home,
                                   dataset_dictionary['TRAIN_FILE'])
    test_file_path = os.path.join(dataset_home, dataset_dictionary['TEST_FILE'])

    # everything will be stored in a dictionary to return
    data = dict()

    dataset_already_downloaded = False
    if os.path.exists(dataset_home):
        dataset_already_downloaded = True

    if not dataset_already_downloaded:
        os.makedirs(dataset_home)

        print "Downloading dataset. This may take a few minutes."

        data_url = dataset_dictionary['DATASET_URL']
        print "Downloading dataset from %s " % data_url
        opener = urlopen(data_url)
        with open(archive_name, 'wb') as f:
            f.write(opener.read())

        print "Decompressing %s" % archive_name
        tarfile.open(archive_name, "r:gz").extractall(path=dataset_home)
        os.remove(archive_name)

    license_agreement = ""
    if dataset_dictionary.get('LICENSE_FILE') is not None:
        for line in open(
                os.path.join(dataset_home, dataset_dictionary['LICENSE_FILE']),
                'r'):
            license_agreement += line

    # filling data structure to return
    data['train'] = train_file_path
    data['test'] = test_file_path

    if dataset_dictionary['VALIDATION_FILE'] is not None:
        validation_file_path = os.path.join(dataset_home, dataset_dictionary[
            'VALIDATION_FILE'])
        data['validation'] = validation_file_path

    data['license_agreement'] = license_agreement

    # MODELS
    if with_models == True:
        models_already_downloaded = False
        if os.path.exists(models_home):
            models_already_downloaded = True

        if not models_already_downloaded:
            os.makedirs(models_home)

            models_url = dataset_dictionary['MODELS_URL']
            print "Downloading letor models from %s" % models_url
            opener = urlopen(models_url)
            with open(models_archive_name, 'wb') as f:
                f.write(opener.read())

            print "Decompressing %s" % models_archive_name
            tarfile.open(models_archive_name, "r:gz").extractall(
                path=models_home)
            os.remove(models_archive_name)

        # filling data structure to return
        matches = []
        for root, dirnames, filenames in os.walk(models_home):
            for filename in fnmatch.filter(filenames, '*.xml'):
                matches.append([root, filename])
        data['models'] = matches

    return data


def load_dataset_and_models(dataset_name, download_if_missing=True,
                            force_download=False, with_models=True):
    """
    The method allow to download a given dataset (and available models)
    by providing its name.

    Datasets and models are available at the following link:
        http://rankeval.isti.cnr.it/rankeval-datasets/dataset_dictionary.json

    Parameters
    ----------
    dataset_name:
        The name of the dataset (and models) to download.
    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.
    force_download : optional, False by default
        If True, download data even if it is on disk.
    with_models : optional, True by default
        When True, the method downloads the models generated with different
        tools (QuickRank, LightGBM, XGBoost, etc.) to ease the comparison.
    """
    dataset_catalogue = __dataset_catalogue__()
    dataset_dictionary = dataset_catalogue.get(dataset_name)
    if dataset_dictionary == None:
        return None

    data_home = __get_data_home__()
    data = __fetch_dataset_and_models__(dataset_dictionary, data_home,
                                        download_if_missing,
                                        force_download,
                                        with_models)

    dataset_name = dataset_dictionary['DATASET_NAME']
    dataset_format = dataset_dictionary['DATASET_FORMAT']

    container = DatasetContainer()

    if data.get('train') is not None:
        train_dataset = Dataset.load(data['train'], name=dataset_name,
                                     format=dataset_format)
        container.train_dataset = train_dataset

    if data.get('test') is not None:
        test_dataset = Dataset.load(data['test'], name=dataset_name,
                                    format=dataset_format)
        container.test_dataset = test_dataset

    if data.get('validation') is not None:
        validation_dataset = Dataset.load(data['validation'], name=dataset_name,
                                          format=dataset_format)
        container.validation_dataset = validation_dataset

    container.license_agreement = data['license_agreement']

    if with_models:
        container.models = data['models']

    return container
