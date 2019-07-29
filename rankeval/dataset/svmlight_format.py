""" This module implements a fast and memory-efficient (no memory copying)
loader for the svmlight / libsvm sparse dataset format.  """

# Authors: Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
# License: Simple BSD.

from ._svmlight_format import _load_svmlight_file, _dump_svmlight_file
import numpy as np
import io
import os
from contextlib import closing
from six import string_types


def _open(f, mode='r'):
    if mode not in ['r', 'w']:
        raise TypeError("expected mode to be 'r' or 'w', got %s" % mode)
    if mode == 'r':
        mode = 'rb'
    else:
        mode = 'wt'
    if isinstance(f, int):  # file descriptor
        return io.open(f, mode, closefd=False)
    elif not isinstance(f, string_types):
        raise TypeError("expected {str, int, file-like}, got %s" % type(f))

    _, ext = os.path.splitext(f)
    if ext == ".gz":
        import gzip
        return gzip.open(f, mode)
    elif ext == ".bz2":
        import bz2
        return bz2.open(f, mode)
    else:
        return open(f, mode)


def load_svmlight_file(f, query_id=False):
    """Load datasets in the svmlight / libsvm format into sparse CSR matrix

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    This format is used as the default format for both svmlight and the
    libsvm command line programs.

    Parsing a text based source can be expensive. When working on
    repeatedly on the same dataset, it is recommended to wrap this
    loader with joblib.Memory.cache to store a memmapped backup of the
    CSR results of the first call and benefit from the near instantaneous
    loading of memmapped structures for the subsequent calls.

    Parameters
    ----------
    f : {str, file-like, int}
        (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. A file-like or file descriptor will not be closed
        by this function. A file-like object must be opened in binary mode.
    query_id : bool
        True if the query ids has to be loaded, false otherwise

    Returns
    -------
    (X, y, [query_ids])

    where X is a dense numpy matrix of shape (n_samples, n_features) and type dtype,
          y is a ndarray of shape (n_samples,).
          query_ids is a ndarray of shape(nsamples,) if query_id is True.
          Otherwise it is not returned.
    """
    if hasattr(f, "read"):
        data, labels, qids = _load_svmlight_file(f)
    else:
        with closing(_open(f)) as f:
            data, labels, qids = _load_svmlight_file(f)

    # reshape the numpy array into a matrix
    n_samples = len(labels)
    n_features = int( len(data) / n_samples )
    data.shape = (n_samples, n_features)
    if data.dtype != np.float32:
        new_data = data.astype(dtype=np.float32)
        del data
        data = new_data

    # Convert infinite values to max_float representation (SVM reader problem)
    # This patch is needed because some dataset have infinite values and because
    # the split condition is <=, while sole software uses <. In order to
    # reconduct the former condition to the latter, we slightly decrease the
    # split. However, slightly decreasing inf does not have any effect.
    data[data == np.inf] = np.finfo(data.dtype).max

    if not query_id:
        return data, labels
    else:
        return data, labels, qids


def load_svmlight_files(files, query_id=False):
    """Load dataset from multiple files in SVMlight format

    This function is equivalent to mapping load_svmlight_file over a list of
    files, except that the results are concatenated into a single, flat list
    and the samples vectors are constrained to all have the same number of
    features.

    Parameters
    ----------
    files : iterable over {str, file-like, int}
        (Paths of) files to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. File-likes and file descriptors will not be
        closed by this function. File-like objects must be opened in binary
        mode

    n_features: int or None
        The number of features to use. If None, it will be inferred from the
        first file. This argument is useful to load several files that are
        subsets of a bigger sliced dataset: each subset might not have
        examples of every feature, hence the inferred shape might vary from
        one slice to another.

    Returns
    -------
    [X1, y1, ..., Xn, yn]

    where each (Xi, yi, [comment_i, query_id_i]) tuple is the result from
    load_svmlight_file(files[i]).

    Rationale
    ---------
    When fitting a model to a matrix X_train and evaluating it against a
    matrix X_test, it is essential that X_train and X_test have the same
    number of features (X_train.shape[1] == X_test.shape[1]). This may not
    be the case if you load them with load_svmlight_file separately.

    See also
    --------
    load_svmlight_file
    """
    files = iter(files)
    result = list(load_svmlight_file(next(files), query_id=query_id))

    for f in files:
        result += load_svmlight_file(f, query_id=query_id)

    return result


def dump_svmlight_file(X, y, f, query_id=None, zero_based=False):
    """Dump the dataset in svmlight / libsvm file format.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    Parameters
    ----------
    X : CSR sparse matrix, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    f : {str, file-like, int}
        (Path to) a file to dump. If a path ends in ".gz" or ".bz2", it will
        be compressed on the fly. If an integer is passed, it is assumed
        to be a file descriptor. A file-like or file descriptor will not be
        closed by this function. A file-like object must be opened in
        text mode.

    comment : list, optional 
        Comments to append to each row after a # character
        If specified, len(comment) must equal n_samples

    query_id: list, optional 
        Query identifiers to prepend to each row
        If specified, len(query_id) must equal n_samples

    zero_based : boolean, optional
        Whether column indices should be written zero-based (True) or one-based
        (False). Default is False.
    """
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if query_id is None:
        query_id = np.ndarray(0, dtype=np.int32)
    else:
        query_id = np.array(query_id, dtype=np.int32)

    if hasattr(f, "write"):
        _dump_svmlight_file(f, X, y, query_id, int(zero_based))
    else:
        with closing(_open(f, mode='w')) as f:
            _dump_svmlight_file(f, X, y, query_id, int(zero_based))