""" This module implements a fast and memory-efficient (no memory copying)
loader for the svmlight / libsvm sparse dataset format.  """

# Authors: Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
# License: Simple BSD.

from ._svmlight_format import _load_svmlight_file, _dump_svmlight_file
import numpy as np


def load_svmlight_file(file_path, buffer_mb=40, query_id=False):
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
    file_path: str
        Path to a file to load.
    buffer_mb : integer
        Buffer size to use for low level read
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
    data, labels, qids = _load_svmlight_file(file_path, buffer_mb)

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


def load_svmlight_files(files, buffer_mb=40, query_id=False):
    """Load dataset from multiple files in SVMlight format

    This function is equivalent to mapping load_svmlight_file over a list of
    files, except that the results are concatenated into a single, flat list
    and the samples vectors are constrained to all have the same number of
    features.

    Parameters
    ----------
    files : iterable over str
        Paths to files to load.

    n_features: int or None
        The number of features to use. If None, it will be inferred from the
        first file. This argument is useful to load several files that are
        subsets of a bigger sliced dataset: each subset might not have
        examples of every feature, hence the inferred shape might vary from
        one slice to another.

    Returns
    -------
    [X1, y1, ..., Xn, yn]

    where each (Xi, yi, [comment_i, query_id_i]) tuple is the result from load_svmlight_file(files[i]).

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
    result = list(load_svmlight_file(next(files), buffer_mb, query_id=query_id))

    for f in files:
        result += load_svmlight_file(f, buffer_mb, query_id=query_id)

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

    f : str
        Specifies the path that will contain the data.

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
    if hasattr(f, "write"):
        raise ValueError("File handler not supported. Use a file path.")

    # todo: this test is ok if we know the number of features! to divide X.shape[0] with
    # if X.shape[0] != y.shape[0]:
    #     raise ValueError("X.shape[0] and y.shape[0] should be the same, "
    #                      "got: %r and %r instead." % (X.shape[0], y.shape[0]))

    # elif X.shape[0] != len(query_id):
    #         raise ValueError("X.shape[0] and len(query_id) should be the same, "
    #                      "got: %r and %r instead." % (X.shape[0], len(query_id)))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if query_id is None:
        query_id = np.ndarray(0, dtype=np.int32)
    else:
        query_id = np.array(query_id, dtype=np.int32)

    _dump_svmlight_file(f, X, y, query_id, int(zero_based))
