"""
sklearn-specific API.
"""
import os

import numpy

import svmlight_loader


def _load_svmlight_file(
    f,
    dtype,
    multilabel,
    zero_based,
    query_id,
    offset,
    length,
):
    """
    An ``sklearn``-compatible svmlight loader.

    Drop-in replacement for
    `sklearn.datasets._svmlight_format._load_svmlight_file` (the
    internal function, not the one with all the parameter munging).
    """

    if multilabel:
        loads = svmlight_loader.multilabel_classification_from_lines
    else:
        loads = svmlight_loader.regression_from_lines

    try:
        result = loads(
            _maybe_slice(f, offset=offset, length=length),
            zero_based=zero_based,
            query_id=query_id,
        )
    except svmlight_loader.InvalidSVMLight as e:  # sklearn expects ValueErrors
        raise ValueError(str(e))

    if query_id:
        X, y, qid = result
    else:
        (X, y), qid = result, numpy.array([])

    return (
        X.dtype,
        X.data,
        X.indices.astype(numpy.longlong),
        X.indptr.astype(numpy.longlong),
        y,
        qid,
    )


def _maybe_slice(file, offset, length):
    if offset:
        # From https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_files.html#sklearn.datasets.load_svmlight_files:
        #     "Ignore the offset first bytes by seeking forward, then
        #      discarding the following bytes up until the next new line
        #      character."
        #
        # Empirically from the test suite, this seems to mean os.SEEK_SET.
        file.seek(offset)
        file.readline()
        if length > 0:
            length += offset
    for line in file:
        yield line.rstrip(b"\n")
        if file.tell() > length > 0:
            return
