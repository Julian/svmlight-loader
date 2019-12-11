from scipy.sparse import csr_matrix
import numpy


def classification_from_lines(lines, zero_based=False):
    """
    Load a series of lines from a classification dataset in svmlight format.
    """
    data, row_ind, col_ind, y = _loads(
        lines,
        zero_based=zero_based,
        load_labels=int,
    )
    X = csr_matrix(
        (numpy.array(data), (numpy.array(row_ind), numpy.array(col_ind))),
    )
    return X, numpy.array(y)


def regression_from_lines(lines, zero_based=False):
    """
    Load a series of lines from a regression dataset in svmlight format.
    """
    data, row_ind, col_ind, y = _loads(
        lines,
        zero_based=zero_based,
        load_labels=float,
    )
    X = csr_matrix(
        (numpy.array(data), (numpy.array(row_ind), numpy.array(col_ind))),
    )
    return X, numpy.array(y)


def multilabel_classification_from_lines(lines, zero_based=False):
    """
    Load a series of lines from a multilabel dataset in svmlight format.
    """
    data, row_ind, col_ind, y = _loads(
        lines,
        zero_based=zero_based,
        load_labels=lambda labels: tuple(
            sorted(int(label) for label in labels.split(b",") if label)
        ),
    )
    X = csr_matrix(
        (numpy.array(data), (numpy.array(row_ind), numpy.array(col_ind))),
    )
    return X, y


def _loads(lines, load_labels, zero_based):
    data, row_ind, col_ind, y = [], [], [], []
    for i, line in enumerate(_strip_comments(lines)):
        labels, rest = line.split(b" ", 1)
        y.append(load_labels(labels))
        features = rest.split()
        row_ind.extend([i] * len(features))
        for each in features:
            column, value = each.split(b":")
            column = int(column)
            if not zero_based:
                column -= 1
            col_ind.append(column)
            data.append(float(value))
    return data, row_ind, col_ind, y


def _strip_comments(lines):
    for line in lines:
        noncomment, _, _ = line.partition(b"#")
        if noncomment:
            yield noncomment
