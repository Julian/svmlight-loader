from scipy.sparse import csr_matrix
import attr
import numpy


@attr.s(hash=True)
class InvalidSVMLight(Exception):

    _example = attr.ib()
    _reason = attr.ib()

    def __str__(self):
        return "{0._reason} (example {0._example})".format(self)


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
        labels, _, rest = line.partition(b" ")
        y.append(load_labels(labels))
        features = rest.split()
        last_column = -1
        for total_real_features, each in enumerate(features):
            column, value = each.split(b":")
            if column == b"qid":
                continue
            column = int(column)
            if not zero_based:
                column -= 1

            if column < last_column:
                raise InvalidSVMLight(
                    reason="features are not in increasing order",
                    example=i + 1,
                )
            last_column = column

            row_ind.append(i)
            col_ind.append(column)
            data.append(float(value))
    return data, row_ind, col_ind, y


def _strip_comments(lines):
    for line in lines:
        noncomment, _, _ = line.partition(b"#")
        if noncomment:
            yield noncomment
