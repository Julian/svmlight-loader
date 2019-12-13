from scipy.sparse import csr_matrix
import attr
import numpy


@attr.s(hash=True)
class InvalidSVMLight(Exception):

    _example = attr.ib()
    _reason = attr.ib()

    def __str__(self):
        return "{0._reason} (example {0._example})".format(self)


def classification_from_lines(lines, zero_based=False, query_id=False):
    """
    Load a series of lines from a classification dataset in svmlight format.
    """
    X, y, qid = _loads(lines, zero_based=zero_based, load_labels=int)
    y = numpy.array(y)
    if query_id:
        return X, y, numpy.array(qid)
    return X, y


def regression_from_lines(lines, zero_based=False, query_id=False):
    """
    Load a series of lines from a regression dataset in svmlight format.
    """
    X, y, qid = _loads(lines, zero_based=zero_based, load_labels=float)
    y = numpy.array(y)
    if query_id:
        return X, y, numpy.array(qid)
    return X, y


def multilabel_classification_from_lines(
    lines,
    zero_based=False,
    query_id=False,
):
    """
    Load a series of lines from a multilabel dataset in svmlight format.
    """
    X, y, qid = _loads(
        lines,
        zero_based=zero_based,
        load_labels=lambda labels: tuple(
            sorted(int(label) for label in labels.split(b",") if label)
        ),
    )
    if query_id:
        return X, y, numpy.array(qid)
    return X, y


def _loads(lines, load_labels, zero_based):
    data, indices, indptr, y, query_id = [], [], [0], [], []
    for line in _strip_comments(lines):
        labels, _, rest = line.partition(b" ")

        if b":" in labels:
            y.append(load_labels(b""))
            features = [labels] + rest.split()
        else:
            y.append(load_labels(labels))
            features = rest.split()

        last_column = -1
        for total_real_features, each in enumerate(features):
            column, value = each.split(b":")
            if column == b"qid":
                query_id.append(int(value))
                continue
            column = int(column)
            if not zero_based:
                if column == 0:
                    raise InvalidSVMLight(
                        reason=(
                            "found a zero index but parsing "
                            "was non-zero indexed"
                        ),
                        example=len(indptr),
                    )
                column -= 1

            if column < last_column:
                raise InvalidSVMLight(
                    reason="features are not in increasing order",
                    example=len(indptr),
                )
            last_column = column

            indices.append(column)
            data.append(float(value))
        indptr.append(len(indices))

    # For completely empty matrices, make them at least have right
    # numbers of rows. scipy complains otherwise that it can't determine
    # the dimensions. For all other cases, let it figure out the
    # dimensions itself.
    if not data:
        shape = (len(indptr) - 1, 0)
    else:
        shape = None
    return csr_matrix((data, indices, indptr), shape=shape), y, query_id


def _strip_comments(lines):
    for line in lines:
        noncomment, _, _ = line.strip().partition(b"#")
        if noncomment:
            yield noncomment
