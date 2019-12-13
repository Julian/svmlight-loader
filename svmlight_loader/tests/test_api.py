from textwrap import dedent

from numpy.testing import assert_array_equal
import pytest

from svmlight_loader import (
    InvalidSVMLight,
    classification_from_lines,
    multilabel_classification_from_lines,
    regression_from_lines,
)


all_loaders = pytest.mark.parametrize(
    "from_lines", [
        classification_from_lines,
        multilabel_classification_from_lines,
        regression_from_lines,
    ],
)


def test_simple():
    X, y = classification_from_lines([b"-1 1:0.43 3:0.12 9:0.2"])
    assert_array_equal(y, [-1])
    assert_array_equal(
        X.toarray(),
        [[0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2]],
    )


def test_multiple_rows():
    X, y = classification_from_lines(
        dedent(
            """\
            1 1:0.43 3:0.12 9:0.2
            0 2:0.12 8:0.2
            1 3:0.01 4:0.3
            """
        ).encode().splitlines(),
    )
    assert_array_equal(y, [1, 0, 1])
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0.12, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0.01, 0.3, 0, 0, 0, 0, 0],
        ],
    )


def test_crazy_whitespace():
    X, y = classification_from_lines(
        (
            b"  1 1:0.43 3:0.12 9:0.2      \n"
            b"0 2:0.12 8:0.2   \n"
            b"              1 3:0.01 4:0.3      "
        ).splitlines(),
    )
    assert_array_equal(y, [1, 0, 1])
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0.12, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0.01, 0.3, 0, 0, 0, 0, 0],
        ],
    )


def test_multilabel():
    X, y = multilabel_classification_from_lines(
        dedent(
            """\
            1,2 1:0.43 3:0.12 9:0.2
            2 2:0.12 8:0.2
            2,3,4 3:0.01 4:0.3
             6:0.01 7:0.3
            """
        ).encode().splitlines(),
    )
    assert_array_equal(y, [(1, 2), (2,), (2, 3, 4), ()])
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0.12, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0.01, 0.3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.01, 0.3, 0, 0],
        ],
    )


def test_crazy_multilabel_whitespace():
    X, y = multilabel_classification_from_lines(
        (
            b"  1,2 1:0.43 3:0.12 9:0.2      \n"
            b"2 2:0.12 8:0.2   \n"
            b"              2,3 3:0.01 4:0.3      "
        ).splitlines(),
    )
    assert_array_equal(y, [(1, 2), (2,), (2, 3)])
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0.12, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0.01, 0.3, 0, 0, 0, 0, 0],
        ],
    )


def test_regression():
    X, y = regression_from_lines(
        dedent(
            """\
            0.2 1:0.43 3:0.12 9:0.2
            3000.7 2:0.12 8:0.2
            240.234 3:0.01 4:0.3
            0.001 6:0.01 7:0.3
            """
        ).encode().splitlines(),
    )
    assert_array_equal(y, [0.2, 3000.7, 240.234, 0.001])
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0.12, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0.01, 0.3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.01, 0.3, 0, 0],
        ],
    )


def test_crazy_regression_whitespace():
    X, y = regression_from_lines(
        (
            b"  1.7 1:0.43 3:0.12 9:0.2      \n"
            b"0.3 2:0.12 8:0.2   \n"
            b"              1 3:0.01 4:0.3      "
        ).splitlines(),
    )
    assert_array_equal(y, [1.7, 0.3, 1])
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0.12, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0.01, 0.3, 0, 0, 0, 0, 0],
        ],
    )


@all_loaders
def test_invalid_order(from_lines):
    with pytest.raises(InvalidSVMLight) as e:
        from_lines(
            dedent(  # second line has column indexes in the wrong order
                """\
                0 2:0.12 8:0.2
                1 3:0.43 1:0.12 9:0.2
                """
            ).encode().splitlines(),
        )
    assert "example 2" in str(e.value)


@all_loaders
def test_zero_index_in_nonzero_based_file(from_lines):
    with pytest.raises(InvalidSVMLight) as e:
        from_lines([b"-1 0:0.12 9:0.2"], zero_based=False)
    assert "example 1" in str(e.value)


@all_loaders
def test_empty_line(from_lines):
    X, y = from_lines(
        dedent(
            """\
            1 1:0.43 3:0.12 9:0.2
            0
            0 3:0.12
            """
        ).encode().splitlines(),
    )
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.12, 0, 0, 0, 0, 0, 0],
        ],
    )


@all_loaders
def test_empty_lines_at_end(from_lines):
    X, y = from_lines(
        dedent(
            """\
            1 1:0.43 3:0.12 9:0.2
            0
            1
            """
        ).encode().splitlines(),
    )
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0] * 9,
            [0] * 9,
        ],
    )


@all_loaders
def test_all_empty_lines(from_lines):
    X, y = from_lines(
        dedent(
            """\
            1
            0
            1
            """
        ).encode().splitlines(),
    )
    assert_array_equal(X.toarray(), [[], [], []])


@all_loaders
def test_query_ids_are_ignored_by_default(from_lines):
    X, _ = from_lines(
        dedent(
            """\
            1 qid:1 1:0.43 3:0.12 9:0.2
            0 qid:2 2:0.12 8:0.2
            1 qid:1 3:0.01 4:0.3
            """
        ).encode().splitlines(),
    )
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0.12, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0.01, 0.3, 0, 0, 0, 0, 0],
        ],
    )


@all_loaders
def test_query_ids_are_returned_if_requested(from_lines):
    X, _, query_id = from_lines(
        dedent(
            """\
            1 qid:1 1:0.43 3:0.12 9:0.2
            0 qid:2 2:0.12 8:0.2
            1 qid:1 3:0.01 4:0.3
            """
        ).encode().splitlines(),
        query_id=True,
    )
    assert_array_equal(query_id, [1, 2, 1])
    assert_array_equal(
        X.toarray(), [
            [0.43, 0, 0.12, 0, 0, 0, 0, 0, 0.2],
            [0, 0.12, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0.01, 0.3, 0, 0, 0, 0, 0],
        ],
    )
