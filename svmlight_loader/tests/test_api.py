from textwrap import dedent

from numpy.testing import assert_array_equal

from svmlight_loader import (
    classification_from_lines,
    multilabel_classification_from_lines,
    regression_from_lines,
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
