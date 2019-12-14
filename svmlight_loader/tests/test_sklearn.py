"""
Run the sklearn test suite using our implementation.

Confused on where that actually happens?

Silly programmer, learn to monkeyBOOM stuff globally.
"""
import pytest

try:
    from sklearn.datasets.tests.test_svmlight_format import *
except ImportError:
    pytest.skip("sklearn not available", allow_module_level=True)
else:
    import svmlight_loader._sklearn

    try:
        import sklearn.datasets._svmlight_format
    except ImportError:  # older sklearns (<=0.20.X, which is latest Py2)
        module = "sklearn.datasets.svmlight_format"
    else:
        module = "sklearn.datasets._svmlight_format"


    @pytest.fixture(autouse=True)
    def skpatch(monkeypatch):
        """
        GLOBAL MONKEYPATCH U GOT A PROBLM?
        """
        monkeypatch.setattr(
            module + "._load_svmlight_file",
            svmlight_loader._sklearn._load_svmlight_file,
        )
