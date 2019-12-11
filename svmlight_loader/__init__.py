try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata
__version__ = metadata.version(__name__)

from svmlight_loader._api import (
    InvalidSVMLight,
    classification_from_lines,
    multilabel_classification_from_lines,
    regression_from_lines,
)
