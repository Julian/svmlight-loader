===============
svmlight-loader
===============

|PyPI| |Pythons| |CI| |Codecov| |ReadTheDocs|

.. |PyPI| image:: https://img.shields.io/pypi/v/svmlight-loader.svg
  :alt: PyPI version
  :target: https://pypi.org/project/svmlight-loader/

.. |Pythons| image:: https://img.shields.io/pypi/pyversions/svmlight-loader.svg
  :alt: Supported Python versions
  :target: https://pypi.org/project/svmlight-loader/

.. |CI| image:: https://github.com/Julian/svmlight-loader/workflows/CI/badge.svg
  :alt: Build status
  :target: https://github.com/Julian/svmlight-loader/actions?query=workflow%3ACI

.. |Codecov| image:: https://codecov.io/gh/Julian/svmlight-loader/branch/master/graph/badge.svg
  :alt: Codecov Code coverage
  :target: https://codecov.io/gh/Julian/svmlight-loader

.. |ReadTheDocs| image:: https://readthedocs.org/projects/svmlight-loader/badge/?version=stable&style=flat
  :alt: ReadTheDocs status
  :target: https://svmlight-loader.readthedocs.io/en/stable/

``svmlight-loader`` is a Cython-less (and ``scikit-learn``-less)
implementation of the `svmlight / libsvm format
<http://svmlight.joachims.org/>`_.

It is designed simply to handle loading this format, which has become
somewhat prevalent in exchanging arbitrary sparse machine learning
datasets.

It also is specifically intended to support PyPy (though of course it also
supports CPython).
