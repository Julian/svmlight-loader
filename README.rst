===============
svmlight-loader
===============

|PyPI| |Pythons| |CI| |Codecov|

.. |PyPI| image:: https://img.shields.io/pypi/v/svmlight-loader.svg
  :alt: PyPI version
  :target: https://pypi.org/project/svmlight-loader/

.. |Pythons| image:: https://img.shields.io/pypi/pyversions/svmlight-loader.svg
  :alt: Supported Python versions
  :target: https://pypi.org/project/svmlight-loader/

.. |CI| image:: https://travis-ci.com/Julian/svmlight-loader.svg?branch=master
  :alt: Build status
  :target: https://travis-ci.com/Julian/svmlight-loader

.. |Codecov| image:: https://codecov.io/gh/Julian/svmlight-loader/branch/master/graph/badge.svg
  :alt: Codecov Code coverage
  :target: https://codecov.io/gh/Julian/svmlight-loader

``svmlight-loader`` is a Cython-less (and ``scikit-learn``-less)
implementation of the `svmlight / libsvm format
<http://svmlight.joachims.org/>`_.

It is designed simply to handle loading this format, which has become
somewhat prevalent in exchanging arbitrary sparse machine learning
datasets.

It also is specifically intended to support PyPy (though of course it also
supports CPython).
