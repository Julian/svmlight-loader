#!/usr/bin/env python
"""
A performance benchmark against sklearn.datasets.load_svmlight_file.
"""
from pyperf import Runner

import sklearn.datasets

import svmlight_loader


if __name__ == "__main__":
    runner = Runner()
