#!/usr/bin/env python
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = 'src/unitest'
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name='dissect',
    version='0.1.0',
    description='COMPOSES DISSECT TOOLKIT',
    author='Georgiana Dinu, The Nghia Pham, Marco Baroni',
    author_email='georgiana.dinu@unitn.it,thenghia.pham@unitn.it',
    url='http://http://clic.cimec.unitn.it/composes/toolkit/',
    install_requires=['numpy', 'scipy', 'sparsesvd'],
    tests_require=['pytest>=2.4.2'],
    cmdclass={'test': PyTest},
    package_dir={'': 'src'},
    packages=[
        'composes',
        'composes.composition',
        'composes.matrix',
        'composes.semantic_space',
        'composes.exception',
        'composes.similarity',
        'composes.transformation',
        'composes.utils',
        'composes.transformation.dim_reduction',
        'composes.transformation.feature_selection',
        'composes.transformation.scaling',
    ],
)
