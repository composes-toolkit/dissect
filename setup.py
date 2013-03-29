#!/usr/bin/env python

from distutils.core import setup

setup(name='dissect',
      version='0.1.0',
      description='COMPOSES DISSECT TOOLKIT',
      author='Georgiana Dinu, The Nghia Pham, Marco Baroni',
      author_email='georgiana.dinu@unitn.it,thenghia.pham@unitn.it',
      url='http://http://clic.cimec.unitn.it/composes/toolkit/',
      requires=['numpy','scipy','sparsesvd'],
      package_dir={'':'src'},
      packages=['composes','composes.composition','composes.matrix','composes.semantic_space',
                'composes.exception','composes.similarity','composes.transformation','composes.utils',
                'composes.transformation.dim_reduction','composes.transformation.feature_selection',
                'composes.transformation.scaling'],
     )