from setuptools import setup
from setuptools import find_packages


setup(
      name='PyTexture',
      version='0.1',
      description='Polycrystal data generation and exploration',
      url='',
      author='Dr Jacob Hochhalter and Cooper Hansen',
      author_email='cooperkellyhansen@gmail.com',
      license='',
      packages=find_packages(),
      classifiers=[
                'Programming Language :: Python :: 3.8.2'
                ],
      keywords='crystal plasticity',
      install_requires=[
                        'numpy',
                        'kosh',
                        'matplotlib',
                        'scikit-learn', 
                        ]

        )
