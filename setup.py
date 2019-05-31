"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))

with open(path.join(here, '_version.py')) as version_file:
    exec(version_file.read())

with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
    changelog = changelog_file.read()

#with open(path.join(here, 'CITATION.md')) as citation_file:
#    citation = citation_file.read()

long_description = readme + '\n\n' + changelog# + '\n\n' + citation

install_requires = [
    'numpy',
    'keras',
    'networkx',
    'osmnx',
    'pandas'
    'scikit-learn'
]

tests_require = [
    'pytest>=3.2.0',
    'pytest-cov',
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(
    name='osm_multiplex',
    version=__version__,

    description='Transforming OpenStreetMap data into a multiplex transportation network',
    long_description=long_description,
    url='https://github.com/SoftwareDevEngResearch/osm_multiplex',

    author='Sylvan Hoover',
    author_email='hooversy@oregonstate.edu',

    packages=['osm_multiplex', 'osm_multiplex.tests'],
    package_dir={'osm_multiplex': 'osm_multiplex'},
    include_package_data=True,
    package_data={'osm_multiplex': ['tests/*.xml', 'tests/*.yaml', 'tests/dataset_file.txt', 'tests/*.cti']},
    install_requires=install_requires,
    zip_safe=False,

    license='MIT License',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ],
    keywords='multiplex_transportation',

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'osm_multiplex=osm_multiplex.__main__:main',
        ],
    },

    tests_require=tests_require,
    setup_requires=setup_requires,
    #extras_require=extras_require,
)