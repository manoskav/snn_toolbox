"""
Setup SNN toolbox

"""

import os
import sys
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


with open('README.rst') as file:
    long_description = file.read()


# Tell setuptools to run 'tox' when calling 'python setup.py test'.
class Tox(TestCommand):
    user_options = [('tox-args=', 'a', "Arguments to pass to tox")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import tox
        import shlex
        args = self.tox_args
        if args:
            args = shlex.split(self.tox_args)
        errno = tox.cmdline(args=args)
        sys.exit(errno)

setup(
    name='snntoolbox (MOD)',
    version='0.2.dev1',  # see https://www.python.org/dev/peps/pep-0440/
    description='Spiking neural network conversion toolbox (MOD)',
    long_description=long_description,
    author='Bodo Rueckauer, Manos Kav',
    author_email='bodo.rueckauer@gmail.com, kavvousanos.em@gmail.com',
    url='https://github.com/manoskav/snn_toolbox',
    download_url='https://github.com/manoskav/snn_toolbox.git',
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        # 'Development Status :: 3 - Alpha',

        # Indicate who this project is intended for
        # 'Intended Audience :: Researchers',
        # 'Topic :: Software Development :: Build Tools',

        # License
        'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='neural networks, deep learning, spiking',

    install_requires=[
        'future',
        'keras',
        'h5py'
    ],

    setup_requires=['pytest-runner'],

    tests_require=['tox', 'pytest'],

    cmdclass={'test': Tox},  # , 'build_doc': BuildDoc},

    # Additional groups of dependencies (e.g. development dependencies).
    # Install them with $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['foo'],
    #     'test': ['bar']
    # },

    packages=find_packages(exclude=['scripts']),

    package_data={
        'snntoolbox': ['config_defaults']
    },

    # Include documentation
    data_files=[
        ('', ['README.rst']),
    ],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': ['snntoolbox=snntoolbox.bin.run:main']
        # 'gui_scripts': ['snntoolbox_gui=snntoolbox.bin.gui.gui:main']
    },
)
