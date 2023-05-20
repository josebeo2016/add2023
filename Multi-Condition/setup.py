from setuptools import find_packages, setup
from audio_augmentor import (
    __version__,
    __author__,
    __license__,
    __description__,
)
setup(
    name='audio_augmentor',
    packages=find_packages(include=['audio_augmentor']),
    version=__version__,
    description=__description__,
    author=__author__,
    license=__license__,
    install_requires=['pydub', 'librosa', 'numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)