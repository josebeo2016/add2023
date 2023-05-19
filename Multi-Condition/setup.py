from setuptools import find_packages, setup
setup(
    name='audio_augmentor',
    packages=find_packages(include=['audio_augmentor']),
    version='0.1.0',
    description='Audio Augmentor',
    author='Thien-Phuc Doan',
    license='MIT',
    install_requires=['pydub', 'librosa', 'numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)