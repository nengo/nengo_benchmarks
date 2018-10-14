from setuptools import setup

setup(
    name='nengo-benchmarks',
    packages=['nengo_benchmarks'],
    version='0.5',
    author='Terry Stewart',
    description='Benchmarking models for Nengo',
    url='https://github.com/nengo/nengo-benchmarks',
    license='See LICENSE.rst',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=[
        'pytry',
        'scikit-learn',
    ]
)
