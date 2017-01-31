from setuptools import setup, find_packages


with open('README.rst', 'r') as file:
    long_description = file.read()

setup(
    name='geneticalgs',
    version='1.0',
    description='Implementation of standard, diffusion and migration models of genetic algorithms.',
    long_description=long_description,
    author='Dmitriy Bobir',
    author_email='bobirdima@gmail.com',
    url='https://github.com/bobirdmi',
    keywords='evolutionary algorithms, genetic algorithms, optimalization, '
             'best combination, function minimum, function maximum',
    license='Public Domain',
    packages=find_packages(exclude=['tests*']),
    setup_requires=['pytest-runner'],
    install_requires=['numpy', 'bitstring>=3.1.5'],
    tests_require=['pytest', 'numpy', 'bitstring>=3.1.5'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'License :: Public Domain',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
)

