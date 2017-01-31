geneticalgs
===========

Implementation of standard, migration and diffusion models of genetic algorithms (GA) in ``python 3.5``.

Benchmarking was conducted by `COCO platform <http://coco.gforge.inria.fr/>`__ ``v15.03``.

The project summary may be found in ``project_summary.pdf``.

Link to `PyPI <https://pypi.python.org/pypi/geneticalgs>`__.

Implemented features
====================

- standard, diffusion and migration models

  - with real values (searching for global minimum or maximum of the specified function)

  - with binary encoding combination of some input data

- old population is completely replaced with a new computed one at the end of each generation
(generational population model)

- two types of fitness value optimalization

  - minimization

  - maximization

- three parent selection types

  - *roulette wheel selection*

  - *rank wheel selection*

  - *tournament*

- may be specified mutation probability

- may be specified any amount of random bits to be mutated

- may be specified crossover probability

- different types of crossover

  - single-point

  - two-point

  - multiple point up to uniform crossover

- elitism may be turned on/off (the best individual may migrate to the next generation)

Content description
===================

- **/geneticalgs/** contains source codes

- **/docs/** contains `sphinx <http://www.sphinx-doc.org/en/stable/>`__ source codes

- **/2.7/** contains files converted from ``python 3.5`` to ``python 2.7`` using `3to2 module <https://pypi.python.org/pypi/3to2>`__ as `COCO platform <http://coco.gforge.inria.fr/>`__ used in benchmarking supports only this version of python.

- **/2.7/benchmark/** contains the following files:

  - ``my_experiment.py`` is used for running benchmarking. Read more `here <http://coco.lri.fr/COCOdoc/runningexp.html#python>`__.

  - ``my_timing.py`` is used for time complexity measurements. It has the same run conditions as the previous file.

  - ``pproc.py`` is a modified file from COCO platform distribution that must be copied to ``bbob.v15.03/python/bbob_pproc/`` in order to post-process measured data of migration GA (other models don't need it). It is necessary due to unexpected format of records in case of migration GA.

- **/benchmarking/** contains measured results and the appropriate plots of benchmarking.

- **/time_complexity/** contains time results measured using ``my_timing.py``.

- **/examples/** contains examples of using the implemented genetic algorithms.

- **/tests/** contains `pytest <http://doc.pytest.org/en/latest/>`__ tests

Requirements
============

- python 3.5+

- `NumPy <http://www.numpy.org/>`__

- `bitstring <https://pypi.python.org/pypi/bitstring/>`__

- `sphinx <http://www.sphinx-doc.org/en/stable/>`__ for documentation

- `pytest <http://doc.pytest.org/en/latest/>`__ for tests

Installation
============

Install package by typing the command

``python -m pip install --extra-index-url https://pypi.python.org/pypi geneticalgs``

Running tests
=============

You may run tests by typing from the package directory

``python setup.py test``

Documentation
=============

Go to the package directory and then to ``docs/`` and type

``pip install -r requirements.txt``

Then type the following command in order to generate documentation in HTML

``make html``

And run doctest

``make doctest``

