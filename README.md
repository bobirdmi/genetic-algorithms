# genetic-algorithms
Implementation of standard, migration and diffusion models of genetic algorithms (GA) in `python 3.5`.

Benchmarking was conducted by [COCO platform](http://coco.gforge.inria.fr/) `v15.03`.

## Content description
* **/2.7/** contains files converted from `python 3.5` to `python 2.7` as [COCO platform](http://coco.gforge.inria.fr/) used in benchmarking works only with this version of python.
* **/2.7/benchmark/** contains the following files:
  * `my_experiment.py` is used for running benchmarking. Read more [here](http://coco.lri.fr/COCOdoc/runningexp.html#python).
  * `my_timing.py` is used for time complexity measurements. It has the same run conditions as the previous file.
  * `pproc.py` is a modified file from COCO platform distribution that must be copied to `bbob.v15.03/python/bbob_pproc/` in order to post-process measured data of migration GA (other algorithms don't need it). It is necessary due to unexpected format of records in case of migration GA.
* **/benchmarking/** contains benchmarking results and plots.
* **/time_complexity/** contains time results measured using `my_timing.py`.
* **/examples/** contains examples of using the implemented genetic algorithms.
