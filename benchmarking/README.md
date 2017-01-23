# Benchmarking by [COCO BBOB](http://coco.gforge.inria.fr/)
This folder contains results of benchmarking of standard genetic algorithm (GA), diffusion GA and migration GA.

## Content description

The folders **standard_ga**, **diffusion_ga** and **migration_ga** contains experiment results of the appropriate algorithms.

**ppdata** contains result plots of the conducted benchmarking: 
* for each algorithm itself
* and comparison between the implemented algorithms (standard, diffusion, migration)

The files **ppdata/templateBBOBmany.html** for comparison of algorithms and **ppdata/*_ga/templateBBOBarticle.html** for each algorithm itself contain summary of plots with descriptions.

## Settings

The experiment was conducted by *COCO BBOB v15.03*, namely, by template [exampleexperiment.py](http://coco.lri.fr/COCOdoc/firsttime.html#running-experiments).

The following parameters of experiment were used:
* maxfunevals = 50 * dim
* minfunevals = dim + 2
* maxrestarts = 10
* tested dimensions: 2, 3, 5, 10, 20, 40

The following parameters of genetic algorithms were used:
* single-point mutation
* mutation probability: 5%
* single-point crossover
* crossover probability: 95%
* elitism is on
* selection type: *rank*

A migration model of GA has a little more parameters:
* total amount of populations: 2
* size of populations: `given_size / 2`
* performing migration every five generations
* amount of migrants: 3
* the migrants are cloned so they still remain in the original population
* the arrived migrants replace the three worst individuals of a current population
