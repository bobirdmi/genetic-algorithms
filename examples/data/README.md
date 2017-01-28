# Knapsack problem
The archives `inst.zip` and `sol.zip` contain input knapsack problem instances and their optimal solutions.

## Input data
Sequence of input instances have the following format:
`ID |	n |	M |	weight | cost |	weight | cost	…	…`

where `|` was added just for readability and 
* `ID` is unique instance identifier
* `n` is an instance size (amount of items)
* `M` is a knapsack capacity
* `weight` is weight of some item
* `cost` is an item cost

## Optimal solutions
Solutions have the following format:
`ID |	n |	solution cost |	0/1 |	0/1	…`

Order of `0/1` (item wasn't added/was added) is the same as order of items in input instances.

Take into account that the amount of optimal solutions may be greater than 1, though there is presented only one.
