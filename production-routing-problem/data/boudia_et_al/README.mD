## Format explanations

The instances are randomly generated in the euclidien plan (100*100).
The depot is located in the origin position (0,0).

The file instances of each folder represents :

#### Column 1 : The customer number
#### column 2 : The X value of the each customer location
#### column 3 : The y value of the each customer location

The cost Cij of travelling from i to j is computed using the euclidien distance 
arounded to the nearest integer value and multiplied by the unitarian distance cost.


The demands of each customer at each period are represented by column 4 to column 23. 

The holding cost = 1 unit/period/unit of product


## A remark regarding the benchmark set of Boudia et al. added by Yossiri Adulyasak in 2012.*

According to the description given in Boudia et al.(2007), production in period t becomes only available in period t+1, 
but no inventory costs are incurred before the production becomes available. As a consequence, the demand in period 1 
must be satisfied from the initial inventory at the plant. In the experiments of  Boudia et al.(2007) and Boudia et al.(2009), 
the initial inventory at the plant is set equal to the total customer demand in the first period and no inventory holding cost 
is incurred for the initial inventory between period 0 and 1 (Prins 2012).

[References:]
Boudia, M., M. A. O. Louly, C. Prins. 2007. A reactive GRASP and path relinking for a combined production distribution
problem. Comput. Oper. Res. 34. 3402-3419.

Boudia, M., C. Prins. 2009. A memetic algorithm with dynamic population management for an integrated
production-distribution problem. Eur. J. Oper. Res. 195. 703-715.

Prins, C. 2012. Private communication.
