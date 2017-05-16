# IICREstimator
A python script for estimating the Inverse Instantaneous Coalescent Rate (IICR) based on independent values of
the coalescence times, simulated with the ms software.

This is the python script that has been used to estimate the IICRs presented in Lounès 2017 (cite the paper). In order to
run the script, a working compiled version of ms (link to ms) must be placed in the same folder as the script *estimateIICR.py*.

This script illustrates that it is possible to estimate the IICRs courves (which corresponds with the PSMC inferences) for
any model of population genetics for which one can write a ms command. Note that in the case of a *panmictic* model, the
IICR corresponds exactly to the changes in the population size. However, when the panmixia hypothesis is violated, the link
between population size changes and the IICR (that can also be estimated from full genomes using the PSMC) is not clear. 
Sometimes, as shown in Mazet et al 2016, the IICR and the population size are inversely related.

What is interested is that the notion of IICR provides a tool for characterizing a wide class of models in population genetics.
This script allows to see the IICR corresponding to any model for wich it is possible to simulate data under ms. Note that the
data used by the script are independent values of coalescence times.

For using the script
--------------------

1. Get a version of ms
2. Set the parameters by modifying the json file
3. Run the script


