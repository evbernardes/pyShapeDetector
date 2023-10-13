# RANSAC

This submodule implements different RANSAC-based routines

## Abstract classes:
 - `RANSAC_BASE`: Base class implementing logic of RANSAC based method, from which they all inherit.
 - `RANSAC_WeightedBase`: Inherits from `RANSAC_BASE` and implements weight-related functionality.

 ## Available methods

 - `RANSAC_Classic`: Classic RANSAC method, decide best fit according to number of inliers.
 - `RANSAC_Weighted`: Uses number of inliers as weight.
 - `MSAC`: Type of weighed RANSAC.
 - `BDSAC`: Type of weighed RANSAC.
 - `LDSAC`: Type of weighed RANSAC.