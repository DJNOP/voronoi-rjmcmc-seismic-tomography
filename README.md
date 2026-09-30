# Voronoi RJMCMC Seismic Tomography

Transdimensional seismic tomography using reversible-jump Markov Chain Monte Carlo (RJMCMC) with a Voronoi-cell parameterization.

This project was developed for **Advanced Methods in Applied Statistics** at the Niels Bohr Institute, University of Copenhagen.

## Overview

The project investigates transdimensional Bayesian inversion of seismic travel-time data. Instead of using a fixed spatial discretization, the subsurface is represented by Voronoi cells whose number, locations, and seismic slowness values can change during sampling.

The method is tested in two settings:

1. **Synthetic surrogate experiment**  
   A known velocity/slowness model containing several anomalies is used to generate synthetic travel-time observations. RJMCMC is then used to reconstruct the underlying structure.

2. **Denmark seismic tomography**  
   The same approach is applied to interstation travel-time observations derived from the Danish Seismological Network.

The Denmark analysis uses multiple independent RJMCMC chains and combines retained posterior samples to estimate posterior mean seismic velocity, uncertainty, and the posterior distribution of model complexity.

## Method

The inversion uses:

- Reversible-jump Markov Chain Monte Carlo
- Voronoi-cell spatial parameterization
- Birth, death, nucleus-move, and slowness-update proposals
- Bayesian likelihood-based inference
- Multiple independent chains
- Burn-in removal and posterior sample combination
- Posterior mean velocity and uncertainty estimation

## Repository structure

### Main analysis

- `denmark_get_data_clean_fixed.py`  
  Acquisition and preprocessing of Danish seismic observations.

- `denmark_voronoi_rjmcmc_clean.py`  
  Main Voronoi RJMCMC seismic tomography implementation.

- `run_many_chains_until_stop_updated.py`  
  Execution of multiple independent RJMCMC chains.

- `combine_chain_posteriors_updated_v12.py`  
  Combination and analysis of posterior samples from multiple chains.

- `make_column_figures_v5.py`  
  Generation of final publication-style figures.

### Figures

`paper_figures/` contains selected figures used for presenting the Denmark inversion results.

## Data and large outputs

Raw seismic waveform data, cached traces, and full RJMCMC chain outputs are not included in this repository because of their size.

The repository therefore contains the analysis code and selected derived figures rather than the complete numerical output of every chain.

## Authors

- Pelle Rechnagel
- Søren Gundesen
- Nicholas Posborg
- Sebastian Hjaltelin

University of Copenhagen  
Niels Bohr Institute  
2026

## References

The project methodology was primarily inspired by:

T. Bodin and M. Sambridge,  
**Seismic tomography with the reversible jump algorithm**,  
Geophysical Journal International 178, 1411–1436 (2009).
