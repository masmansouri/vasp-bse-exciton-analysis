# VASP GW/BSE Exciton Analysis Tools

Python tools for post-processing excitonic properties obtained from
GW/BSE calculations in VASP.

The code reconstructs real-space electron and hole densities from
BSE eigenvectors and Kohn–Sham wavefunctions and extracts quantitative
descriptors of exciton localization and charge transfer character.

## Features

- electron and hole density reconstruction
- exciton centroid calculation
- electron–hole separation
- RMS exciton size
- molecular localization probabilities

## Scientific background

The electron and hole densities are defined as

ρ_e/h(r) = Σ_vc |w_vc|² |ψ_c/v(r)|²

where w_vc are BSE eigenvector weights and ψ are HSE eigenfunctions.

From these densities the code computes

- electron and hole centroids
- centroid separation
- RMS electron–hole distance
- molecular localization probabilities

These quantities allow classification of excitons into localized/charge transfer excitons
and help rationalize substrate-induced screening effects.

## Requirements

Python 
numpy  

## Installation
Clone the repository
git clone https://github.com/masmansouri/vasp-bse-exciton-analysis.git

## Usage

Example:

python exciton_eh_analysis.py \
    --cube electron.cube \
    --cube-hole hole.cube \
    --poscar POSCAR

The code will output

electron centroid  
hole centroid  
centroid separation  
RMS exciton size  
molecular localization probabilities

## Citation

If you use this code in scientific work please cite

Masoud Mansouri, *vasp-bse-exciton-analysis* (GitHub repository)

DOI will be provided via Zenodo.

## License

MIT License
