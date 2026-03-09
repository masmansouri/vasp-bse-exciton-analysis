# VASP GW/BSE Exciton Analysis Tools

Python tools for post-processing excitonic properties obtained from
GW/BSE calculations in VASP.

The code reconstructs real-space electron and hole densities from
BSE solutions and computes the following quantitative descriptors in hybrid molecule/2D interfaces.

- electron and hole density reconstruction
- exciton centroid calculation
- electron–hole separation
- RMS exciton size
- molecular localization probabilities


## Scientific background

The electron and hole densities are defined as $\rho_{e/h}(\mathbf{r}) = \sum_{v c} \left| w_{vc} \right|^2 \left| \psi_{c/v}(\mathbf{r}) \right|^2$, 
where $w_{vc}$ are  the normalized weight of the transition $v\rightarrow c$ and $\psi_{c/v}(\mathbf{r})$ are eigenfunctions.
From these densities, the code computes

- **electron and hole centroids**, representing the average spatial positions of the electron and hole

$$
\langle \mathbf{r}_{e/h} \rangle =
\int \mathbf{r} \rho_{e/h}(\mathbf{r}) d\mathbf{r}
$$

- **centroid separation**, measuring the spatial separation between the electron and hole centroids

$$
\langle r_{eh} \rangle =
\left|
\langle \mathbf{r}_e \rangle -
\langle \mathbf{r}_h \rangle
\right|
$$


- **RMS electron–hole distance** as a descriptor of electron–hole overlap and spatial delocalization, defined as the root-mean-square separation

$$
d_{\mathrm{RMS}} =
\sqrt{
\left\langle r_e^2 \right\rangle +
\left\langle r_h^2 \right\rangle -
2 \langle \mathbf{r}_e \rangle \cdot \langle \mathbf{r}_h \rangle
}, \mathrm{with} \left\langle r_{e/h}^2 \right\rangle =
\int r^2 \rho_{e/h}(\mathbf{r})\, d\mathbf{r}.
$$


- **Molecular localization probability**, describing the probability of finding the electron or hole on the molecular adsorbate

$$
P_{e/h}^{\mathrm{mol}} =
\int_{\Omega_{\mathrm{mol}}}
\rho_{e/h}(\mathbf{r}) d\mathbf{r}.
$$

These quantities allow classification of excitons into localized/charge transfer excitons
and help rationalize substrate-induced screening effects. Particularly, Small $d_{RMS}$ indicates 
strongly overlapping electron and hole densities (localized excitons), whereas a large 
value corresponds to spatially separated charge-transfer excitons.



## Requirements

Python, numpy  

## Installation
git clone https://github.com/masmansouri/vasp-bse-exciton-analysis.git

## Usage
The script requires the following input files from a GW/BSE calculation in VASP:

- **POSCAR** – atomic structure used to identify the substrate and molecular region  
- **BSEFATBAND** – contains the BSE eigenvectors (transition weights) for each exciton  
- **PARCHG files** – band- and k-resolved charge densities contributing to the exciton transitions  

The code reconstructs real-space **electron and hole densities** associated with a selected exciton.

#### Running the code

Specify the exciton of interest through `EXCITON_INDEX`.  
The script automatically selects the dominant transitions until a cumulative transition weight reaches a cutoff value (`WEIGHT_CUTOFF`, default = `0.9`).

Run:

```python exciton_eh_analysis.py```

A loop over multiple `EXCITON_INDEX` values can also be used to analyze several excitons.



For each exciton analyzed, the code produces:

- Summary file containing numerical descriptors and sanity checks (exciton_index.out). 
- Electron and hole density cube files, to be visualized by tools like VESTA or VMD.

#### Notes:

The code attempts to automatically determine the molecular region along the z-direction by identifying the topmost atoms relative to the surface.
This detection can be adjusted with:
- `CLUSTER_GAP`as a vertical separation threshold (Ang) between surface and adsorbate.
- `MOL_Z_MIN`, `MOL_Z_MAX` as manual limits defining the molecular region.

If the PARCHG grids are not uniform, it is recommended to first generate FFT-based uniform grids.
The helper script `python uniform_PARCHARGE.py` resamples files of the form
`PARCHG.{iband:04d}.{kpoint:04d}`/`PARCHG.{iband:04d}.ALLK` to ensure consistent density reconstruction.

## Citation
If you use this code in scientific work, please cite it as: https://doi.org/10.5281/zenodo.18889149
