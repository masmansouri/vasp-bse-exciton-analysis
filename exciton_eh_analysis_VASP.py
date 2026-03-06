#!/usr/bin/env python3
"""
exciton_eh_analysis_VASP.py

VASP-5.4 compatible tool to build exciton electron/hole densities from:
 - BSEFATBAND          -> exciton coefficients / weights
 - PARCHG.<band>.<kpt> -> band-resolved |psi|^2 on FFT grid
 - POSCAR              -> lattice + ionic positions (optional, used for auto molecule detection)

Outputs:
 - calculates centroid positions (cart, ang), separation (ang)
 - calculates integrated electron/hole probability on molecular slab
 - calculates root-mean-square (RMS) electron-hole separation
 - saves electron and hole cube files, or numpy arrays: rho_e.npy, rho_h.npy
 - basic sanity checks
 - prints out a summary

Notes:
 - Script expects PARCHG files named as standard VASP: e.g., PARCHG.0193.0001, although it falls back similar patterns as well.
 - For slabs, periodicity is applied only in x-y plane when computing min-image separation.
"""

import os, re, glob, sys
import numpy as np
from collections import OrderedDict
ANG_TO_BOHR = 1.0 / 0.52917721092


#------------------------ INPUTS ------------------------#
#--------------------------------------------------------#

BSEFATBAND = "path/to/BSEFATBAND"     
PARCHG_DIR = "path/to/PARCHGs"
POSCAR = "path/to/POSCAR"


# keep transitions until cumulative weight >= WEIGHT_CUTOFF
WEIGHT_CUTOFF = 0.9

# to adjust molecular slab limits manually (Ang)
MOL_Z_MIN = None
MOL_Z_MAX = None

# If auto-detect of molecule from POSCAR should look for topmost cluster separated by at least this gap (Ang)
CLUSTER_GAP = 2.5

#----------------------------------------------------------#
#----------------------------------------------------------#

def read_poscar(poscar):
    """
    POSCAR reader (VASP 5 format).

    Returns
    -------
    cell : (3,3) ndarray
    atoms : list of (Z, cartesian_position)
    elements : list[str]
    counts : list[int]
    frac_coords : (N,3) ndarray
    cart_coords : (N,3) ndarray
    """

    ELEMENTS = ['X',
    'H','He','Li','Be','B','C','N','O','F','Ne',
    'Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',
    'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
    'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
    'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
    'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',
    'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm',
    'Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds',
    'Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

    SYMBOL_TO_Z = {sym: i for i, sym in enumerate(ELEMENTS)}

    with open(poscar, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    if len(lines) < 8:
        raise RuntimeError("POSCAR too short or malformed.")

    # --- Lattice vectors ---
    scale = float(lines[1])
    cell = scale * np.array([
        np.fromstring(lines[2], sep=" "),
        np.fromstring(lines[3], sep=" "),
        np.fromstring(lines[4], sep=" ")
    ])

    if cell.shape != (3, 3):
        raise RuntimeError(f"Lattice vectors' shape invalid: {cell.shape}")

    # --- Elements and counts ---
    toks5 = lines[5].split()
    toks6 = lines[6].split()

    # VASP 5 format detection
    if any(not re.match(r'^-?\d+(\.\d+)?$', t) for t in toks5):
        elements = toks5
        counts = list(map(int, toks6))
        idx = 7
    else:
        raise RuntimeError("Only VASP 5 format with element symbols is supported.")

    nat = sum(counts)

    # --- Coordinate mode handling ---
    line = lines[idx].lower()

    if line.startswith("s"):   # Selective dynamics present
        idx += 1
        line = lines[idx].lower()

    is_direct = line.startswith("d")
    idx += 1

    # --- Read coordinates ---
    coords = []
    for i in range(nat):
        vec = np.fromstring(lines[idx + i], sep=" ")[:3]
        if vec.size != 3:
            raise RuntimeError("Malformed atomic coordinate line.")
        coords.append(vec)

    coords = np.array(coords)

    if coords.shape != (nat, 3):
        raise RuntimeError("Coordinate array shape mismatch.")

    # --- Convert coordinates ---
    if is_direct:
        frac = coords
        cart_coords = frac @ cell
    else:
        cart_coords = coords
        frac = cart_coords @ np.linalg.inv(cell)

    # --- Atomic number mapping ---
    atoms = []
    atom_index = 0

    for sp, n in zip(elements, counts):
        if sp not in SYMBOL_TO_Z:
            raise ValueError(f"Unknown element symbol: {sp}")

        Z = SYMBOL_TO_Z[sp]

        for _ in range(n):
            atoms.append((Z, cart_coords[atom_index]))
            atom_index += 1

    # Final consistency check
    assert atom_index == nat, "Atom counting mismatch."

    return cell, atoms, elements, counts, frac, cart_coords



def read_cell_from_parchg(parchg_path):
    """Read cell vectors from a PARCHG-style file (first lines similar to POSCAR)."""
    with open(parchg_path, 'r') as f:
        lines = [l.rstrip() for l in f.readlines()]
    # Similar layout: title, scale, 3 vectors
    scale = float(lines[1].strip())
    a1 = np.fromstring(lines[2], sep=' ')
    a2 = np.fromstring(lines[3], sep=' ')
    a3 = np.fromstring(lines[4], sep=' ')
    cell = np.vstack([a1, a2, a3]) * scale
    return cell



def find_parchg_file_for(band, k, dirpath=PARCHG_DIR):
    """Try to locate a PARCHG file for (band,k) using the configured pattern and fallbacks."""

    PARCHG_PATTERN = "PARCHG.{band:04d}.{k:04d}"

    # first try primary pattern
    fname = os.path.join(dirpath, PARCHG_PATTERN.format(band=band, k=k))
    if os.path.exists(fname):
        print("Reading {}.".format(fname))
        return fname
    
    # try simple pattern without zero padding
    fname2 = os.path.join(dirpath, f"PARCHG.{band}.{k}")
    if os.path.exists(fname2):
        print("Given path doesnot exist, now reading {}".format(fname2))
        return fname2
    
    # try fallback patterns
    FALLBACK_PATTERNS = [
        "PARCHG.{band:04d}.{k:04d}_uniform",
        "PARCHG.{band}.{k}",
        "PARCHG.{band:03d}.{k:03d}",
        "PARCHG.{band:04d}.{k:01d}"
    ]
    for pat in FALLBACK_PATTERNS:
        try:
            pf = os.path.join(dirpath, pat.format(band=band, k=k))
        except:
            continue
        if os.path.exists(pf):
            print("Given path doesnot exist, now reading from {}".format(pf))
            return pf
    
    #try scanning files in dir and match by containing band and k as tokens
    for f in os.listdir(dirpath):
        if f.startswith("PARCHG") and f.find(str(band)) != -1 and f.find(str(k)) != -1:
            print("Given path doesnot exist, now reading from {}".format(dirpath))
            return os.path.join(dirpath, f)
    # no match
    return None



def parse_parchg_file(parchg_path):
    """Read real-space density grid from a PARCHG-like file.
    Returns: rho (nx,ny,nz) numpy array, grid_shape (nx,ny,nz).
    """
    with open(parchg_path, 'r') as f:
        lines = f.readlines()
    # find the line with three integers (grid dims)
    grid_idx = None
    for i, line in enumerate(lines):
        m = re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s*$', line)
        if m:
            nx, ny, nz = map(int, m.groups())
            grid_idx = i
            break
    if grid_idx is None:
        raise RuntimeError(f"Could not find grid dims in {parchg_path}")
    #print("nx, ny, nz",nx, ny, nz)

    # the density floats start at next line
    float_lines = lines[grid_idx+1:]
    # collect floats
    floats = []
    for L in float_lines:
        for tok in L.split():
            # handle possible non-numeric tails robustly
            try:
                floats.append(float(tok))
            except:
                pass
    expected = nx * ny * nz
    #print("nx * ny * nz, len(floats) ",nx * ny * nz, len(floats))
    if len(floats) < expected:
        raise RuntimeError(f"PARCHG {parchg_path}: expected {expected} floats, found {len(floats)}")
    arr = np.array(floats[:expected], dtype=float)
    # reshape: VASP typically orders x fastest, then y, then z
    arr = arr.reshape((nx, ny, nz), order='C')
    return arr, (nx, ny, nz)




def parse_bsefatband(bsefile, exciton_index):
    """
    Parse VASP 5.4 BSEFATBAND and extract transitions for a given exciton.
    1BSE eigenvalue    E_BSE      IP-eigenvalue:    E_IP
    Kx, Ky, Kz, E_v, E_c, Abs(X_BSE)/W_k, NB_v, NB_c, Re(X_BSE)+ i*Im(X_BSE)
    Returns:
      transitions: list of dicts {v, c, k, weight}
      kmap: dict mapping k-vector -> k-index (1-based)
    """
    with open(bsefile, 'r') as f:
        lines = f.readlines()

    exciton_count = 0
    in_block = False
    raw = []
    kvecs = []

    for line in lines:
        if "BSE eigenvalue" in line:
            exciton_count += 1
            in_block = (exciton_count == exciton_index)
            continue

        if not in_block:
            continue

        nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line)
        if len(nums) < 10:
            continue

        kx, ky, kz = map(float, nums[0:3])
        A = float(nums[5])
        v = int(nums[6])
        c = int(nums[7])
        reX = float(nums[8])
        imX = float(nums[9])

        w = reX**2 + imX**2

        kvec = (round(kx, 8), round(ky, 8), round(kz, 8))
        if kvec not in kvecs:
            kvecs.append(kvec)

        raw.append({
            "v": v,
            "c": c,
            "kvec": kvec,
            "A": A,
            "weight": w
        })

    if not raw:
        raise RuntimeError("No transitions parsed")

    # normalize weights
    total = sum(r["weight"] for r in raw)
    for r in raw:
        r["weight"] /= total

    kmap = {kvecs[i]: i + 1 for i in range(len(kvecs))}

    transitions = [{
        "v": r["v"],
        "c": r["c"],
        "A": r["A"],
        "k": kmap[r["kvec"]],
        "weight": r["weight"]
    } for r in raw]

    #print(transitions)
    return transitions, kmap


def select_top_transitions(transitions, cutoff=WEIGHT_CUTOFF):
    """
    Select transitions contributing up to a fraction of the total exciton weight.
    cutoff   : cumulative fraction of total weight (Re(X_BSE)+ i*Im(X_BSE))**2 
    """

    # positive-definite weights
    weights = np.array([abs(t["weight"]) for t in transitions])
    total = weights.sum()

    if total <= 0:
        raise RuntimeError("Total exciton weight is zero.")

    # normalize
    weights /= total

    # sort by contribution
    idx = np.argsort(weights)[::-1]

    selected = []
    cum = 0.0
    for i in idx:
        selected.append({
            "A": transitions[i]["A"],
            "v": transitions[i]["v"],
            "c": transitions[i]["c"],
            "k": transitions[i]["k"],
            "weight": weights[i],
        })
        cum += weights[i]
        if cum >= cutoff:
            break

    return selected


def select_top_transitions_by_A(transitions, cutoff=WEIGHT_CUTOFF):
    """
    Select transitions contributing up to a fraction of the total exciton weight.
    cutoff   : cumulative fraction of total weight based Abs(X_BSE)/W_k values
    """

    weights = [abs(t["weight"]) for t in transitions]

    norm = max(weights)*cutoff
    print("norm", norm)
    trans_norm = []
    for t, w in zip(transitions, weights):
        if w >= norm:
            trans_norm.append({
                "v": t["v"],
                "c": t["c"],
                "k": t["k"],
                "weight": w 
            })


    # sort by normalized contribution
    trans_sorted = sorted(trans_norm, key=lambda x: x["weight"], reverse=True)
    #print("trans_sorted",trans_sorted)
    return trans_sorted



def build_rho_eh(selected_transitions, cell, parchg_dir=PARCHG_DIR):
    """Accumulate rho_e and rho_h from PARCHG files using the selected transitions.
    Returns rho_e, rho_h arrays and grid shape and voxel fractional coordinates arrays.
    """
    
    # attempt to find PARCHG to get grid shape and header cell
    found_one = False
    sample_file = None
    for t in selected_transitions:
        fpath = find_parchg_file_for(t['v'], t['k'], parchg_dir)
        if fpath and os.path.exists(fpath):
            sample_file = fpath
            found_one = True
            break
    if not found_one:
        raise RuntimeError("No PARCHG files found for any selected transitions. Check naming patterns.")
    
    rho_sample, grid_shape = parse_parchg_file(sample_file)
    nx, ny, nz = grid_shape
    rho_e = np.zeros_like(rho_sample)
    rho_h = np.zeros_like(rho_sample)
    missing = []
    
    for t in selected_transitions:
        v = t['v']; c = t['c']; k = t['k']; w = t['weight']
        fname_v = find_parchg_file_for(v, k, parchg_dir)
        fname_c = find_parchg_file_for(c, k, parchg_dir)
        if fname_v is None or fname_c is None:
            missing.append((v, c, k))
            continue
        rho_v, _ = parse_parchg_file(fname_v)
        rho_c, _ = parse_parchg_file(fname_c)
        # ensure shapes match
        if rho_v.shape != rho_sample.shape or rho_c.shape != rho_sample.shape:
            raise RuntimeError(f"Grid mismatch for files {fname_v} or {fname_c}")
        rho_h += w * rho_v
        rho_e += w * rho_c
    
    if missing:
        print("Warning: missing PARCHG files for transitions (v,c,k):", missing)
    
    # normalize to unit integral (probability)
    sum_h = rho_h.sum()
    sum_e = rho_e.sum()
    if sum_h > 0:
        rho_h = rho_h / sum_h
    if sum_e > 0:
        rho_e = rho_e / sum_e
    return rho_e, rho_h, grid_shape


def grid_centroid(rho, cell):
    """Compute centroid (cart) of rho defined on grid. Assumes grid shape (nx,ny,nz) and 3x3 array of the cell."""
    nx, ny, nz = rho.shape

    # fractional coordinates of voxel centers
    ix = (np.arange(nx) + 0.5) / nx
    iy = (np.arange(ny) + 0.5) / ny
    iz = (np.arange(nz) + 0.5) / nz
    Xf, Yf, Zf = np.meshgrid(ix, iy, iz, indexing='ij')

    # convert fractional to cart: r = f_x * a1 + f_y * a2 + f_z * a3
    # create coordinate arrays for each cartesian component
    # Flatten grid for integration
    fracs = np.stack((Xf, Yf, Zf), axis=-1).reshape((-1,3))

    # compute cart positions
    cart_coords = fracs @ cell  # (N,3)
    rho_flat = rho.ravel()
    norm = rho_flat.sum()
    if norm <= 0:
        return np.array([np.nan, np.nan, np.nan])
    r_avg = (rho_flat[:,None] * cart_coords).sum(axis=0) / norm
    return r_avg


def min_image_separation(r_e_cart, r_h_cart, cell, periodic=(True,True,False)):
    """Compute minimum-image vector r_e - r_h, applying periodic wrap only on axes flagged True."""
    # convert cart to fractional coordinates
    inv_cell = np.linalg.inv(cell.T)  # maps cart -> fractional when using row-vector * cell

    # warning: r = frac @ cell used earlier, so mapping back is frac = r @ inv_cell
    r_e_frac = r_e_cart @ inv_cell
    r_h_frac = r_h_cart @ inv_cell
    delta = r_e_frac - r_h_frac

    # apply minimum image on periodic axes only (x,y)
    for i in range(3):
        if periodic[i]:
            delta[i] -= np.round(delta[i])
    delta_cart = delta @ cell
    dist = np.linalg.norm(delta_cart)
    return delta_cart, dist


def detect_molecule_zrange_from_poscar(poscar, gap_threshold=CLUSTER_GAP):
    """read POSCAR atoms, partition by z coordinate gaps, return zmin,zmax of top adsorbate distribution."""
    try:
        cell, atoms, elements, counts, frac, cart_coords = read_poscar(poscar)
    except Exception as e:
        print("Auto-detect molecule: failed to parse POSCAR:", e)
        return None, None

    z = cart_coords[:,2]
    order = np.argsort(z)
    z_sorted = z[order]

    # find big gaps
    diffs = np.diff(z_sorted)
    if (diffs.max() >  gap_threshold):
        msg = 'The difference in layers hight {:.3f} is larger than the given gap_threshold = {}'.format(diffs.max(), gap_threshold)

    # cluster edges where diffs > gap_threshold
    split_idx = np.where(diffs > gap_threshold)[0]
    
    # clusters boundaries
    clusters = []
    start = 0
    for idx in split_idx:
        clusters.append((start, idx))
        start = idx+1
    clusters.append((start, len(z_sorted)-1))
    # choose cluster with highest mean z (adsorbate usually top)
    best_cluster = None
    best_mean = -1e9
    for (a,b) in clusters:
        zs = z_sorted[a:b+1]
        meanz = zs.mean()
        if meanz > best_mean:
            best_mean = meanz
            best_cluster = (zs.min(), zs.max())
    if best_cluster is None:
        return None, None
    zmin, zmax = best_cluster
    # expand a little margin
    margin = 0.5
    return zmin - margin, zmax + margin, msg



def integrate_on_slab(rho, cell, slab_zmin, slab_zmax):
    """Integrate rho probability inside slab defined by z (cartesian coordinates). Returns fraction [0,1]."""
    nx, ny, nz = rho.shape
    iz = (np.arange(nz) + 0.5) / nz
    # get cart z of each slice
    z_coords = (iz[:,None] * cell).sum(axis=1)  # not correct general form; do using fractional basis
    # Better compute using mapping of fractional to cart for z only
    # Build f vectors for each iz: fx=0,fy=0,fz=iz  then cart_z = f @ cell -> last component
    fzs = np.vstack([np.zeros((nz,2)), iz]).T  # shape glitch; simpler below
    # simpler: compute cart coords for center points for each iz by using frac vector [0,0,fz]
    cart_z = np.array([np.array([0.0,0.0,zz]) @ cell for zz in iz])[:,2]
    # sum over x,y for each z-slice
    slice_sum = rho.sum(axis=(0,1))
    mask = (cart_z >= slab_zmin) & (cart_z <= slab_zmax)
    frac = slice_sum[mask].sum() / slice_sum.sum()
    return frac


def ct_index_z(rho_e, rho_h, cell, z0, z_window=None):
    """
    Compute z-resolved charge-transfer index:
      CT_z = Int{z>z0} rho_e(z) dz  -  Int{z>z0} rho_h(z) dz

    Parameters
    ----------
    rho_e, rho_h : ndarray (nx, ny, nz)
        Electron and hole densities (unnormalized OK).
    cell : ndarray (3,3)
        Lattice vectors in Cartesian coordinates (Angstrom).
    z0 : float
        Dividing plane in Cartesian z (Angstrom).
    z_window : tuple (zmin, zmax), optional
        Restrict integration to a slab region (Angstrom).

    Returns
    -------
    CT_z : float
        Dimensionless charge-transfer index in [-1, 1].
    """

    assert rho_e.shape == rho_h.shape
    nx, ny, nz = rho_e.shape

    # fractional z of voxel centers
    iz = (np.arange(nz) + 0.5) / nz

    # Cartesian z coordinates of grid planes
    # assumes z is along cell[2]; valid for slab geometries
    z_cart = iz * cell[2, 2]

    # project densities onto z
    rho_e_z = rho_e.sum(axis=(0, 1))
    rho_h_z = rho_h.sum(axis=(0, 1))

    # normalize
    rho_e_z /= rho_e_z.sum()
    rho_h_z /= rho_h_z.sum()

    # apply z-window if requested
    mask = np.ones(nz, dtype=bool)
    if z_window is not None:
        zmin, zmax = z_window
        mask &= (z_cart >= zmin) & (z_cart <= zmax)

    # integrate above dividing plane
    above = mask & (z_cart > z0)

    CT_z = rho_e_z[above].sum() - rho_h_z[above].sum()
    return CT_z



def rms_eh_separation(rho_e, rho_h, cell):
    """
    Compute RMS electron at hole separation:
      d_RMS = sqrt( <|r_e - r_h|^2> )

    Uses full 3D densities.

    Parameters
    ----------
    rho_e, rho_h : ndarray (nx, ny, nz)
        Electron and hole densities (unnormalized OK).
    cell : ndarray (3,3)
        Lattice vectors in Cartesian coordinates (Angstrom).

    Returns
    -------
    d_rms : float
        RMS electron at hole separation (Angstrom).
    """

    assert rho_e.shape == rho_h.shape
    nx, ny, nz = rho_e.shape

    # fractional coordinates of voxel centers
    ix = (np.arange(nx) + 0.5) / nx
    iy = (np.arange(ny) + 0.5) / ny
    iz = (np.arange(nz) + 0.5) / nz
    Xf, Yf, Zf = np.meshgrid(ix, iy, iz, indexing='ij')

    fracs = np.stack((Xf, Yf, Zf), axis=-1).reshape((-1, 3))
    cart = fracs @ cell  # (N,3)

    rho_e_flat = rho_e.ravel()
    rho_h_flat = rho_h.ravel()

    # normalize
    rho_e_flat /= rho_e_flat.sum()
    rho_h_flat /= rho_h_flat.sum()

    # centroids
    re = (rho_e_flat[:, None] * cart).sum(axis=0)
    rh = (rho_h_flat[:, None] * cart).sum(axis=0)

    # <r^2>
    re2 = (rho_e_flat * np.sum(cart**2, axis=1)).sum()
    rh2 = (rho_h_flat * np.sum(cart**2, axis=1)).sum()

    # RMS separation
    d_rms_sq = re2 + rh2 - 2.0 * np.dot(re, rh)
    return np.sqrt(d_rms_sq)




def write_cube(filename, rho, lattice, atoms=None, origin=None):
    """
    lattice: 3x3 array in Ang
    rho: 3D density array
    atoms: list of (Z, pos) tuples
    origin: optional, default = [0,0,0]
    """

    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin, dtype=float).ravel()
        if origin.size != 3:
            raise ValueError(f"Origin must be a 3-element vector, got shape {origin.shape}")


    if atoms is None:
        atoms = []

    nx, ny, nz = rho.shape
    #print("CUBE nx, ny, nz", nx, ny, nz)

    # convert lattice to Bohr
    lattice_bohr = lattice * ANG_TO_BOHR
    origin_bohr = origin * ANG_TO_BOHR

    # voxel vectors (Bohr)
    vx = lattice_bohr[0] / nx
    vy = lattice_bohr[1] / ny
    vz = lattice_bohr[2] / nz

    with open(filename, "w") as f:
        f.write("Exciton density\n")
        f.write("Generated from VASP BSE\n")

        f.write(f"{len(atoms):5d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}\n")
        f.write(f"{nx:5d} {vx[0]:12.6f} {vx[1]:12.6f} {vx[2]:12.6f}\n")
        f.write(f"{ny:5d} {vy[0]:12.6f} {vy[1]:12.6f} {vy[2]:12.6f}\n")
        f.write(f"{nz:5d} {vz[0]:12.6f} {vz[1]:12.6f} {vz[2]:12.6f}\n")

        for Z, pos in atoms:
            pos = np.asarray(pos, dtype=float)
            pos_bohr = pos * ANG_TO_BOHR
            f.write(f"{Z:5d} 0.0 {pos_bohr[0]:12.6f} {pos_bohr[1]:12.6f} {pos_bohr[2]:12.6f}\n")

        flat = rho.flatten(order="F")
        for i in range(0, len(flat), 6):
            f.write(" ".join(f"{x:13.5e}" for x in flat[i:i+6]) + "\n")



# ---------------------- Main driver ----------------------
def main(EXCITON_INDEX):

    OUTPREFIX = "exciton"+str(EXCITON_INDEX)
    
    with open(OUTPREFIX+".out", "w") as f:
        f.write("******************************\n")
        f.write("Exciton {}\n".format(EXCITON_INDEX))
        f.write("******************************\n\n")


        if not os.path.exists(BSEFATBAND):
            f.write("Error: BSEFATBAND file not found at {}\n".format(BSEFATBAND))
            sys.exit(1)
        f.write("Parsing BSEFATBAND...\n")
        transitions_all, kmap = parse_bsefatband(BSEFATBAND, exciton_index=EXCITON_INDEX)
        f.write("Found {}!\n".format(len(transitions_all)))


        # select top transitions by weight
        selected = select_top_transitions(transitions_all, cutoff=WEIGHT_CUTOFF)
        f.write("Selected {}.\n\n".format(len(selected)))
        f.write("Dominant transitions (cumulative weight >= {:.0f}%): \n".format(WEIGHT_CUTOFF*100))
        # write top contributors
        for t in selected:
            f.write("{}-->{} at K {}, Re(BSE)+ Im(BSE) {}, and Abs(BSE)/Wk {}.\n".format(t['v'],t['c'], t['k'],t['weight'], t['A']))
        f.write("\n\n")


        # read cell from POSCAR if exists, else try a PARCHG sample
        if os.path.exists(POSCAR):
            cell, atoms, elements, counts, frac, cart_coords = read_poscar(POSCAR)
            f.write("Reading POSCAR.\n")
        else:
            # try to find any PARCHG file to read its header cell
            fallback = None
            for f in os.listdir(PARCHG_DIR):
                if f.startswith("PARCHG"):
                    fallback = os.path.join(PARCHG_DIR, f)
                    break
            if fallback is None:
                f.write("No POSCAR and no PARCHG found to read cell. Aborting.\n")
                sys.exit(1)
            cell = read_cell_from_parchg(fallback)
            f.write("Read cell from {}.\n\n".format(fallback))


        # build densities
        f.write("Building rho_e and rho_h from PARCHG files (this can be slow for many transitions)...\n")
        rho_e, rho_h, grid_shape = build_rho_eh(selected, cell, parchg_dir=PARCHG_DIR)
        nx, ny, nz = grid_shape
        if not (nx == ny == nz):
            f.write("WARNING: grid mismatch nx={nx}, ny={ny}, nz={nz} can cause error in the final cube prints.")
        f.write("rho arrays shape: {} \n\n".format(rho_e.shape))
        f.write("Sanity checks:\n")
        f.write("  rho_e integral (should be 1):{}\n".format(rho_e.sum()))
        f.write("  rho_h integral (should be 1):{}\n\n".format(rho_h.sum()))


        # compute centroids
        r_e = grid_centroid(rho_e, cell)
        r_h = grid_centroid(rho_h, cell)
        delta_cart, dist = min_image_separation(r_e, r_h, cell, periodic=(True,True,False))
        f.write("Centroid electron (cart): {}\n".format(np.array2string(r_e, precision=4)))
        f.write("Centroid hole     (cart): {}\n".format(np.array2string(r_h, precision=4)))
        f.write("Electron-hole separation (Ang, min-image xy): {}\n\n".format(round(dist,4)))

        # expected scales:
        approx_exciton_radius = dist
        max_allowed = max(np.linalg.norm(cell[0]), np.linalg.norm(cell[1])) * 0.9
        if approx_exciton_radius > max_allowed:
            f.write("  Warning: computed separation is large (>{:.1f} Ang). Check PARCHG grid, cell, or minimum-image handling.".format(max_allowed),"\n")


        # molecule/surface partition
        global MOL_Z_MIN, MOL_Z_MAX
        if MOL_Z_MIN is None or MOL_Z_MAX is None:
            if os.path.exists(POSCAR):
                zmin, zmax, msg = detect_molecule_zrange_from_poscar(POSCAR)
                if zmin is None:
                    f.write("Auto-detect of molecule z-range failed. Please set MOL_Z_MIN and MOL_Z_MAX manually in script.\n")
                    sys.exit(1)
                if msg:
                    f.write(msg)
                MOL_Z_MIN = zmin; MOL_Z_MAX = zmax
                f.write("\nAuto-detected molecule/surface z-range (Ang):{:.4f}, {:.4f}\n".format(MOL_Z_MIN, MOL_Z_MAX))
            else:
                f.write("\nNo POSCAR found and MOL_Z_MIN/MOL_Z_MAX not set. Please set them in the script.\n")
                sys.exit(1)


        # compute CT fractions by integrating slices in z (approx)
        # compute cart z for slice centers
        nx, ny, nz = grid_shape
        iz = (np.arange(nz) + 0.5) / nz
        cart_z = np.array([np.array([0.0,0.0,zz]) @ cell for zz in iz])[:,2]
        slice_e = rho_e.sum(axis=(0,1))
        slice_h = rho_h.sum(axis=(0,1))
        q_e_mol = slice_e[(cart_z >= MOL_Z_MIN) & (cart_z <= MOL_Z_MAX)].sum()
        q_h_mol = slice_h[(cart_z >= MOL_Z_MIN) & (cart_z <= MOL_Z_MAX)].sum()
        f.write("\nIntegrated electron prob on molecular slab: {:.4f}\n".format(q_e_mol))
        f.write("Integrated hole prob on molecular slab:     {:.4f}\n".format(q_h_mol))
        ct_ind = q_e_mol * (1-q_h_mol) + (1-q_e_mol) * q_h_mol
        f.write("CT index (Zero for purely localized excitons, maximum for perfect CT:     {:.4f}\n".format(ct_ind))
        z0 = MOL_Z_MIN - 1 #safe dividing plane between surface & molecule
        ct_ind_z = ct_index_z(rho_e, rho_h, cell, z0, z_window=None)
        f.write("Z-resolved CT index for z0={:.4f}: {}\n".format(z0 , ct_ind_z))
        rms_eh = rms_eh_separation(rho_e, rho_h, cell)
        f.write("RMS electron-hole separation: {}\n\n".format(rms_eh))


        # save arrays
        #np.save(f"{OUTPREFIX}_rho_e.npy", rho_e)
        #np.save(f"{OUTPREFIX}_rho_h.npy", rho_h)
        #f.write("Saved {}.\n".format(OUTPREFIX + "_rho_e.npy and _rho_h.npy"))


        # save cubes
        #rho_e = np.load("excitonS_rho_e.npy")
        #rho_h = np.load("excitonS_rho_h.npy")
        cell, atoms, elements, counts, frac, cart_coords = read_poscar(POSCAR)
        write_cube(OUTPREFIX +"_electron.cube", rho_e, cell, atoms)
        write_cube(OUTPREFIX +"_hole.cube", rho_h, cell, atoms)
        f.write("Cube files of e h saved!\n\n")


        # summary
        f.write("Summary:")
        f.write(f"  \n Exciton {EXCITON_INDEX}: electron@molecule={q_e_mol:.3f}, hole@molecule={q_h_mol:.3f}")
        f.write(f"  \n Exciton {EXCITON_INDEX}: centroid separation = {dist:.3f} Ang, RMS electron-hole separation = {rms_eh:.3f} Ang.")
        f.write("\n Done!")
        f.close()



if __name__ == "__main__":
    for i in range(1,10):
        EXCITON_INDEX = i
        main(EXCITON_INDEX)
