import numpy as np

# ---- FFT-based resampling ----
def fft_resample(data, new_shape):
        fft_data = np.fft.fftn(data)
        fft_data = np.fft.fftshift(fft_data)

        old = np.array(data.shape)
        new = np.array(new_shape)

        out = np.zeros(new, dtype=complex)
        keep = np.minimum(old, new)

        o0 = (old - keep) // 2
        n0 = (new - keep) // 2

        out[
            n0[0]:n0[0]+keep[0],
            n0[1]:n0[1]+keep[1],
            n0[2]:n0[2]+keep[2],
        ] = fft_data[
            o0[0]:o0[0]+keep[0],
            o0[1]:o0[1]+keep[1],
            o0[2]:o0[2]+keep[2],
        ]

        out = np.fft.ifftshift(out)
        return np.real(np.fft.ifftn(out)) * np.prod(new) / np.prod(old)
        
        
# ==========================
# USER INPUT
# ==========================
mas=[228]
for iband in range(193,243):
    input_file  = "PARCHG.0{}.ALLK".format(iband)
    output_file = "PARCHG.0{}_uniform".format(iband)
    print(input_file,'-->', output_file)
    new_grid = (200, 200, 200)   # (nx, ny, nz)



    with open(input_file, "r") as f:
        lines = f.readlines()

    # ---- Parse POSCAR-like header ----
    header = []
    i = 0

    header.append(lines[i]); i += 1          # comment
    header.append(lines[i]); i += 1          # scaling

    # lattice vectors
    for _ in range(3):
        header.append(lines[i])
        i += 1

    # atom symbols or counts
    tokens = lines[i].split()
    if all(t.isalpha() for t in tokens):
        header.append(lines[i])
        i += 1

    # atom counts
    natoms = list(map(int, lines[i].split()))
    natom_total = sum(natoms)
    header.append(lines[i])
    i += 1

    # selective dynamics (optional)
    if lines[i].strip().lower().startswith("s"):
        header.append(lines[i])
        i += 1

    # coordinate system
    header.append(lines[i])
    i += 1

    # atomic positions
    for _ in range(natom_total):
        header.append(lines[i])
        i += 1

    # ---- Grid line (this is the key fix) ----
    # ---- Find grid line robustly ----
    while True:
        if i >= len(lines):
            raise RuntimeError("Failed to locate grid line in PARCHG")

        tokens = lines[i].split()
        if len(tokens) == 3:
            try:
                nx, ny, nz = map(int, tokens)
                break
            except ValueError:
                pass
        i += 1

    print(f"Original grid: {nx} {ny} {nz}")
    i += 1


    # ---- Read charge density ----
    rho_vals = []
    for line in lines[i:]:
        rho_vals.extend(map(float, line.split()))

    rho = np.array(rho_vals).reshape((nx, ny, nz), order="F")



    rho_new = fft_resample(rho, new_grid)

    # ---- Write output ----
    with open(output_file, "w") as f:
        for line in header:
            f.write(line)
        f.write(f"{new_grid[0]} {new_grid[1]} {new_grid[2]}\n")

        flat = rho_new.flatten(order="F")
        for j in range(0, len(flat), 5):
            f.write(" ".join(f"{x:18.11E}" for x in flat[j:j+5]) + "\n")

    # ---- Charge check ----
    q_old = rho.sum() / rho.size
    q_new = rho_new.sum() / rho_new.size

    print(f"Integrated charge (old): {q_old:.6e}")
    print(f"Integrated charge (new): {q_new:.6e}")
    print(f"Difference            : {q_new - q_old:.2e}")
