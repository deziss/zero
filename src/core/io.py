import h5py
import numpy as np


def write_xyz(filepath, positions, box_length=None, comment=""):
    n = positions.shape[0]
    with open(filepath, "w") as f:
        f.write(f"{n}\n")
        if box_length is not None:
            f.write(f"Lattice=\"{box_length} 0.0 0.0 0.0 {box_length} 0.0 0.0 0.0 {box_length}\" {comment}\n")
        else:
            f.write(f"{comment}\n")
        for pos in positions:
            f.write(f"Ar {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")


def write_hdf5(filepath, positions, velocities=None, box_length=None, metadata=None):
    with h5py.File(filepath, "w") as f:
        f.create_dataset("positions", data=positions, compression="gzip")
        if velocities is not None:
            f.create_dataset("velocities", data=velocities, compression="gzip")
        if box_length is not None:
            f.attrs["box_length"] = box_length
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


def read_xyz(filepath):
    with open(filepath, "r") as f:
        n = int(f.readline().strip())
        f.readline()
        positions = np.empty((n, 3))
        for i in range(n):
            parts = f.readline().split()
            positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return positions
