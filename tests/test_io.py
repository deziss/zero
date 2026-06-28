import numpy as np
import h5py
import tempfile
import os
from src.core.io import write_xyz, write_hdf5, read_xyz


def test_xyz_round_trip():
    n = 10
    positions = np.random.randn(n, 3).astype(np.float64)
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w") as f:
        tmp = f.name
    try:
        write_xyz(tmp, positions)
        read_back = read_xyz(tmp)
        assert positions.shape == read_back.shape, "Shape mismatch"
        assert np.allclose(positions, read_back, atol=1e-8), "Position mismatch"
    finally:
        os.unlink(tmp)


def test_xyz_with_box_length():
    n = 5
    positions = np.random.rand(n, 3).astype(np.float64)
    box_length = 10.0
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w") as f:
        tmp = f.name
    try:
        write_xyz(tmp, positions, box_length=box_length, comment="Test frame")
        read_back = read_xyz(tmp)
        assert np.allclose(positions, read_back, atol=1e-8)
    finally:
        os.unlink(tmp)


def test_hdf5_write_and_read():
    n = 20
    positions = np.random.randn(n, 3).astype(np.float64)
    velocities = np.random.randn(n, 3).astype(np.float64)
    box_length = 12.0
    metadata = {"timestep": 100, "temperature": 1.0}
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        tmp = f.name
    try:
        write_hdf5(tmp, positions, velocities, box_length, metadata)
        with h5py.File(tmp, "r") as h5:
            assert np.allclose(h5["positions"][:], positions, atol=1e-8)
            assert np.allclose(h5["velocities"][:], velocities, atol=1e-8)
            assert h5.attrs["box_length"] == box_length
            assert h5.attrs["timestep"] == 100
            assert abs(h5.attrs["temperature"] - 1.0) < 1e-6
    finally:
        os.unlink(tmp)


def test_hdf5_positions_only():
    n = 10
    positions = np.random.randn(n, 3).astype(np.float64)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        tmp = f.name
    try:
        write_hdf5(tmp, positions)
        with h5py.File(tmp, "r") as h5:
            assert np.allclose(h5["positions"][:], positions, atol=1e-8)
            assert "velocities" not in h5
    finally:
        os.unlink(tmp)
