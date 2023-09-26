import json
import logging
import pathlib
import struct
from typing import List, Tuple

import cloudpickle
import ipdb
import numpy as np
import pandas as pd
from rich.progress import track

from grit_dset.planets_data import PlanetsData

log = logging.getLogger(__file__)


def _mat_keys(prefix: str) -> List[str]:
    out = []
    for i in range(1, 4):
        for j in range(1, 4):
            out.append(f"{prefix}{i}{j}")
    return out


def _vec_key(prefix: str) -> List[str]:
    return [f"{prefix}{i}" for i in range(1, 4)]


def _cat_keys(_df: pd.DataFrame, keys: List[str]) -> np.ndarray:
    arrs = [_df[key] for key in keys]

    return np.stack(arrs, axis=-1)


def _check_shapes(pos, vel, rot, lin_mom, ang_mom) -> None:
    T = pos.shape[0]

    assert pos.shape == (T, 3)
    assert vel.shape == (T, 3)
    assert rot.shape == (T, 3, 3)
    assert lin_mom.shape == (T, 3)
    assert ang_mom.shape == (T, 3)


def read_fwf(path: pathlib.Path) -> pd.DataFrame:
    assert path.exists() and path.is_file()

    width = 30
    rows = []

    with open(path, "r") as f:
        # Remove newline.
        headers = next(f).rstrip("\n")

        len_headers = len(headers)
        assert len_headers % width == 0
        n_headers = len_headers // width

        fmtstring = " ".join("{}s".format(width) for _ in range(n_headers))
        fieldstruct = struct.Struct(fmtstring)

        # Get header names.
        header_names = [s.decode().rstrip() for s in fieldstruct.unpack_from(headers.encode())]

        def parse_floats(_line: str) -> Tuple[float, ...]:
            return tuple(float(s.decode().rstrip()) for s in fieldstruct.unpack_from(_line.rstrip("\n").encode()))

        for line in f:
            row = np.array(parse_floats(line), dtype=np.float64)
            rows.append(row)

    # Discard the last row because the time may be fked up.
    del rows[-1]

    data_arr = np.stack(rows, axis=1)
    n_rows = len(rows)

    assert data_arr.shape == (n_headers, n_rows)

    data_dict = {header_names[ii]: data_arr[ii] for ii in range(n_headers)}

    return pd.DataFrame.from_dict(data_dict)


def _sanity_check_rotations(rot: np.ndarray) -> None:
    # Make sure that the rotation matrices are valid rotations.

    # 1: All entries should be in [-1, 1].
    max_val = np.abs(rot).max()
    assert np.all(max_val <= 1.0)

    # 2: All determinants should be 1.
    dets = np.linalg.det(rot)
    assert np.all(np.isclose(dets, 1.0))


def parse_grit(path: pathlib.Path) -> PlanetsData:
    assert path.exists()
    assert path.is_dir()
    mat_dir = path / "data_in_mat"
    assert mat_dir.exists()

    # Read in the current_system.json for the true mass and inertia for each planet.
    current_system_path = path / "current_system.json"
    assert current_system_path.exists()
    with open(current_system_path, "r") as f:
        data = json.load(f)

    bodies = data["body"]
    dt = data["parameters"]["step_size"]
    planets = []
    masses = []
    inertias = []
    for ii, body in enumerate(bodies):
        body_name = body["name"]
        body_mass = body["mass"]
        body_inertia = np.array(body["inertia_tensor"], dtype=np.float64)

        planets.append(body_name)
        masses.append(body_mass)
        inertias.append(body_inertia)

    masses = np.array(masses)
    inertias = np.stack(inertias, axis=0)
    assert inertias.shape == (len(planets), 3)

    keys = ["time", "x", "y", "z", "u", "v", "w", *_mat_keys("r"), *_vec_key("p"), *_vec_key("pi")]

    pos_list, vel_list, rot_list, linmom_list, angmom_list = [], [], [], [], []
    df = None

    for planet in planets:
        df = read_fwf(mat_dir / planet)

        # Check required keys.
        for key in keys:
            assert key in df

        pos = _cat_keys(df, ["x", "y", "z"])
        vel = _cat_keys(df, ["u", "v", "w"])
        rot = _cat_keys(df, _mat_keys("r"))
        lin_mom = _cat_keys(df, _vec_key("p"))
        ang_mom = _cat_keys(df, _vec_key("pi"))

        T = pos.shape[0]
        rot = rot.reshape((T, 3, 3))

        _check_shapes(pos, vel, rot, lin_mom, ang_mom)

        pos_list.append(pos)
        vel_list.append(vel)
        rot_list.append(rot)
        linmom_list.append(lin_mom)
        angmom_list.append(ang_mom)

    pos = np.stack(pos_list, axis=1)
    vel = np.stack(vel_list, axis=1)
    rot = np.stack(rot_list, axis=1)
    linmom = np.stack(linmom_list, axis=1)
    angmom = np.stack(angmom_list, axis=1)

    _sanity_check_rotations(rot)

    return PlanetsData(planets, dt, pos, vel, rot, linmom, angmom, masses, inertias)


def parse_grit_raw_dset(path: pathlib.Path) -> List[PlanetsData]:
    assert path.exists()
    assert path.is_dir()
    assert (path / "000").exists()

    run_dirs = sorted(list(path.glob("*/")))

    runs = [run for run in run_dirs if run.is_dir()]
    out = []
    for run in track(runs):
        pkl_path = path / f"{run.name}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                parsed = cloudpickle.load(f)
        else:
            parsed = parse_grit(run)

            with open(pkl_path, "wb") as f:
                cloudpickle.dump(parsed, f)
        out.append(parsed)

    # Make sure the dt and T are the same for all runs.
    names, dt, n_steps = out[0].names, out[0].dt, out[0].n_steps

    assert [data.names == names and data.dt == dt and data.n_steps == n_steps for data in out]

    return out


def main():
    path = pathlib.Path("/grit_data/newtonian_sun_earth_moon/data_in_mat")
    parse_grit(path)


if __name__ == "__main__":
    main()
