import pathlib
from typing import Dict, List, Optional, Tuple

import cloudpickle
import numpy as np
from attr import define, s
from rich.progress import Progress, track

from grit_dset.parse_grit import parse_grit_raw_dset
from grit_dset.planets_data import PlanetsData, log
from utils.jax_utils import register_dataclass
from utils.types import ja


@s(auto_attribs=True, slots=True, auto_exc=True, auto_detect=True, order=False)
class PlanetsDSet:
    names: List[str]
    dt: float
    # (batch, T + 1, n_planets, 3)
    q: ja
    # (batch, T + 1, n_planets, 3)
    p: ja
    # (batch, T + 1, n_planets, 3, 3)
    R: ja
    # (batch, T + 1, n_planets, 3)
    Pi: ja
    # How many batches each unique traj contributes.
    n_batch_per_traj: int

    @property
    def n_bodies(self) -> int:
        return len(self.names)

    @property
    def batch(self) -> int:
        return self.q.shape[0]

    @property
    def n_times(self) -> int:
        return self.q.shape[1]

    @property
    def n_unique_trajs(self) -> int:
        return self.batch // self.n_batch_per_traj

    @property
    def T(self) -> int:
        return self.n_times - 1


register_dataclass(PlanetsDSet)


@define
class SerializedDSetPair:
    train_dset: PlanetsDSet
    val_dset: PlanetsDSet
    dt: float
    scaling: int


class PlanetsDSetCol:
    def __init__(self, runs_path: pathlib.Path):
        self.runs_path = runs_path

        self._info = None

    def parse_grit_raw_dset(self) -> List[PlanetsData]:
        data = parse_grit_raw_dset(self.runs_path)

        self._info = dict(n_data=len(data), names=data[0].names, dt=data[0].dt, n_steps=data[0].n_steps)

        return data

    def _populate_info(self) -> None:
        if self._info is None:
            self.parse_grit_raw_dset()

    @property
    def n_data(self) -> int:
        self._populate_info()
        return self._info["n_data"]

    @property
    def names(self) -> List[str]:
        self._populate_info()
        return self._info["names"]

    @property
    def dt(self) -> float:
        self._populate_info()
        return self._info["dt"]

    @property
    def n_steps(self) -> int:
        self._populate_info()
        return self._info["n_steps"]

    def split_datas(
        self, T: int, scalings: List[int], train_frac: float = 0.8, max_trajs: int = 4096, force: bool = False
    ) -> None:
        log.info("Reading data from GRIT data...")
        datas = self.parse_grit_raw_dset()

        # Use hierarchical scheme for windowing data.
        assert len(datas) > 0

        n_datas = len(datas)
        n_train = int(train_frac * n_datas)

        orig_dt = datas[0].dt

        with Progress() as progress:
            task1 = progress.add_task("", total=len(scalings))
            task2 = progress.add_task("", total=len(datas))

            for ii, scaling in enumerate(scalings):
                progress.update(task1, description="Splitting data for scaling={}".format(scaling), completed=ii)

                assert isinstance(scaling, int)
                dt = scaling * orig_dt

                # If we've already processed this scaling then skip it.
                if not force and self.has_scaling(scaling):
                    continue

                dsets = []

                # Window all the data for this scaling.
                for jj, data in enumerate(datas):
                    progress.update(
                        task2,
                        description="Windowing data... orig_dt={:.5f}, dt={:.5f}".format(orig_dt, dt),
                        completed=jj,
                    )

                    # (nt, n_planets, *)
                    arrs = dict(q=data.pos, p=data.lin_mom, R=data.rot, Pi=data.ang_mom)

                    arrs = _subsample_data(arrs, scaling)
                    arrs = {key: _window_data(arr, T, max_trajs=max_trajs) for key, arr in arrs.items()}

                    n_batch = arrs["q"].shape[0]
                    dsets.append(PlanetsDSet(data.names, dt, **arrs, n_batch_per_traj=n_batch))

                # Split into train and val.
                train_dsets = dsets[:n_train]
                val_dsets = dsets[n_train:]
                train_dsets = merge_dsets(train_dsets)
                val_dsets = merge_dsets(val_dsets)

                # Save
                self.add_dset_pair(train_dsets, val_dsets, dt, scaling)
                del train_dsets, val_dsets, dsets

    def has_scaling(self, scaling: int) -> bool:
        path = self.runs_path / "dset_scale={:03}.pkl".format(scaling)

        return path.exists()

    def add_dset_pair(self, train_dset: PlanetsDSet, val_dset: PlanetsDSet, dt: float, scaling: int) -> None:
        # Save the pair as a pickle.
        path = self.runs_path / "dset_scale={:03}.pkl".format(scaling)

        serialize_tuple = SerializedDSetPair(train_dset, val_dset, dt, scaling)

        with open(path, "wb") as f:
            cloudpickle.dump(serialize_tuple, f)
        log.info("Saved dset pair to {}. batch={}".format(path, train_dset.batch))

    def load_scaling(self, scaling: int) -> Tuple[PlanetsDSet, PlanetsDSet, float, int]:
        path = self.runs_path / "dset_scale={:03}.pkl".format(scaling)

        if not path.exists():
            raise RuntimeError("Path {} doesn't exist!".format(path))

        with open(path, "rb") as f:
            ser = cloudpickle.load(f)
            assert isinstance(ser, SerializedDSetPair)

        return ser.train_dset, ser.val_dset, ser.dt, ser.scaling


def _window_data(arr: np.ndarray, T: int, max_trajs: Optional[int] = None) -> np.ndarray:
    # Input: (total_steps, ...)
    # Output: (batch, T + 1, ...)
    total_steps = arr.shape[0]

    batch = total_steps - T
    if max_trajs is not None:
        batch = min(max_trajs, batch)

    start_idxs = np.around(np.linspace(0, total_steps - T - 1, num=batch)).astype(int)

    windows = [arr[start_idx : start_idx + T + 1] for start_idx in start_idxs]
    windowed_data = np.stack(windows, axis=0)

    assert windowed_data.shape == (batch, T + 1, *arr.shape[1:])
    #
    # log.info(
    #     "Input array has shape {}. T: {}, batch: {}. windowed_data.shape: {}".format(
    #         arr.shape, T, batch, windowed_data.shape
    #     )
    # )

    return windowed_data


def split_data(data: PlanetsData, T: int, train_frac: float = 0.8) -> Tuple[PlanetsDSet, PlanetsDSet]:
    # 1: Split data into train and test.
    # 2: Split up the data into windows of length T + 1.

    n_steps = data.n_steps

    if n_steps > 2048 + T:
        n_train = 2048 + T
    elif n_steps > 1024 + T:
        n_train = 1024 + T
    else:
        raise ValueError("")

    arrs = dict(q=data.pos, p=data.lin_mom, R=data.rot, Pi=data.ang_mom)
    train_arrs = {key: _window_data(arr[:n_train], T) for key, arr in arrs.items()}
    test_arrs = {key: _window_data(arr[n_train:], T) for key, arr in arrs.items()}

    train_dset = PlanetsDSet(data.names, data.dt, **train_arrs)
    test_dset = PlanetsDSet(data.names, data.dt, **test_arrs)

    return train_dset, test_dset


def merge_dsets(dsets: List[PlanetsDSet]) -> PlanetsDSet:
    # Make sure the names are all the same.
    names, dt, batch_per_traj = dsets[0].names, dsets[0].dt, dsets[0].n_batch_per_traj
    assert all([dset.names == names and dset.dt == dt and dset.n_batch_per_traj == batch_per_traj for dset in dsets])

    qs, ps, Rs, Pis = [], [], [], []
    for dset in dsets:
        qs.append(dset.q)
        ps.append(dset.p)
        Rs.append(dset.R)
        Pis.append(dset.Pi)

    q = np.concatenate(qs, axis=0)
    p = np.concatenate(ps, axis=0)
    R = np.concatenate(Rs, axis=0)
    Pi = np.concatenate(Pis, axis=0)

    return PlanetsDSet(names, dt, q, p, R, Pi, batch_per_traj)


def _subsample_data(arrs: Dict[str, np.ndarray], scaling: int) -> Dict[str, np.ndarray]:
    return {k: v[::scaling] for k, v in arrs.items()}


def _split_datas(datas: List[PlanetsData], T: int, train_frac: float = 0.8) -> Tuple[PlanetsDSet, PlanetsDSet]:
    # Use hierarchical scheme for windowing data.
    #
    assert len(datas) > 0

    n_datas = len(datas)
    n_train = int(train_frac * n_datas)

    dsets = []

    for data in track(datas, description="Windowing data..."):
        arrs = dict(q=data.pos, p=data.lin_mom, R=data.rot, Pi=data.ang_mom)
        arrs = {key: _window_data(arr, T) for key, arr in arrs.items()}

        n_batch = arrs["q"].shape[0]

        dsets.append(PlanetsDSet(data.names, data.dt, **arrs, n_batch_per_traj=n_batch))

    train_dsets = dsets[:n_train]
    val_dsets = dsets[n_train:]

    train_dset = merge_dsets(train_dsets)
    val_dset = merge_dsets(val_dsets)

    return train_dset, val_dset
