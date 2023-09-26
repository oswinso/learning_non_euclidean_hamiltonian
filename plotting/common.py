from typing import Any, Dict, List, Optional, Union

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from sklearn.decomposition import PCA

import plotting.utils
from grit_dset.planets_data import PlanetsData


def compare_orbit_positions(pred_q: np.ndarray, true_q: np.ndarray) -> plt.Figure:
    # pred_q: (T, n_bodies, 3), true_q: (T, n_bodies, 3)
    assert pred_q.shape == true_q.shape

    T, n_bodies, _ = pred_q.shape

    pred_q, true_q = np.array(pred_q), np.array(true_q)

    # # First, find the axis which can show the largest error. i.e., PCA on the error vectors, and then project.
    # # (
    # q_error = (pred_q - true_q).reshape((-1, 3))
    # pca = PCA(n_components=2)
    # pca.fit(q_error)

    def transform(arr: np.ndarray) -> np.ndarray:
        # return pca.transform(arr)
        return arr[..., :2]

    # Project pred_q and true_q.
    pred_red = transform(pred_q.reshape((-1, 3))).reshape((T, n_bodies, 2))
    true_red = transform(true_q.reshape((-1, 3))).reshape((T, n_bodies, 2))

    fig, ax = plt.subplots(constrained_layout=True)

    colors = plotting.utils.default_color_cycle()
    pred_style = dict(ls="-")
    true_style = dict(ls="--")
    start_style = dict(marker="s", ms=15)

    for body in range(n_bodies):
        color = colors[body]
        ax.plot(pred_red[:, body, 0], pred_red[:, body, 1], color=color, **pred_style)
        ax.plot(true_red[:, body, 0], true_red[:, body, 1], color=color, **true_style)

        # Plot the (common) starting point.
        ax.plot(true_red[0, body, 0], true_red[0, body, 1], color=color, **start_style)

    handles = [
        plt.Line2D([0], [0], **pred_style, label="Predicted"),
        plt.Line2D([0], [0], **true_style, label="True"),
    ]
    ax.legend(handles=handles)

    return fig


def compare_orbit_positions_sunearth(
    pred_qs: Union[np.ndarray, Dict[str, np.ndarray]], true_q: np.ndarray, fig: Optional[plt.Figure] = None, fig_kw={}
) -> plt.Figure:
    # pred_q: [(batch, pred_T, n_bodies, 3)], true_q: (batch, true_T, n_bodies, 3)

    if not isinstance(pred_qs, dict):
        assert isinstance(pred_qs, np.ndarray) or isinstance(pred_qs, jnp.ndarray)
        pred_qs = dict(pred=pred_qs)
    batch, T, n_bodies, _ = true_q.shape
    batch, pred_T, n_bodies, _ = list(pred_qs.values())[0].shape

    def transform(arr: np.ndarray) -> np.ndarray:
        # return pca.transform(arr)
        return arr[..., :2]

    # Project pred_q and true_q.
    pred_reds = {
        key: transform(pred_q.reshape((-1, 3))).reshape((batch, pred_T, n_bodies, 2)) for key, pred_q in pred_qs.items()
    }
    true_red = transform(true_q.reshape((-1, 3))).reshape((batch, T, n_bodies, 2))

    if fig is None:
        fig = plt.Figure(constrained_layout=True, **fig_kw)
    ax = fig.subplots()
    ax.set(aspect="equal")

    n_preds = len(pred_qs)
    colors = plotting.utils.default_color_cycle()

    if len(pred_qs) > 1:
        lss = plotting.utils.distinct_linestyles(len(pred_qs))
    else:
        lss = ["-"]

    pred_style = dict(alpha=0.7, zorder=3.5)
    true_style = dict(ls="-", alpha=0.2, lw=8.0, color=colors[n_preds])
    start_style = dict(marker="s", s=10 ** 2, zorder=4)

    for body in [1]:
        for ii, (key, pred_red) in enumerate(pred_reds.items()):
            color = colors[ii]
            pred_segs = pred_red[:, :, body, :]
            assert pred_segs.shape == (batch, pred_T, 2)
            pred_col = LineCollection(pred_segs, color=color, linestyles=lss[ii], **pred_style)
            ax.add_collection(pred_col)

        true_segs = true_red[:, :, body, :]
        assert true_segs.shape == (batch, T, 2)
        true_col = LineCollection(true_segs, **true_style)
        ax.add_collection(true_col)

        # Plot the (common) starting point.
        start_segs = true_red[:, [0], body, :]
        ax.scatter(start_segs[:, :, 0], start_segs[:, :, 1], **start_style)

    ax.autoscale_view()

    pred_handles = [
        plt.Line2D([0], [0], linestyle=lss[ii], **pred_style, color=colors[ii], label=key)
        for ii, key in enumerate(pred_qs.keys())
    ]

    handles = [
        *pred_handles,
        plt.Line2D([0], [0], **true_style, label="True"),
    ]
    ax.legend(handles=handles)

    return fig


def compare_orbit_positions_batch(
    pred_qs: Union[np.ndarray, Dict[str, np.ndarray]], true_q: np.ndarray, fig: Optional[plt.Figure] = None, fig_kw={}
) -> plt.Figure:
    # pred_q: [(batch, pred_T, n_bodies, 3)], true_q: (batch, true_T, n_bodies, 3)

    batch, true_T, n_bodies, _ = true_q.shape

    if not isinstance(pred_qs, dict):
        assert isinstance(pred_qs, np.ndarray) or isinstance(pred_qs, jnp.ndarray)
        pred_qs = dict(pred=pred_qs)

    _, pred_T, n_bodies, _ = list(pred_qs.values())[0].shape

    for key, pred_q in pred_qs.items():
        # The prediction horizon can be different.
        same_shape = pred_q.shape[0] == true_q.shape[0] and pred_q.shape[2:] == true_q.shape[2:]
        if not same_shape:
            print("pred_q.shape: {}, true_q.shape: {}".format(pred_q.shape, true_q.shape))
            assert pred_q.shape == true_q.shape

    if n_bodies == 2:
        return compare_orbit_positions_sunearth(pred_qs, true_q, fig=fig, fig_kw=fig_kw)

    def transform(arr: np.ndarray) -> np.ndarray:
        # return pca.transform(arr)
        return arr[..., :2]

    # Project pred_q and true_q.
    pred_reds = {
        key: transform(pred_q.reshape((-1, 3))).reshape((batch, pred_T, n_bodies, 2)) for key, pred_q in pred_qs.items()
    }
    true_red = transform(true_q.reshape((-1, 3))).reshape((batch, true_T, n_bodies, 2))

    if fig is None:
        fig = plt.Figure(constrained_layout=True, **fig_kw)
    ax = fig.subplots()
    ax.set(aspect="equal")

    n_lines = len(pred_qs) + 1
    colors = plotting.utils.default_color_cycle()

    if len(pred_qs) > 1:
        lss = plotting.utils.distinct_linestyles(len(pred_qs))
    else:
        lss = ["-"]

    pred_style = dict(alpha=0.8, zorder=3.5)
    true_style = dict(ls="-", alpha=0.2, lw=5.0)
    start_style = dict(marker="s", s=10 ** 2, zorder=4)

    for body in range(n_bodies):
        color = colors[body]
        body_style = dict(color=color)

        for ii, (key, pred_red) in enumerate(pred_reds.items()):
            ls = lss[ii]
            pred_segs = pred_red[:, :, body, :]
            assert pred_segs.shape == (batch, pred_T, 2)
            pred_col = LineCollection(pred_segs, **body_style, **pred_style, ls=ls)
            ax.add_collection(pred_col)

        true_segs = true_red[:, :, body, :]
        assert true_segs.shape == (batch, true_T, 2)
        true_col = LineCollection(true_segs, **body_style, **true_style)
        ax.add_collection(true_col)

        # Plot the (common) starting point.
        start_segs = true_red[:, [0], body, :]
        ax.scatter(start_segs[:, :, 0], start_segs[:, :, 1], **body_style, **start_style)

    ax.autoscale_view()

    pred_handles = [plt.Line2D([0], [0], **pred_style, ls=lss[ii], label=key) for ii, key in enumerate(pred_qs.keys())]

    handles = [
        *pred_handles,
        plt.Line2D([0], [0], **true_style, label="True"),
    ]
    ax.legend(handles=handles)

    return fig


def orbit_position_errors(
    pred_qs: Union[np.ndarray, Dict[str, np.ndarray]],
    true_q: np.ndarray,
    bodies: List[str],
    dt: float,
    fig: Optional[plt.Figure] = None,
    fig_kw={},
) -> plt.Figure:
    batch, T, n_bodies, _ = true_q.shape

    if not isinstance(pred_qs, dict):
        assert isinstance(pred_qs, np.ndarray) or isinstance(pred_qs, jnp.ndarray)
        pred_qs = dict(pred=pred_qs)

    for key, pred_q in pred_qs.items():
        if pred_q.shape != true_q.shape:
            print("pred_q.shape: {}, true_q.shape: {}".format(pred_q.shape, true_q.shape))
            assert pred_q.shape == true_q.shape

    batch, T, n_bodies, _ = true_q.shape

    # Project pred_q and true_q.
    # (batch, T)
    pred_errs = {key: np.linalg.norm(pred_q - true_q, axis=-1) for key, pred_q in pred_qs.items()}

    if fig is None:
        fig = plt.Figure(constrained_layout=True, **fig_kw)
    axes = fig.subplots(n_bodies)
    [ax.set(title=body_name) for body_name, ax in zip(bodies, axes)]

    colors = plotting.utils.default_color_cycle()

    err_style = dict()

    ts = dt * np.arange(T)

    for body in range(n_bodies):
        ax = axes[body]

        for ii, (key, pred_err) in enumerate(pred_errs.items()):
            color = colors[ii]

            pred_err = pred_err[:, :, body]
            assert pred_err.shape == (batch, T)

            traj = np.stack(np.broadcast_arrays(ts, pred_err), axis=2)
            assert traj.shape == (batch, T, 2)
            col = LineCollection(traj, **err_style, color=color)
            ax.add_collection(col)

        ax.autoscale_view()

    handles = [plt.Line2D([0], [0], **err_style, color=colors[ii], label=key) for ii, key in enumerate(pred_qs.keys())]
    fig.legend(handles=handles)

    return fig


def axis_to_idx(s: str) -> int:
    if s == "x":
        return 0
    if s == "y":
        return 1
    if s == "z":
        return 2
    raise RuntimeError("Unknown s {}".format(s))


def plot_orbits(
    data: PlanetsData,
    bodies: List[int],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fig_kw: Dict[str, Any] = {},
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    axis_order: List[str] = ["x", "y", "z"],
) -> plt.Figure:
    # For now, plot the x-y plane. Line plot for trajectory, scatter for z-axis, use alpha to show z-axis?

    # Assume body
    assert max(bodies) < data.n_bodies

    if ax is None:
        if fig is None:
            fig: plt.Figure = plt.figure(constrained_layout=True)
        ax: plt.Axes = fig.subplots()
    ax.set(xlabel=axis_order[0], ylabel=axis_order[1])
    fig = ax.figure

    if axis_order == ["x", "y", "z"]:
        ax.set(aspect="equal")

    colors = plotting.utils.seaborn_tab10()

    start_idx = None
    end_idx = None

    times = data.dt * np.arange(data.n_times)

    if end_time is not None:
        end_idx = np.searchsorted(times, end_time, "left")
        times = times[:end_idx]
    if start_time is not None:
        start_idx = np.searchsorted(times, start_time, "right")
        times = times[start_idx:]

    # Put a marker at (0, 0).
    ax.plot([0], [0], marker="o", color=colors[-1])

    traj_style = dict(zorder=2, lw=0.2, alpha=0.5)
    marker_style = dict(marker="o", cmap="turbo", zorder=3)

    axis_order_idx = [axis_to_idx(s) for s in axis_order]

    for ii, body_idx in enumerate(bodies):
        color = colors[ii]
        pos = data.pos[:, body_idx]

        if end_idx is not None:
            pos = pos[:end_idx]
        if start_idx is not None:
            pos = pos[start_idx:]

        # (T, )
        x_pos = pos[:, axis_order_idx[0]]
        y_pos = pos[:, axis_order_idx[1]]
        z_pos = pos[:, axis_order_idx[2]]

        # Plot the trajectory.
        ax.plot(x_pos, y_pos, **traj_style, color=color, label=data.names[body_idx])

        # Subtract the mean.
        mean_z = np.mean(z_pos)
        sign_changes = plotting.utils.get_sign_changes(z_pos - mean_z)
        x_pos, y_pos, _time = x_pos[sign_changes], y_pos[sign_changes], times[sign_changes]

        # print("Sign changes at times {}, idx={}".format(_time, sign_changes.nonzero()))

        # Put a marker every time the z_pos intersects zero.
        cp = ax.scatter(x_pos, y_pos, c=_time, **marker_style)
        fig.colorbar(cp, ax=ax)
    ax.legend()

    return fig


def plot_nodes(
    data: PlanetsData,
    bodies: List[int],
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fig_kw: Dict[str, Any] = {},
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    label_prefix: str = "",
    color_offset: int = 0,
) -> plt.Figure:
    assert max(bodies) < data.n_bodies

    if ax is None:
        if fig is None:
            fig: plt.Figure = plt.figure(constrained_layout=True)
        ax: plt.Axes = fig.subplots()
    fig = ax.figure

    colors = plotting.utils.seaborn_tab10()

    start_idx = None
    end_idx = None

    times = data.dt * np.arange(data.n_times)

    if end_idx is not None:
        assert end_time is None
        times = times[:end_idx]
    if start_idx is not None:
        assert start_time is None
        times = times[start_idx:]

    if end_time is not None:
        end_idx = np.searchsorted(times, end_time, "left")
        times = times[:end_idx]
    if start_time is not None:
        start_idx = np.searchsorted(times, start_time, "right")
        times = times[start_idx:]

    marker_style = dict(marker="o", s=4 ** 2, zorder=4)

    for ii, body_idx in enumerate(bodies):
        color = colors[color_offset + ii]
        pos = data.pos[:, body_idx]

        if end_idx is not None:
            pos = pos[:end_idx]
        if start_idx is not None:
            pos = pos[start_idx:]

        x_pos, y_pos, z_pos = pos[:, 0], pos[:, 1], pos[:, 2]

        mean_z = np.mean(z_pos)
        sign_changes = plotting.utils.get_sign_changes(z_pos - mean_z)
        x_pos, y_pos, _time = x_pos[sign_changes], y_pos[sign_changes], times[sign_changes]

        angles = np.arctan2(y_pos, x_pos)

        ax.scatter(_time, angles, label=f"{label_prefix}{data.names[body_idx]}", color=color, **marker_style)

    ax.legend()

    return fig
