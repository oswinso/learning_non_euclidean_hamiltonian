from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def use_cairo_qt() -> None:
    try:
        matplotlib.use("module://mplcairo.qt")
    except:
        pass


def set_style() -> None:
    matplotlib.set_loglevel("warning")

    plt.style.use(["seaborn"])
    params = {
        # Minor Ticks
        "ytick.minor.size": 2.0,
        "ytick.minor.width": 1.0,
        # Fonts
        "legend.fontsize": "large",
        "figure.titlesize": "xx-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
        # Max figures
        "figure.max_open_warning": 150,
    }
    plt.rcParams.update(params)


# fmt: off
_xgfs_normal6 = [(64, 83, 211), (221, 179, 16), (181, 29, 20), (0, 190, 255), (251, 73, 176), (0, 178, 93), (202, 202, 202)]
_xgfs_normal12 = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
_xgfs_bright6 = [(239, 230, 69), (233, 53, 161), (0, 227, 255), (225, 86, 44), (83, 126, 255), (0, 203, 133), (238, 238, 238)]
_xgfs_dark6 = [(0, 89, 0), (0, 0, 120), (73, 13, 0), (138, 3, 79), (0, 90, 138), (68, 53, 0), (88, 88, 88)]
_xgfs_fancy6 = [(86, 100, 26), (192, 175, 251), (230, 161, 118), (0, 103, 138), (152, 68, 100), (94, 204, 171), (205, 205, 205)]
_xgfs_tarnish6 = [(39, 77, 82), (199, 162, 166), (129, 139, 112), (96, 78, 60), (140, 159, 183), (121, 104, 128), (192, 192, 192)]
# fmt: on


def xgfs_normal6() -> np.ndarray:
    return np.array(_xgfs_normal6) / 255


def xgfs_normal12() -> np.ndarray:
    """[yellow, purple, sea blue, dark green, turquoise, magenta, red-brown, coral, dark blue,
    sky blue, olive, green, grey]
    """
    return np.array(_xgfs_normal12) / 255


def default_color_cycle() -> np.ndarray:
    # Length of 8.
    colors = np.array(
        ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#ba4fa7", "#d78b5c"], dtype="<U7"
    )
    return colors
    # return np.array(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def seaborn_tab10() -> np.ndarray:
    return np.array(
        [
            [0.121568, 0.466667, 0.705882],
            [1.000000, 0.498039, 0.054902],
            [0.172549, 0.627451, 0.172549],
            [0.839215, 0.152941, 0.156863],
            [0.580392, 0.403922, 0.741176],
            [0.549019, 0.337255, 0.294118],
            [0.890196, 0.466667, 0.760784],
            [0.498039, 0.498039, 0.498039],
            [0.737255, 0.741176, 0.133333],
            [0.090196, 0.745098, 0.811765],
        ]
    )


def _register_cmaps():
    sns_cmaps = ["icefire"]

    for cmap_name in sns_cmaps:
        if cmap_name not in matplotlib.colormaps:
            cmap = sns.color_palette(cmap_name, as_cmap=True)
            matplotlib.colormaps.register(cmap, cmap_name)
            print("Registered colormap {}!".format(cmap_name))


_register_cmaps()

_linestyles = [()]


def gen_linestyles(n_modes: int) -> List[Tuple[float, Tuple[float, float]]]:
    # Random offset + random lengths.
    rng = np.random.default_rng()
    space_len = 5
    line_len = 4
    eps = 0.5

    offsets = rng.uniform(0, space_len + line_len, n_modes)
    spaces = rng.uniform(space_len - eps, space_len + eps, n_modes)
    lengths = rng.uniform(line_len - eps, line_len + eps, n_modes)

    linestyles = []
    for idx in range(n_modes):
        ls = (offsets[idx], (spaces[idx], lengths[idx]))
        linestyles.append(ls)

    return linestyles


def distinct_linestyles(n_modes: int) -> List[Tuple[float, Tuple[float, float]]]:
    linestyles = [
        (0, (5, 0)),
        (0, (5, 5)),
        (0, (3, 3, 1, 3)),
        (0, (0.8, 1.0)),
        (0, (6, 2, 0.8, 1, 0.8, 1, 0.8, 2)),
        (0, (5, 1, 5, 2, 0.8, 2)),
    ]
    return linestyles[:n_modes]


def get_sign_changes(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 1

    # Perturb by tiny bit so 0.0 -> 1.0 doesn't count as a sign change.
    a = np.where(arr == 0, 1e-80, arr)
    mask = np.sign(a[:-1]) != np.sign(a[1:])
    mask = np.concatenate([[False], mask], axis=0)

    assert mask.shape == arr.shape

    return mask
