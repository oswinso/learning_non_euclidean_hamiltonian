import pathlib
from typing import Dict, List, Tuple

import chex
import haiku
import ipdb
import rich.columns
import typer

from grit.config import BodyCfg


def km_to_au(km: float) -> float:
    km_per_au = 149_597_870.700

    return km / km_per_au


def perturb_add(cfg: Dict[str, Tuple[float, float, float]], key: str, std: float, noise) -> None:
    import numpy as np

    assert noise.shape == (3,)
    orig_val = np.array(cfg[key])
    new_val = orig_val + std * noise

    cfg[key] = tuple(np.array(new_val))


def perturb_mult(cfg: Dict[str, Tuple[float, float, float]], key: str, std: float, noise) -> None:
    import numpy as np

    assert noise.shape == (3,)
    orig_val = np.array(cfg[key])
    new_val = np.clip(1 + std * noise, a_min=0.25, a_max=1.5) * orig_val

    cfg[key] = tuple(np.array(new_val))


def uniform_euler(cfg: Dict[str, Tuple[float, float, float]], key: str, rng: chex.PRNGKey) -> None:
    import jax
    import jax.numpy as jnp

    # Sample from SO3 uniformly.
    keys = jax.random.split(rng, 2)
    u1 = jax.random.uniform(keys[0], (1,))
    u2, u3 = jax.random.uniform(keys[2], (2,), minval=0, maxval=jnp.pi)

    a = jnp.sqrt(1 - u1)
    b = jnp.sqrt(u1)

    quat = jnp.array([a * jnp.sin(u2), a * jnp.cos(u2), b * jnp.sin(u3), b * jnp.cos(u3)])
    q0, q1, q2, q3 = quat

    x = jnp.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    y = jnp.arcsin(2 * (q0 * q2 - q3 * q1))
    z = jnp.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))

    cfg[key] = (float(x), float(y), float(z))


def perturb_orbital_elements(cfg: BodyCfg, mult_stds: List[float], rng: chex.PRNGKey) -> None:
    import jax
    import numpy as np

    # [a, e, i, Omega, omega, Mean Anomaly]
    # a: semimajor-axis
    # e: eccentricity
    # i: inclination
    # Omega: longitude of ascending node
    # omega: Argument of periapsis
    # Mean Anomaly:

    assert len(mult_stds) == 6
    noise = np.array(mult_stds) * jax.random.truncated_normal(rng, -3.0, 3.0, shape=(6,))
    noise = np.clip(1.0 + noise, a_min=0.25, a_max=1.5)

    orbital_elements = np.array(cfg["orbital_elements"])
    orbital_elements = np.array(orbital_elements * noise)

    cfg["orbital_elements"] = [*tuple(orbital_elements)]


def p_tup(arr: Tuple[float, ...]) -> str:
    strs = ["{:+12.6f}".format(num) for num in arr]

    return "({})".format(" ".join(strs))


def perturb_ic(init_cfg, rng):
    import copy

    import haiku as hk
    import jax

    from grit.config import InitSystemJson

    init_cfg: InitSystemJson
    rng: hk.PRNGSequence

    cfg = copy.deepcopy(init_cfg)

    n_bodies = len(cfg["body"])
    noise = jax.random.truncated_normal(next(rng), -3.0, 3.0, shape=(n_bodies, 20, 3))

    # Get the radius and masses of each object.
    for ii, body_cfg in enumerate(cfg["body"]):
        if "radius" in body_cfg:
            radius_au = km_to_au(body_cfg["radius"])
        elif "diameter" in body_cfg:
            radius_au = km_to_au(body_cfg["diameter"]) / 2
        else:
            raise RuntimeError("radius / diameter not in cfg")

        is_sun = body_cfg["name"] in ["Sun", "TSun"]
        if is_sun:
            # Don't move the Sun's position by that much.
            perturb_add(body_cfg, "position", 1e-6 * radius_au, noise[ii, 0])
        else:
            if "position" in body_cfg:
                perturb_add(body_cfg, "position", 0.1 * radius_au, noise[ii, 0])
                perturb_mult(body_cfg, "velocity", 0.05, noise[ii, 1])
            elif "orbital_elements" in body_cfg:
                perturb_orbital_elements(body_cfg, [0.1, 0.1, 0.1, 0.25, 0.25, 0.25], next(rng))

        perturb_add(body_cfg, "angular_velocity", 0.05, noise[ii, 2])
        perturb_mult(body_cfg, "angular_velocity", 0.05, noise[ii, 3])

        if is_sun:
            perturb_add(body_cfg, "angular_velocity", 0.001, noise[ii, 4])
            perturb_mult(body_cfg, "euler_angles", 0.05, noise[ii, 3])
            perturb_add(body_cfg, "euler_angles", 0.001, noise[ii, 5])
        else:
            perturb_add(body_cfg, "angular_velocity", 1.0, noise[ii, 4])
            uniform_euler(body_cfg, "euler_angles", next(rng))

        if "position" in body_cfg:
            print(
                "    {:12}: radius={:8.1e}, {}    {}    {}    {}".format(
                    body_cfg["name"],
                    radius_au,
                    p_tup(body_cfg["position"]),
                    p_tup(body_cfg["velocity"]),
                    p_tup(body_cfg["angular_velocity"]),
                    p_tup(body_cfg["euler_angles"]),
                )
            )

    return cfg


def main(path: pathlib.Path, n_trajs: int = typer.Option(16, help="Number of trajs to generate")):
    from rich import print
    from rich.progress import track

    from grit.config import read_init_system_json, read_target_time_json
    from grit.grit import run_grit
    from utils.jax_utils import jax_use_cpu, jax_use_double
    from utils.paths import grit_gen_data_dir, root_dir
    from utils.stamp import get_datetime_stamp

    if not path.exists():
        print("Path {} doesn't exist!".format(path))
        exit(1)

    if not path.is_dir():
        print("Path should be a dir.")
        exit(1)

    GRIT_PATH = root_dir() / "external/GRIT/cmake-build-release/src/simulate"
    assert GRIT_PATH.exists() and GRIT_PATH.is_file()

    GRIT_DATA_PATH = root_dir() / "grit_data"

    # 1: Read in the json file.
    init_cfg = read_init_system_json(path)
    target_time = read_target_time_json(path)

    datetime_stamp = get_datetime_stamp()
    folder_name = path.name
    rel_path = path.absolute().relative_to(GRIT_DATA_PATH)

    if rel_path.parent != pathlib.Path("."):
        output_dir = grit_gen_data_dir() / rel_path.parent / f"{datetime_stamp}_{folder_name}"
    else:
        output_dir = grit_gen_data_dir() / f"{datetime_stamp}_{folder_name}"

    output_dir.mkdir(exist_ok=True, parents=True)

    jax_use_cpu()
    jax_use_double()

    rng = haiku.PRNGSequence(13214)

    # 2: Generate the runs.
    for idx in track(range(n_trajs), "Running GRIT"):
        run_dir = output_dir / "{:03}".format(idx)

        if idx == 0:
            cfg = init_cfg
        else:
            cfg = perturb_ic(init_cfg, rng)

        run_grit(cfg, target_time, run_dir, GRIT_PATH)

    print("Done! Saved to output folder {}".format(output_dir))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
