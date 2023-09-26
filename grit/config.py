import json
import pathlib
from typing import Any, Dict, List, Tuple, TypedDict, Optional

Vec3 = Tuple[float, float, float]


class BodyCfg(TypedDict):
    name: str
    rigid: bool
    mass: float
    position: Vec3
    velocity: Vec3
    orbital_elements: List[float]
    angular_velocity: Vec3
    euler_angles: Vec3
    radius: float
    diameter: float
    oblateness: float
    time_lag: float
    Q: float


class Parameters(TypedDict):
    tide: bool
    GR: bool
    step_size: bool
    save_interval: float
    output_gaps: float
    scheme: str
    potential_order: int


class InitSystemJson(TypedDict):
    system_name: str
    euler_convention: Optional[str]
    body: List[BodyCfg]
    current_time: float
    parameters: Parameters
    output_format: Dict[str, bool]
    tidal_pairs: Dict[str, Any]


def read_init_system_json(path: pathlib.Path) -> InitSystemJson:
    json_path = path / "init_system.json"
    assert json_path.exists()
    with open(json_path, "r") as f:
        return json.load(f)


def read_target_time_json(path: pathlib.Path) -> float:
    json_path = path / "target_time.json"
    assert json_path.exists()
    with open(json_path, "r") as f:
        out = json.load(f)

    assert "target_time" in out
    target_time = out["target_time"]
    assert isinstance(target_time, float)

    return target_time


def save_init_system_json(cfg: InitSystemJson, path: pathlib.Path) -> None:
    path.mkdir(exist_ok=True, parents=True)
    path = path / "init_system.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)


def save_target_time_json(target_time: float, path: pathlib.Path) -> None:
    path.mkdir(exist_ok=True, parents=True)
    path = path / "target_time.json"

    cfg = dict(target_time=target_time)

    with open(path, "w") as f:
        json.dump(cfg, f)
