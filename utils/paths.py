import pathlib


def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def grit_data_dir() -> pathlib.Path:
    return root_dir() / "grit_data"


def grit_gen_data_dir() -> pathlib.Path:
    return root_dir() / "grit_data/generated"


def dsets_dir() -> pathlib.Path:
    path = root_dir() / "dsets"
    path.mkdir(exist_ok=True)
    return path


def runs_dir() -> pathlib.Path:
    path = root_dir() / "runs"
    path.mkdir(exist_ok=True)
    return path
