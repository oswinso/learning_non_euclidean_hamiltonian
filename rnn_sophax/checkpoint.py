import pathlib
from typing import TypedDict, Union

import cloudpickle

from rnn_sophax.fit_multi import TrainState


class Checkpoint(TypedDict):
    step: int
    train_state: TrainState


def load_checkpoint(path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)

    assert path.exists() and path.is_file()
    ckpt_folder = path.parent.parent
    model_path = ckpt_folder / "model.pkl"

    # Get model.
    with open(model_path, "rb") as f:
        model, model_fn = cloudpickle.load(f)

    # Load checkpoint.
    with open(path, "rb") as f:
        ckpt: Checkpoint = cloudpickle.load(f)

    return model, model_fn, ckpt
