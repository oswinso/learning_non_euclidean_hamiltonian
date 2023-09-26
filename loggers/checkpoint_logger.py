import logging
import pathlib
from typing import Dict, NamedTuple, Optional, TypedDict

import cloudpickle
import jax

from loggers.base_logger import BaseLogger
from loggers.logger_types import LogKey as LK
from loggers.logger_types import TrainLogData, ValLogData

log = logging.getLogger(__file__)


class Checkpoint(TypedDict):
    step: int
    train_state: NamedTuple


def save_checkpoint(path: pathlib.Path, data: ValLogData) -> None:
    ckpt: Checkpoint = dict(train_state=data[LK.train_state], step=data[LK.idx])

    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "wb") as f:
        cloudpickle.dump(ckpt, f)


def remove_checkpoint(path: pathlib.Path) -> None:
    if not path.exists():
        print("Trying to remove checkpoint {}, but doesn't exist??")
        return

    path.unlink()


class CheckpointLogger(BaseLogger):
    def __init__(self, log_dir: pathlib.Path, save_every: int, save_top_k: int = 3):
        self._log_dir = log_dir
        self.save_every = save_every

        self.best_k_models: Dict[pathlib.Path, float] = {}
        self.kth_best_model_path = None
        self.kth_value = None
        self.best_model_path = None
        self.best_model_score = None
        self.k = save_top_k

    def log_train(self, data: TrainLogData) -> None:
        ...

    def _check_save_every(self, data: ValLogData) -> None:
        step = data[LK.idx]
        if step % self.save_every == 0:
            filepath = self._get_filepath(step, topk=False)
            save_checkpoint(filepath, data)
            log.info("Saved checkpoint to {}".format(filepath))

    def log_val(self, data: ValLogData) -> None:
        with jax.profiler.TraceAnnotation("checkpoint_logger log_val"):
            self._check_save_every(data)

        # cost = float(data.loss_aux.final_costs.V.mean())
        # del_filepath = None
        # if len(self.best_k_models) == self.k:
        #     del_filepath = self.kth_best_model_path
        #     self.best_k_models.pop(del_filepath)
        #
        # filepath = self._get_filepath(data.step, cost, topk=True)
        #
        # # Save the current score.
        # self.best_k_models[filepath] = cost
        #
        # if len(self.best_k_models) == self.k:
        #     self.kth_best_model_path = max(self.best_k_models, key=self.best_k_models.get)
        #     self.kth_value = self.best_k_models[self.kth_best_model_path]
        #
        # self.best_model_path = min(self.best_k_models, key=self.best_k_models.get)
        # self.best_model_score = self.best_k_models[self.best_model_path]
        #
        # save_checkpoint(filepath, data)
        #
        # if del_filepath is not None and filepath != del_filepath:
        #     remove_checkpoint(del_filepath)

    def _get_filepath(self, step: int, metric: Optional[float] = None, topk: bool = False) -> pathlib.Path:
        if topk:
            name = "top_{step:07}_{metric:.1f}.ckpt".format(step=step, metric=metric)
        else:
            name = "{step:07}.ckpt".format(step=step, metric=metric)

        return self._log_dir / "checkpoints" / name
