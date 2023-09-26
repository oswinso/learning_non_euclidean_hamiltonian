import pathlib
from typing import Any, Callable, Optional

import ipdb
import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from loggers.base_logger import BaseLogger
from loggers.logger_types import LogKey as LK
from loggers.logger_types import TrainLogData, ValLogData
from utils.types import ja


def to_float(x: Any) -> float:
    if isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):
        return float(x.mean())
    else:
        return float(x)


class TBLogCtx:
    def __init__(self, writer: SummaryWriter, prefix: str):
        self._writer = writer
        self._data = None
        self._prefix = prefix

    def __enter__(self):
        assert self._data is not None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._data = None

    def __call__(self, data: TrainLogData):
        self._data = data
        return self

    def add_scalar(self, tag: str, key: str, fn: Optional[Callable[[ja], float]] = None) -> None:
        if key not in self._data:
            return

        val = self._data[key]

        if val is None:
            # If val is None then don't log it.
            return

        if fn is not None:
            val = fn(val)

        return self.add_scalar_manual(tag, val)

    def add_scalar_manual(self, tag: str, value) -> None:
        val = to_float(value)
        step = self._data[LK.idx]
        self._writer.add_scalar("{}/{}".format(self._prefix, tag), val, global_step=step)

    def add_scalars(self, key: str, tag_prefix: Optional[str] = None) -> None:
        if key not in self._data:
            return

        val = self._data[key]
        assert isinstance(val, dict)

        step = self._data[LK.idx]
        for name, arr in val.items():
            val = to_float(arr)
            if tag_prefix is not None:
                self._writer.add_scalar("{}/{}/{}".format(self._prefix, tag_prefix, name), val, global_step=step)
            else:
                self._writer.add_scalar("{}/{}".format(self._prefix, name), val, global_step=step)

    def add_histogram_manual(self, tag: str, value):
        assert value.ndim == 1

        step = self._data[LK.idx]
        self._writer.add_histogram(tag, value, global_step=step)


def idx(num: int):
    def _extract_idx(loss: ja) -> float:
        return float(loss[num])

    return _extract_idx


def final_idx():
    return idx(-1)


def compute_mean(loss: ja) -> float:
    return jnp.mean(loss)


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir: pathlib.Path):
        log_dir = str(log_dir.absolute())
        self.writer = SummaryWriter(log_dir, flush_secs=10)

        self.train_details_ctx = TBLogCtx(self.writer, "TrainDetails")
        self.train_gradnorms = TBLogCtx(self.writer, "TrainGradNorms")
        self.train_ctx = TBLogCtx(self.writer, "Train")
        self.val_ctx = TBLogCtx(self.writer, "Val")
        self.val_V_ctx = TBLogCtx(self.writer, "Val V")

        self.hist_ctx = TBLogCtx(self.writer, "WeightHist")

    def log_train(self, data: TrainLogData) -> None:
        with jax.profiler.TraceAnnotation("tensorboard_logger log_train"):
            with self.train_ctx(data) as l:
                l.add_scalar("Loss", LK.loss)
                l.add_scalar("LR", LK.lr)
                l.add_scalar("Mean p MSE", LK.p_loss)
                l.add_scalar("Mean q MSE", LK.q_loss)
                l.add_scalar("Mean R MSE", LK.R_loss)
                l.add_scalar("Mean Pi MSE", LK.Pi_loss)
                l.add_scalar("Grad Norm (Before)", LK.grad_global_norm_before)
                l.add_scalar("Grad Norm (After)", LK.grad_global_norm_after)
                l.add_scalar("Integrate T", LK.integrate_T)

            with self.train_gradnorms(data) as l:
                l.add_scalar("q gradnorm", LK.q_gradnorm)
                l.add_scalar("p gradnorm", LK.p_gradnorm)
                l.add_scalar("R gradnorm", LK.R_gradnorm)
                l.add_scalar("Pi gradnorm", LK.Pi_gradnorm)

            with self.train_details_ctx(data) as l:
                l.add_scalar("Final Step Weight", LK.final_step_weight)
                l.add_scalar("t=1 p MSE", LK.p_loss, idx(1))
                l.add_scalar("t=1 q MSE", LK.q_loss, idx(1))
                l.add_scalar("t=1 R MSE", LK.R_loss, idx(1))
                l.add_scalar("t=1 Pi MSE", LK.Pi_loss, idx(1))
                l.add_scalar("t=2 p MSE", LK.p_loss, idx(2))
                l.add_scalar("t=2 q MSE", LK.q_loss, idx(2))
                l.add_scalar("t=2 R MSE", LK.R_loss, idx(2))
                l.add_scalar("t=2 Pi MSE", LK.Pi_loss, idx(2))
                l.add_scalar("Final p MSE", LK.p_loss, final_idx())
                l.add_scalar("Final q MSE", LK.q_loss, final_idx())
                l.add_scalar("Final R MSE", LK.R_loss, final_idx())
                l.add_scalar("Final Pi MSE", LK.Pi_loss, final_idx())

    def log_val(self, data: ValLogData) -> None:
        with jax.profiler.TraceAnnotation("tensorboard_logger log_val"):
            with self.val_ctx(data) as l:
                l.add_scalar("Loss", LK.loss)
                l.add_scalar("Final p MSE", LK.p_loss, final_idx())
                l.add_scalar("Final q MSE", LK.q_loss, final_idx())
                l.add_scalar("Final R MSE", LK.R_loss, final_idx())
                l.add_scalar("Final Pi MSE", LK.Pi_loss, final_idx())
                #
                l.add_scalar("p MSE", LK.p_loss, compute_mean)
                l.add_scalar("q MSE", LK.q_loss, compute_mean)
                l.add_scalar("R MSE", LK.R_loss, compute_mean)
                l.add_scalar("Pi MSE", LK.Pi_loss, compute_mean)
                #
                l.add_scalar("Weight Norm", LK.weight_norm)

            with self.val_V_ctx(data) as l:
                l.add_scalar("Vdiff mean", LK.Vdiff_mean)
                l.add_scalar("Vdiff max", LK.Vdiff_mean)

            with self.hist_ctx(data) as l:
                # Log the distribution
                keys = ["multi_so3_rnn/~/rigid_V_net/~/dec/~/linear_2", "multi_so3_rnn/~/rigid_V_net/~/last_layer"]
                params = data[LK.train_state].params
                for key in keys:
                    if key in params:
                        last_layer_weights = params[key]["w"]
                        last_layer_weights = np.array(last_layer_weights.flatten())
                        l.add_histogram_manual("Last Layer Weights", last_layer_weights)
                        break

    def close(self) -> None:
        self.writer.close()
