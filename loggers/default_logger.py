import pathlib

from loggers.base_logger import BaseLogger, make_logger
from loggers.stdout_logger import StdoutLogger
from loggers.tensorboard_logger import TensorboardLogger
from loggers.checkpoint_logger import CheckpointLogger


def get_default_logger(log_dir: pathlib.Path) -> BaseLogger:
    loggers = [TensorboardLogger(log_dir), StdoutLogger(), CheckpointLogger(log_dir, save_every=200)]
    return make_logger(loggers)
