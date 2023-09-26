import pathlib

import ipdb
import numpy as np
from rich import print

from loggers.base_logger import BaseLogger
from loggers.logger_types import LogKey as LK
from loggers.logger_types import TrainLogData, ValLogData


def to_duration_str(seconds: float) -> str:
    seconds = round(seconds)
    minutes, s = divmod(seconds, 60)
    h, m = divmod(minutes, 60)

    return "{:02d}:{:02d}:{:02d}".format(h, m, s)


class StdoutLogger(BaseLogger):
    def __init__(self):
        self.separator = "  |  "

        self._printed_header = False

    def _print_header(self) -> None:
        labels = [
            "{:5}".format("idx"),
            "{:8}".format("loss"),
            "{:8}".format("lr"),
            "{:8}".format("iter_s"),
            "{:8}".format("train_s"),
        ]
        label_str = self.separator.join(labels)
        print(label_str)
        print("-" * len(label_str))

    def close(self):
        ...

    def log_train(self, data: TrainLogData) -> None:
        if data[LK.idx] % 20 != 0:
            return

        if not self._printed_header:
            self._print_header()
            self._printed_header = True

        cols = [
            "{:5d}".format(data[LK.idx]),
            "{:8.2e}".format(data[LK.loss]),
            "{:8.1e}".format(data[LK.lr]),
            "{:8.1e}".format(data[LK.iter_time]),
            "{:8}".format(to_duration_str(data[LK.train_time])),
        ]
        print(self.separator.join(cols))

    def log_val(self, data: ValLogData) -> None:
        ...
