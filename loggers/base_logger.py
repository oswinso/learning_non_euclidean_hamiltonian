import weakref
from abc import ABC, abstractmethod
from typing import List, Union

from loggers.logger_types import TrainLogData, ValLogData


class BaseLogger(ABC):
    @abstractmethod
    def log_train(self, data: TrainLogData) -> None:
        ...

    @abstractmethod
    def log_val(self, data: ValLogData) -> None:
        ...

    def close(self) -> None:
        ...


class AutoCloseLogger(BaseLogger):
    def __init__(self, logger: BaseLogger):
        self._base = logger
        self._finalizer = weakref.finalize(self, logger.close)

    def log_train(self, data: TrainLogData) -> None:
        self._base.log_train(data)

    def log_val(self, data: ValLogData) -> None:
        self._base.log_val(data)

    def close(self) -> None:
        if self._finalizer.detach():
            self._base.close()
        self._base = None


class AggregateLogger(BaseLogger):
    def __init__(self, loggers: List[BaseLogger]):
        self._loggers = loggers

    def log_train(self, data: TrainLogData) -> None:
        for logger in self._loggers:
            logger.log_train(data)

    def log_val(self, data: ValLogData) -> None:
        for logger in self._loggers:
            logger.log_val(data)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()
        self._loggers = None


def make_logger(loggers: Union[BaseLogger, List[BaseLogger]]) -> BaseLogger:
    if not isinstance(loggers, list):
        return AutoCloseLogger(loggers)
    if len(loggers) == 1:
        return loggers[0]

    return AutoCloseLogger(AggregateLogger(loggers))
