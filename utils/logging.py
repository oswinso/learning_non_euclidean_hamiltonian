import logging
import pathlib

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(log_dir: pathlib.Path) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = open(log_dir / "log.txt", "w")
    file_console = Console(file=log_file, width=150)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[RichHandler(), RichHandler(console=file_console)],
    )
