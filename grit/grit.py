import pathlib
import subprocess

from rich import print

from grit.config import InitSystemJson, save_init_system_json, save_target_time_json


def run_grit(cfg: InitSystemJson, target_time: float, output_path: pathlib.Path, grit_exe: pathlib.Path) -> None:
    # 1: Save the init cfg to the output folder.
    save_init_system_json(cfg, output_path)
    save_target_time_json(target_time, output_path)

    # 2: Run the executable.
    run_args = [str(grit_exe.absolute()), str(output_path.absolute())]
    print("Running with args  {}".format(run_args))
    proc = subprocess.run(run_args, check=True, capture_output=True)
    proc.check_returncode()
