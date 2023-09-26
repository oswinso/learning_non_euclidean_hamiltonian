import logging
import pathlib
import random
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import cloudpickle
import ipdb
import jax.profiler
import optax

from grit_dset.planets_dset import PlanetsDSet, PlanetsDSetCol
from integrators.integrator_enum import IntegratorEnum
from loggers.default_logger import get_default_logger
from models.multiso3rnn import (
    AnalyticMode,
    MultiSO3RNN,
    TransformedSO3RNN,
    dRPiMode,
    get_so3rnn_model,
)
from rnn_sophax.fit_multi import TrainState, fit, fit_pointmass
from rnn_sophax.fit_rigid import fit_rigid
from rnn_sophax.warmstart import WarmstartInfo
from utils.git_utils import log_git_info
from utils.haiku_utils import Activation, MLPCfg
from utils.jax_utils import jax_use_double
from utils.logging import setup_logger
from utils.paths import grit_data_dir, runs_dir
from utils.stamp import get_date_stamp, get_time_stamp

log = logging.getLogger(__file__)


def get_dset_path() -> pathlib.Path:
    # # TRAPPIST T2 Pointmass, output gap = 1.
    # dset_path = grit_data_dir() / "generated" / "20220317_135657_trappist_b_pt_T2"

    # # TRAPPIST T2, output gap = 1.
    # dset_path = grit_data_dir() / "generated" / "20220317_141308_trappist_b_T2"

    # # TRAPPIST T2, output gap = 1, 32 trajs
    # dset_path = grit_data_dir() / "generated" / "20220318_113849_trappist_b_T2"

    # # TRAPPIST T2, output gap = 1, 32 trajs, symmetric tensor
    # dset_path = grit_data_dir() / "generated" / "20220323_165127_trappist_b_T2_sym"

    # # Toy T2. output gap = 1, 32 trajs. Symmetric.
    # dset_path = grit_data_dir() / "generated/toy/20220326_003703_toy_T2"

    # Toy3 T2. output gap = 1, 32 trajs. Symmetric.
    dset_path = grit_data_dir() / "generated/toy3/20220328_173843_toy3_T2"

    # # Toy3 T2. output gap = 1, 64 trajs. Symmetric.
    # dset_path = grit_data_dir() / "generated/toy3/20220328_213518_toy3_T2"

    return dset_path


def get_dset(scale: int = 1, remake_dset: bool = False) -> Tuple[PlanetsDSet, PlanetsDSet]:
    dset_path = get_dset_path()
    dset_col = PlanetsDSetCol(dset_path)
    T = 49

    # scalings = [1, 2, 10, 100, 500, 1000]
    scalings = [1, 2, 10]
    assert scale in scalings

    # 1: Get dataset.
    if remake_dset or not all(dset_col.has_scaling(_scale) for _scale in scalings):
        if remake_dset:
            log.info("Remaking dset due to --remake-dset flag!")
        log.info("Splitting data....")
        dset_col.split_datas(T, scalings)

    log.info("Loading scale={}...".format(scale))
    train_dset, val_dset, _, scaling = dset_col.load_scaling(scale)
    log.info(
        "n_data: {}, n_train_trajs: {}, train_batch={}, bodies={}, dt={}, n_steps={}, scaling={}".format(
            dset_col.n_data,
            train_dset.n_unique_trajs,
            train_dset.batch,
            dset_col.names,
            dset_col.dt,
            dset_col.n_steps,
            scaling,
        )
    )

    return train_dset, val_dset


def get_log_dir(run_name: Optional[str], run_type: str) -> pathlib.Path:
    from datetime import datetime

    now = datetime.now()

    time_str = get_time_stamp(now)
    dt_str = get_date_stamp(now)

    if run_name is None:
        run_name = time_str
    else:
        run_name = f"{time_str}_{run_name}"

    return runs_dir() / run_type / dt_str / run_name


def get_model(
    n_bodies: int,
    rigid_V_coeff: float,
    final_weight_coeff: Optional[float] = None,
    analytic_mode: AnalyticMode = AnalyticMode.NoAnalytic,
) -> Tuple[TransformedSO3RNN, Callable[[], MultiSO3RNN]]:
    tanh = Activation.Tanh
    int_tanh = Activation.IntTanh
    swish = Activation.Swish

    # act = tanh
    act = swish

    learn_Vx = False
    dRPi_mode = dRPiMode.Analytic
    # analytic_mode = AnalyticMode.AnalyticRigidFull
    # analytic_mode = AnalyticMode.AnalyticRigidTrace
    # analytic_mode = AnalyticMode.AnalyticPoint
    # analytic_mode = AnalyticMode.NoAnalytic

    if analytic_mode.is_analytic():
        log.info("=======================================================")
        log.info("=              ANALYTIC MODE IS ON!!!                 =")
        log.info("=======================================================")

    print("Scaling final weights init by {}".format(final_weight_coeff))

    point_V_cfg = MLPCfg(
        hidden=[32, 32, 32], act=[act, act, act], w_init_gain=1.0, final_weight_coeff=final_weight_coeff
    )
    h = 128
    # h = 256
    rigid_V_cfg = MLPCfg(hidden=[h, h, h], act=[act, act, act], w_init_gain=1.0, final_weight_coeff=final_weight_coeff)

    return get_so3rnn_model(
        point_V_cfg, rigid_V_cfg, n_bodies, learn_Vx, rigid_V_coeff, analytic_mode=analytic_mode, dRPi_mode=dRPi_mode
    )


def setup_common(run_name: Optional[str], run_type: str) -> pathlib.Path:
    jax_use_double()

    # 1. Setup logging.
    log_dir = get_log_dir(run_name, run_type)
    setup_logger(log_dir)

    # Log git info.
    log_git_info(log_dir)

    return log_dir


def run_pointmass(run_name: Optional[str], remake_dset: bool, exp: str) -> None:
    # 1: Setup logging.
    log_dir = setup_common(run_name, f"{exp}_pt")

    # 2: Get dset.
    train_dset, val_dset = get_dset(scale=1, remake_dset=remake_dset)

    # 3: Save the model first via cloudpickle.
    rigid_V_coeff = 0.0
    models = get_model(train_dset.n_bodies, rigid_V_coeff)
    model_path = log_dir / "model.pkl"
    with open(model_path, "wb") as f:
        cloudpickle.dump(models, f)
    log.info("Saved models to {}".format(model_path))
    model = models[0]

    # 4: Train.
    logger = get_default_logger(log_dir)
    fit_result = fit_pointmass(model, train_dset, val_dset, logger)

    # 5: Save the result.
    save_path = log_dir / "train_state.pkl"
    with open(save_path, "wb") as f:
        cloudpickle.dump(fit_result.train_state, f)

    log.info("Saved final train_state to {}!".format(save_path))


def run_rigid(
    run_name: Optional[str],
    exp: str,
    warmstart: Optional[WarmstartInfo],
    integrator: IntegratorEnum,
    all_args: Dict[str, Any],
) -> None:
    # 1: Setup logging.
    # log_dir = setup_common(run_name, "trappist_rigid")
    log_dir = setup_common(run_name, f"{exp}_rigid")

    log.info("------- All args ----------")
    for k, v in all_args.items():
        log.info("{} - {}".format(k, v))
    log.info("---------------------------")

    # port = 9980 + random.randint(0, 10)
    # server = jax.profiler.start_server(port)
    # log.info("Profiling at localhost:{}".format(port))

    # 2: Get dset.
    train_dset, val_dset = get_dset(scale=1, remake_dset=False)

    # 3: Save the model first via cloudpickle.
    rigid_V_coeff = 1e-2
    final_weight_coeff = 1e-2
    # final_weight_coeff = 1e-2
    # final_weight_coeff = 1e-20

    if run_name == "pt_baseline":
        analytic_mode = AnalyticMode.AnalyticPoint
        log.info("============================")
        log.info("   Analytic Point Baseline ")
        log.info("============================")
    elif run_name == "rigid_baseline":
        analytic_mode = AnalyticMode.AnalyticRigidFull
        log.info("============================")
        log.info("   Analytic Rigid Baseline  ")
        log.info("============================")
    else:
        analytic_mode = AnalyticMode.NoAnalytic

    models = get_model(train_dset.n_bodies, rigid_V_coeff, final_weight_coeff, analytic_mode)
    model_path = log_dir / "model.pkl"
    with open(model_path, "wb") as f:
        cloudpickle.dump(models, f)
    log.info("Saved models to {}".format(model_path))
    model = models[0]

    # 4: Train.
    logger = get_default_logger(log_dir)
    fit_result = fit_rigid(model, train_dset, val_dset, logger, warmstart, integrator)

    # 5: Save the result.
    save_path = log_dir / "train_state.pkl"
    with open(save_path, "wb") as f:
        cloudpickle.dump(fit_result.train_state, f)

    log.info("Saved final train_state to {}!".format(save_path))
