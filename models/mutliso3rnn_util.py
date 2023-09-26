import logging
import math

import jax.numpy as jnp

from utils.types import ja

# from rich import print


log = logging.getLogger(__file__)

kG = 4.0 * math.pi ** 2


def rigid_correction(q: ja, R: ja, mass: ja, Js: ja, rigid_trace: bool, rigid_full: bool) -> ja:
    # q: (n_bodies, 3)
    # Js: (n_bodies, 3, 3)
    n_bodies, _ = q.shape

    assert rigid_trace or (rigid_trace and rigid_full)

    trace_Js = Js[..., 0, 0] + Js[..., 1, 1] + Js[..., 2, 2]
    assert trace_Js.shape == (n_bodies,)

    R_T = R.swapaxes(-2, -1)
    RJR = R @ Js @ R_T
    assert RJR.shape == (n_bodies, 3, 3)

    E = 0
    for ii in range(1, n_bodies):
        for jj in range(ii):
            qi_qj = q[ii] - q[jj]
            div2 = jnp.sum(qi_qj ** 2)
            div = jnp.sqrt(div2)

            tmp = 0

            div3 = div2 * div
            div5 = div3 * div2

            if rigid_trace:
                # i rigid
                tmp = tmp - trace_Js[ii] * mass[jj] / (2.0 * div3)
                # j rigid
                tmp = tmp - trace_Js[jj] * mass[ii] / (2.0 * div3)

            if rigid_full:
                mat = jnp.zeros((3, 3))
                mat = mat + RJR[ii] * mass[jj]
                mat = mat + RJR[jj] * mass[ii]
                tmp = tmp + (1.5 / div5) * jnp.dot(qi_qj, mat @ qi_qj)

            E = E + tmp

    return kG * E


def analytical_V(q: ja, R: ja, mass: ja, Js: ja, rigid_trace: bool, rigid_full: bool) -> ja:
    # q: (n_bodies, 3)
    # Js: (n_bodies, 3, 3)
    n_bodies, _ = q.shape

    trace_Js = Js[..., 0, 0] + Js[..., 1, 1] + Js[..., 2, 2]
    assert trace_Js.shape == (n_bodies,)

    R_T = R.swapaxes(-2, -1)
    RJR = R @ Js @ R_T
    assert RJR.shape == (n_bodies, 3, 3)

    E = 0
    for ii in range(1, n_bodies):
        for jj in range(ii):
            qi_qj = q[ii] - q[jj]
            div2 = jnp.sum(qi_qj ** 2)
            div = jnp.sqrt(div2)

            tmp = -mass[ii] * mass[jj] / div

            if rigid_trace or rigid_full:
                div3 = div2 * div
                div5 = div3 * div2

                mat = jnp.zeros((3, 3))

                # i rigid
                tmp = tmp - trace_Js[ii] * mass[jj] / (2.0 * div3)

                # j rigid
                tmp = tmp - trace_Js[jj] * mass[ii] / (2.0 * div3)

                # Only include the rotation dependent terms for rigid_full.
                if rigid_full:
                    mat = mat + RJR[ii] * mass[jj]
                    mat = mat + RJR[jj] * mass[ii]
                    tmp = tmp + (1.5 / div5) * jnp.dot(qi_qj, mat @ qi_qj)
            E = E + tmp

    return kG * E


def tostr(arr) -> str:
    s = ["{:12.5e}".format(n) for n in arr.flatten()]
    s = ", ".join(s)
    return "[{}]".format(s)


def analytic_dVdq(q: ja, R: ja, mass: ja, Js: ja, rigid_trace: bool, rigid_full: bool) -> ja:
    # q: (n_bodies, 3)
    n_bodies, _ = q.shape

    trace_Js = Js[..., 0, 0] + Js[..., 1, 1] + Js[..., 2, 2]
    assert trace_Js.shape == (n_bodies,)

    R_T = R.swapaxes(-2, -1)
    RJR = R @ Js @ R_T
    assert RJR.shape == (n_bodies, 3, 3)

    # for ii in range(n_bodies):
    #     print(
    #         "Body {} - R={}\n         I={}\n       RIR={}\n".format(ii, tostr(R[ii]), tostr(Js[ii]), tostr(RJR[ii]))
    #     )
    #
    ret = jnp.zeros((n_bodies, 3))

    for ii in range(n_bodies):
        for jj in range(n_bodies):
            if ii == jj:
                continue

            qi_qj = q[ii] - q[jj]
            div2 = jnp.sum(qi_qj ** 2)
            div = jnp.sqrt(div2)
            div3 = div2 * div
            div5 = div3 * div2
            div7 = div5 * div2

            tmp = mass[ii] * mass[jj] / div3
            # print("    Body {} - tmp={}".format(ii, tmp))

            if rigid_trace or rigid_full:
                tmp = tmp + 1.5 / div5 * (mass[jj] * trace_Js[ii] + mass[ii] * trace_Js[jj])

                # The terms below involve R.
                if rigid_full:
                    # print("    Body {} - tmp={}".format(ii, tmp))
                    tmp = tmp - jnp.dot(qi_qj, (mass[jj] * RJR[ii] + mass[ii] * RJR[jj]) @ qi_qj) * (7.5 / div7)
                    # print("    Body {} - tmp={}, RIR[i]={}, RIR[j]={}".format(ii, tmp, tostr(RJR[ii]), tostr(RJR[jj])))

                    term = (mass[jj] * RJR[ii] + mass[ii] * RJR[jj]) @ qi_qj * 3.0 / div5
                    ret = ret.at[ii].add(term)
                    # print("    Body {} - ret={}".format(ii, tostr(ret[ii])))

            assert tmp.shape == tuple()

            ret = ret.at[ii].add(tmp * qi_qj)
            # print("    Body {} - qi_qj={}, tmp={}, ret={}\n".format(ii, tostr(qi_qj), tmp, tostr(ret[ii])))

    return kG * ret


def analytic_dVdR(q: ja, R: ja, mass: ja, Js: ja, rigid_trace: bool, rigid_body: bool) -> ja:
    # q: (n_bodies, 3)
    n_bodies, _ = q.shape

    for arr in [q, R, mass, Js]:
        assert arr.dtype == jnp.float64

    # dVdR = 0 for point mass.
    if not rigid_body:
        return jnp.zeros((n_bodies, 3, 3))

    ret = jnp.zeros((n_bodies, 3, 3))

    for ii in range(n_bodies):
        for jj in range(n_bodies):
            if ii == jj:
                continue

            # (3, )
            qi_qj_m = q[ii] - q[jj]
            div2 = jnp.sum(qi_qj_m ** 2)
            div = jnp.sqrt(div2)
            div5 = div2 * div2 * div

            # (3, 1)
            qi_qj = jnp.expand_dims(qi_qj_m, axis=1)
            qi_qj_T = qi_qj.swapaxes(0, 1)

            term = mass[jj] * 3.0 / div5 * qi_qj @ qi_qj_T
            ret = ret.at[ii].add(term)

        fkfk = R[ii] @ Js[ii]
        fku = ret[ii] @ fkfk
        term = kG * fku
        ret = ret.at[ii].set(term)

    return ret
