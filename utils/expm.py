import jax.numpy as jnp
from jax import lax
from jax._src.numpy import linalg as np_linalg
from jax.scipy.linalg import solve_triangular


def expm(A, *, upper_triangular=False, max_squarings=16):
    P, Q, n_squarings = _calc_P_Q(A)

    def _inf(args):
        A, *_ = args
        return jnp.full_like(A, jnp.inf)

    def _compute(args):
        A, P, Q = args
        R = _solve_P_Q(P, Q, upper_triangular)
        R = _squaring(R, n_squarings, max_squarings)
        return R

    R = lax.cond(n_squarings > max_squarings, _inf, _compute, (A, P, Q))
    return R


def _calc_P_Q(A):
    A = jnp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected A to be a square matrix")
    A_L1 = np_linalg.norm(A, 1)
    n_squarings = 0
    if A.dtype == "float64" or A.dtype == "complex128":
        maxnorm = 5.371920351148152
        n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
        A = A / 2 ** n_squarings
        conds = jnp.array(
            [1.495585217958292e-002, 2.539398330063230e-001, 9.504178996162932e-001, 2.097847961257068e000]
        )
        idx = jnp.digitize(A_L1, conds)
        U, V = lax.switch(idx, [_pade3, _pade5, _pade7, _pade9, _pade13], A)
    elif A.dtype == "float32" or A.dtype == "complex64":
        maxnorm = 3.925724783138660
        n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
        A = A / 2 ** n_squarings
        conds = jnp.array([4.258730016922831e-001, 1.880152677804762e000])
        idx = jnp.digitize(A_L1, conds)
        U, V = lax.switch(idx, [_pade3, _pade5, _pade7], A)
    else:
        raise TypeError("A.dtype={} is not supported.".format(A.dtype))
    P = U + V  # p_m(A) : numerator
    Q = -U + V  # q_m(A) : denominator
    return P, Q, n_squarings


def _solve_P_Q(P, Q, upper_triangular=False):
    if upper_triangular:
        return solve_triangular(Q, P)
    else:
        return np_linalg.solve(Q, P)


def _precise_dot(A, B):
    return jnp.dot(A, B, precision=lax.Precision.HIGHEST)


def _squaring(R, n_squarings, max_squarings):
    # squaring step to undo scaling
    def _squaring_precise(x):
        return _precise_dot(x, x)

    def _identity(x):
        return x

    def _scan_f(c, i):
        return lax.cond(i < n_squarings, _squaring_precise, _identity, c), None

    res, _ = lax.scan(_scan_f, R, jnp.arange(max_squarings))

    return res


def _pade3(A):
    b = (120.0, 60.0, 12.0, 1.0)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = _precise_dot(A, A)
    U = _precise_dot(A, (b[3] * A2 + b[1] * ident))
    V = b[2] * A2 + b[0] * ident
    return U, V


def _pade5(A):
    b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = _precise_dot(A, A)
    A4 = _precise_dot(A2, A2)
    U = _precise_dot(A, b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V


def _pade7(A):
    b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = _precise_dot(A, A)
    A4 = _precise_dot(A2, A2)
    A6 = _precise_dot(A4, A2)
    U = _precise_dot(A, b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V


def _pade9(A):
    b = (17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = _precise_dot(A, A)
    A4 = _precise_dot(A2, A2)
    A6 = _precise_dot(A4, A2)
    A8 = _precise_dot(A6, A2)
    U = _precise_dot(A, b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V


def _pade13(A):
    b = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = _precise_dot(A, A)
    A4 = _precise_dot(A2, A2)
    A6 = _precise_dot(A4, A2)
    U = _precise_dot(
        A, _precise_dot(A6, b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
    )
    V = _precise_dot(A6, b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V
