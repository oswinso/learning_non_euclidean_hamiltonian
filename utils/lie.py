import math

import ipdb
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from utils.jax_utils import block
from utils.types import ja


def hat(x: ja) -> ja:
    """hat operator, i.e., maps R^3 to so(3) (skew symmetric matrix).
    :param x: (..., 3)
    :return: (...., 3, 3)
    """
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    z = jnp.zeros_like(a)

    out = block(
        [
            [z, -c, b],
            [c, z, -a],
            [-b, a, z],
        ]
    )
    return out


def vee(X: ja) -> ja:
    """vee operator, i.e., maps from so(3) (skew symmetric matrix) to R^3.
    :param X: (..., 3, 3)
    :return: (..., 3)
    """
    c = -X[..., 0, 1]
    b = X[..., 0, 2]
    a = -X[..., 1, 2]

    return jnp.stack([a, b, c], axis=-1)


def clamp_acos(x: ja, eps: float = 1e-80) -> ja:
    thresh = 1.0 - eps
    clipped_x = x.clip(-thresh, thresh)

    return jnp.arccos(clipped_x)


def one_sided_acos(x: ja, eps: float = 1e-3) -> ja:
    # For x ~ +1, clip. This is because acos(1) = 0, and so there is basically no error.
    # For x ~ -1, do linear approximation. This is beacuse acos(-1) = -pi, which is basically max error so we want to
    # keep the gradient.
    lin_thresh = 1.0 - eps
    abs_x = jnp.abs(x)
    safe_x = x > -lin_thresh

    sign = jnp.sign(x)
    one_minus_thresh_sq = 2 * eps - eps ** 2
    slope = -1.0 / np.sqrt(one_minus_thresh_sq)

    clipped_x = x.clip(-1.0, 1.0 - 1e-20)
    lin_approx = (x < 0) * jnp.pi + sign * (jnp.arccos(lin_thresh) + slope * (abs_x - lin_thresh))

    return jnp.where(safe_x, jnp.arccos(clipped_x), lin_approx)


def safe_acos(x: ja, eps: float = 1e-17) -> ja:
    thresh = 1.0 - eps
    abs_x = jnp.abs(x)
    safe_x = abs_x < thresh

    sign = jnp.sign(x)
    one_minus_thresh_sq = 2 * eps - eps ** 2
    slope = -1.0 / np.sqrt(one_minus_thresh_sq)

    clipped_x = x.clip(-thresh, thresh)
    lin_approx = (x < 0) * jnp.pi + sign * (jnp.arccos(thresh) + slope * (abs_x - thresh))

    return jnp.where(safe_x, jnp.arccos(clipped_x), lin_approx)


def cos_geodesic(R1: ja, R2: ja) -> ja:
    # R: (..., 3, 3),
    assert R1.shape[-2:] == (3, 3) and R2.shape[-2:] == (3, 3)

    R1_T = R1.swapaxes(-1, -2)
    R1_T_R2 = R1_T @ R2

    tr_rot = R1_T_R2[..., 0, 0] + R1_T_R2[..., 1, 1] + R1_T_R2[..., 2, 2]
    cos_angle = (tr_rot - 1) / 2

    return cos_angle


def geodesic(R1: ja, R2: ja) -> ja:
    cos_angle = cos_geodesic(R1, R2)

    return safe_acos(cos_angle, 1e-8)
    # return one_sided_acos(cos_angle)


def RotZ(theta: ja) -> ja:
    """
    :param theta: (..., )
    :return: (..., 3, 3)
    """
    s, c = jnp.sin(theta), jnp.cos(theta)
    z, one = jnp.zeros_like(theta), jnp.ones_like(theta)

    out = block(
        [
            [c, -s, z],
            [s, c, z],
            [z, z, one],
        ]
    )
    return out


def RotAxisAngle(axis: ja, angle: ja) -> ja:
    """
    :param axis: (..., 3)
    :param angle: (..., )
    :return: (..., 3, 3)
    """
    norm = jnp.linalg.norm(axis, axis=-1, keepdims=True)
    zero_norm = norm == 0.0
    clipped_norm = jnp.where(zero_norm, 1.0, norm)

    unit_vec = jnp.where(zero_norm, axis, axis / clipped_norm)

    x, y, z = unit_vec[..., 0], unit_vec[..., 1], unit_vec[..., 2]

    C, S = jnp.cos(angle), jnp.sin(angle)
    t = 1 - C

    out = block(
        [
            [t * x * x + C, t * x * y - S * z, t * x * z + S * y],
            [t * x * y + S * z, t * y * y + C, t * y * z - S * x],
            [t * x * z - S * y, t * y * z + S * x, t * z * z + C],
        ]
    )
    return out


def to_xzx_angles(R: ja) -> ja:
    # X(phi), Z(theta), X(psi).
    # R: (..., 3, 3)
    a, b, c = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    d, e, f = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    g, h, j = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    kEps = 1e-15

    mask1 = jnp.abs(jnp.abs(a) - 1.0) > kEps
    theta1 = jnp.arccos(a.clip(-1.0 + kEps, 1.0 - kEps))
    sin = jnp.sin(theta1)
    psi1 = jnp.arctan2(c / sin, -b / sin)
    phi1 = jnp.arctan2(g / sin, d / sin)

    psi23 = 0
    mask2 = jnp.abs(a - 1.0) < kEps
    theta2 = 0
    phi2 = jnp.arctan2(h, e) - psi23

    theta3 = jnp.pi
    phi3 = psi23 + jnp.arctan2(h, -e)

    theta = jnp.where(mask1, theta1, jnp.where(mask2, theta2, theta3))
    psi = jnp.where(mask1, psi1, psi23)
    phi = jnp.where(mask1, phi1, jnp.where(mask2, phi2, phi3))

    return jnp.stack([phi, theta, psi], axis=-1)


def to_xyz_angles(R: ja) -> ja:
    # R: (..., 3, 3)
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    kEps = 1e-15

    mask1 = r20 < (1 - kEps)
    mask2 = r20 > (-1 + kEps)

    y1 = jnp.arcsin(-r20)
    z1 = jnp.arctan2(r10, r00)
    x1 = jnp.arctan2(r21, r22)

    y2 = jnp.pi / 2
    z2 = -jnp.arctan2(-r12, r11)
    x2 = 0.0

    y3 = -jnp.pi / 2
    z3 = z2
    x3 = 0.0

    x = jnp.where(mask1, jnp.where(mask2, x1, x2), x3)
    y = jnp.where(mask1, jnp.where(mask2, y1, y2), y3)
    z = jnp.where(mask1, jnp.where(mask2, z1, z2), z3)

    return jnp.stack([x, y, z], axis=-1)


# --------------------------------------------------------------------------------------------------------


def hat_so2(theta: ja) -> ja:
    """
    :param theta:
    :return: (2, 2)
    """
    z = jnp.zeros_like(theta)
    out = block([[z, -theta], [theta, z]])
    return out


def vee_so2(Omega: ja) -> ja:
    """vee operator in so2.
    :param Omega: (2, 2)
    :return: (, )
    """
    return Omega[..., 1, 0]


def exp_so2(theta: ja) -> ja:
    """expmat( hat( theta ) ), i.e. just return the rotation matrix.
    :param theta: (..., )
    :return: (..., 2, 2)
    """
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    out = block([[c, -s], [s, c]])
    return out


def log_so2(Rs: ja) -> ja:
    c = Rs[..., 0, 0]
    s = Rs[..., 1, 0]
    return jnp.arctan2(s, c)


def make_cts(thetas: ja) -> ja:
    assert thetas.ndim == 1
    dphi = _diff(thetas)
    dphi_m = ((dphi + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    dphi_m = dphi_m.at[(dphi_m == -jnp.pi) & (dphi > 0)].set(jnp.pi)
    phi_adj = dphi_m - dphi
    phi_adj = phi_adj.at[jnp.abs(dphi) < jnp.pi].set(0)

    return thetas + jnp.cumsum(phi_adj)


def _diff(x: ja) -> ja:
    return jnp.pad(x[..., 1:] - x[..., :-1], (1, 0))
