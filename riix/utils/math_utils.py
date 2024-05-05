"""math utility functions for rating systems"""
import math
import statistics
import numpy as np
from scipy.special import expit
from scipy.stats import norm


def sigmoid(x):
    """a little faster than implementing it in numpy for d < 100000"""
    return expit(x)


def sigmoid_scalar(x):
    """no need to use numpy on scalars"""
    return 1.0 / (1.0 + math.exp(-x))


def base_10_sigmoid(x):
    """some methods prefer base 10 unfortunately"""
    return 1.0 / (1.0 + (10.0**-x))


INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def norm_cdf(x):
    """cdf of standard normal"""
    return 0.5 * (1.0 + math.erf(x * INV_SQRT_2))


STANDARD_NORMAL = statistics.NormalDist()


def norm_pdf(x):
    """pdf of standard normal"""
    return STANDARD_NORMAL.pdf(x)


def v_and_w_win_vector(t, eps):
    """calculate v and w for a win in a vectorized fashion"""
    diff = t - eps
    v = np.empty_like(diff)
    denom = norm.cdf(diff)
    bad_mask = denom < 1e-50
    v[bad_mask] = -1.0 * diff[bad_mask]
    v[~bad_mask] = norm.pdf(diff[~bad_mask]) / denom[~bad_mask]
    w = v * (v + diff)
    return v, w


def v_and_w_draw_vector(t, eps):
    """calculate v and w for a draw in a vectorized fashion"""
    abs_t = np.abs(t)  # the papers do NOT do this but ALL open source implementations DO...
    diff_a = eps - abs_t
    diff_b = -eps - abs_t

    # TODO maybe this would be faster if I concatenated and put through pdf/cdf together?
    pdf_a = norm.pdf(diff_a)
    pdf_b = norm.pdf(diff_b)
    cdf_a = norm.cdf(diff_a)
    cdf_b = norm.cdf(diff_b)

    v_num = pdf_b - pdf_a
    shared_denom = cdf_a - cdf_b
    bad_mask = shared_denom < 1e-5
    good_mask = ~bad_mask

    v = np.empty_like(t)
    if eps.shape != t.shape:
        eps = np.repeat(eps, repeats=2, axis=1)
    v[bad_mask] = -t[bad_mask] + (np.sign(t[bad_mask]) * eps[bad_mask])
    v[good_mask] = v_num[good_mask] / shared_denom[good_mask]

    w = np.empty_like(t)
    w_bad_mask = np.isnan(shared_denom) | np.isinf(shared_denom) | (shared_denom < 1e-50)
    w_good_mask = ~w_bad_mask
    w[w_bad_mask] = 1.0

    w_num = (diff_a[w_good_mask] * pdf_a[w_good_mask]) - (diff_b[w_good_mask] * pdf_b[w_good_mask])
    w[w_good_mask] = (w_num / shared_denom[w_good_mask]) + np.square(v[w_good_mask])
    return v, w


def v_and_w_win_scalar(t, eps):
    """calculate v and w for a win in a scalar fashion"""
    diff = t - eps
    cdf = norm_cdf(diff)
    if cdf > 2.222758749e-162:
        v = norm_pdf(diff) / cdf
    else:
        v = -diff
    w = v * (v + diff)
    return v, w


def v_and_w_draw_scalar(t, eps):
    """calculate v and w for a draw in a scalar fashion"""
    abs_t = math.fabs(t)  # the papers do NOT do this but ALL open source implementations DO...
    diff_a = eps - abs_t
    diff_b = -eps - abs_t

    cdf_a = norm_cdf(diff_a)
    cdf_b = norm_cdf(diff_b)

    pdf_a = norm_pdf(diff_a)
    pdf_b = norm_pdf(diff_b)
    v_num = pdf_a - pdf_b
    shared_denom = cdf_a - cdf_b
    sign = math.copysign(1.0, t)
    if shared_denom < 1e-5:
        v = -t + (sign * eps)
    else:
        v = sign * v_num / shared_denom
    if shared_denom < 1e-50:
        w = 1.0
    else:
        w_num = (diff_a * pdf_a) - (diff_b * pdf_b)
        w = math.copysign(1.0, t) * ((w_num / shared_denom) + (v**2.0))
    return v, w
