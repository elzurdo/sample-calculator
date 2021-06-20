import numpy as np

from scipy.optimize import fmin
from scipy.stats import beta

CI_FRACTION = 0.95
CI_TYPE = "HDI"
Z_QUANTILE_ERRORRATE = 1.96
MIN_COUNTS = 1.
SUCCESS_RATE_BOUNDARY =0.93


def agresti_coull_interval(p, n, z=Z_QUANTILE_ERRORRATE):
    X = p * n  # success

    n_ac = n + z ** 2
    p_ac = 1. / n_ac * (X + 0.5 * z ** 2)
    ci_ac = z * np.sqrt(p_ac * (1 - p_ac) / n_ac)

    return p_ac, ci_ac


def agresti_coull_interval_min_max(p, n, z=Z_QUANTILE_ERRORRATE):
    p_ac, ci_ac = agresti_coull_interval(p, n, z=z)

    return p_ac - ci_ac, p_ac + ci_ac


def HDIofICDF(dist_name, ci_fraction=CI_FRACTION, **args):
    """
    This program finds the HDI of a probability density function that is specified
    mathematically in Python.

    Example usage: HDIofICDF(beta, a=100, b=100)
    """
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for HDIlowTailPr
    incredMass =  1.0 - ci_fraction


    def intervalWidth(lowTailPr):
        return distri.ppf(ci_fraction + lowTailPr) - distri.ppf(lowTailPr)

    # find lowTailPr that minimizes intervalWidth
    HDIlowTailPr = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return distri.ppf([HDIlowTailPr, ci_fraction + HDIlowTailPr])


def test_value(x):
    if x == 0:
        return MIN_COUNTS
    else:
        return x


def equal_tail_credible_interval(a, b, ci_fraction = CI_FRACTION, test=False):
    if test:
        a_ = test_value(a)
        b_ = test_value(b)
    else:
        a_ = a
        b_ = b

    buffer = 0.5 * (1. - ci_fraction)

    b_up = btdtri(a_, b_, 1. - buffer)
    b_lo = btdtri(a_, b_, buffer)

    return b_lo, b_up

def successes_failures_to_ci_min_max(a, b, ci_type=CI_TYPE, ci_fraction=CI_FRACTION, test=False):
    assert ci_type in ["ET", "HDI"]

    if "HDI" == ci_type:
        ci_min, ci_max = HDIofICDF(beta, a=a, b=b, ci_fraction=ci_fraction)
    else:
        ci_min, ci_max = equal_tail_credible_interval(a, b, ci_fraction=ci_fraction, test=test)

    return ci_min, ci_max

def get_success_rates(d_success = 0.00001, min_range=0., max_range=1., including_max=False):
    # d_success determines how accurate the audit_success_rate=99% results are. more zeros: more accurrate but slower

    max_range_ = max_range
    if including_max: # when max_range = 1. one must be cautious about beta going to infinity when failure = 0
        max_range_ += d_success

    success_rates = np.arange(min_range, max_range_, d_success)

    return d_success, success_rates


def true_rate_posterior(success_rate, size, model_success_rates, min_ab=0.5,
                        method="jeffreys"):
    assert method in ["flat_min", "jeffreys", "flat"]

    a = success_rate * size
    b = (1 - success_rate) * size

    if "flat_min" == method:
        if a < min_ab or b < min_ab:
            a += min_ab
            b += min_ab

        #a = np.max([success_rate * size, min_ab])
        #b = np.max([(1 - success_rate) * size, min_ab])  # this causes 0.99 area to act weird, because it secures a min of 1. perhaps set to 0.01?
    elif "jeffreys" == method:
        a += min_ab
        b += min_ab
    #print(a, b)

    generator_success_pdf = beta.pdf(model_success_rates, a, b)

    return generator_success_pdf


def sample_to_tpr(observed_success_rate, sample_size, generator_success_rates=None, success_rate_boundary=SUCCESS_RATE_BOUNDARY,
                  min_ab=0.5, method="flat_min"):

    if generator_success_rates is not None:
        d_generator_success = generator_success_rates[1] - generator_success_rates[0]
    else:
        d_generator_success, generator_success_rates = get_success_rates()

    generator_success_pdf = true_rate_posterior(observed_success_rate, sample_size, generator_success_rates,
                                            min_ab=min_ab, method=method)

    positive_bools = generator_success_rates > success_rate_boundary

    true_positive_rate = np.sum(
        (positive_bools).astype(int) * generator_success_pdf) * d_generator_success

    return true_positive_rate


def observed_success_correctness(observed_success_rate, sample_size, generator_success_rates=None,
                                 success_rate_boundary=SUCCESS_RATE_BOUNDARY,
                                 tpr_method="flat_min", min_ab=0.5):
    true_positive_rate = sample_to_tpr(observed_success_rate, sample_size,
                                       generator_success_rates, success_rate_boundary=success_rate_boundary,
                                       min_ab=min_ab, method=tpr_method)

    is_success = observed_success_rate > success_rate_boundary

    if is_success:
        # returns TPR/FPR
        rates = {"true_rate": true_positive_rate,
                 "false_rate": 1. - true_positive_rate,
                 "rate_type": "success"}
    else:
        # returns TNR = 1 - TPR
        rates = {"true_rate": 1. - true_positive_rate,
                 "false_rate": true_positive_rate,
                 "rate_type": "failure"}

    return rates


# TODO - consider depricating
def multiple_observed_successes_correctness(sample_size, all_observed_success_rates, generator_success_rates=None,
                                            success_rate_boundary=SUCCESS_RATE_BOUNDARY,
                                            tpr_method="flat_min", min_ab=0.5):
    success_correctness = {}

    for observed_success_rate in all_observed_success_rates:
        rates = observed_success_correctness(observed_success_rate, sample_size,
                                     generator_success_rates=generator_success_rates,
                                     success_rate_boundary=success_rate_boundary,
                                     tpr_method=tpr_method, min_ab=min_ab)

        success_correctness[observed_success_rate] = rates

    return success_correctness