import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from utils_stats import (agresti_coull_interval_min_max,
                         CI_FRACTION,
                         CI_TYPE,
                         get_success_rates,
                         successes_failures_to_ci_min_max,
                         SUCCESS_RATE_BOUNDARY,
                         test_value,
                         true_rate_posterior,
                         Z_QUANTILE_ERRORRATE
                         )


def plot_success_rates(success, failure, ci_fraction=CI_FRACTION, ci_type=CI_TYPE,
                       min_psuccess=0.85, max_psucess=1.,d_psuccess=0.0001,
                       color="purple", format='-', label=None, fill=False, display_ci=True,
                       alpha=1., factor=1., ci_label=None,
                       xlabel="success rate",
                       ylabel="probability distribution function"):

    ci_min, ci_max = successes_failures_to_ci_min_max(success, failure, ci_type=ci_type, ci_fraction=ci_fraction)

    _, p_success = get_success_rates(d_success = d_psuccess, min_range=0., max_range=1., including_max=True)
    beta_pdf = beta.pdf(p_success, test_value(success), test_value(failure))

    plt.plot(p_success, beta_pdf * factor, format, linewidth=3, color=color, label=label, alpha=alpha)

    xmin = np.max([p_success.min(), ci_min * 0.95])
    xmax = np.min([p_success.max(), ci_max * 1.05])

    plt.xlim(xmin, xmax)

    if display_ci == True:
        plt.plot([ci_min, ci_min],
                 [0, beta.pdf(ci_min, success, failure) * factor],
                 "--", color=color, alpha=alpha, label=ci_label)
        plt.plot([ci_max, ci_max],
                 [0, beta.pdf(ci_max, success, failure) * factor],
                 "--", color=color, alpha=alpha)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)


def plot_success_rates_methods(audit_success_rate, audit_size, ci_fraction, ci_type="HDI", legend_title="Interval Algorithm", ac_display=True, display_ci=True):
    plot_success_rates(audit_success_rate * audit_size,
                     (1 - audit_success_rate) * audit_size,
                     ci_fraction=ci_fraction, ci_label=f"Jeffreys {ci_fraction * 100.:0.1f}% CI", label="Posterior", ci_type=ci_type,
                       xlabel="model success rate", display_ci=display_ci)

    # Agresti-Coull
    if ac_display:
        ac_min, ac_max = agresti_coull_interval_min_max(audit_success_rate, audit_size, z=Z_QUANTILE_ERRORRATE)
        plt.scatter([ac_min, ac_max], [0, 0], s=100., marker="^", color="orange", label=f"Agresti-Coull {ci_fraction * 100.:0.1f}% CI")

    plt.legend(title=legend_title)
    plt.title(f"Audit success rate {audit_success_rate * 100:0.0f}%, Audit Size={audit_size:,}")

    ax = plt.gca()
    #grid
    ax.grid(alpha=0.3)


def plot_boundary_true_false_positive_rates(observed_success_rate, sample_size, success_rate_boundary=SUCCESS_RATE_BOUNDARY, generator_success_rates=None, postior_method ="flat_min"):

    if generator_success_rates is None:
        d_generator_success, generator_success_rates = get_success_rates()

    model_success_pdf = true_rate_posterior(observed_success_rate, sample_size,
                                            generator_success_rates, min_ab=0.5,
                                            method=postior_method)

    positive_bools = generator_success_rates > success_rate_boundary
    true_positive_rate = np.sum(
        (positive_bools).astype(int) * model_success_pdf) * d_generator_success

    # visualising
    idxs_positive = [success_rate for idx, success_rate in
                     enumerate(generator_success_rates) if positive_bools[idx]]
    pdf_positive = [pdf_ for idx, pdf_ in enumerate(model_success_pdf) if
                    positive_bools[idx]]

    idxs_negative = [success_rate for idx, success_rate in
                     enumerate(generator_success_rates) if ~positive_bools[idx]]
    pdf_negative = [pdf_ for idx, pdf_ in enumerate(model_success_pdf) if
                    ~positive_bools[idx]]

    plt.plot(generator_success_rates, model_success_pdf, color="purple", linewidth=3)
    plt.fill_between(idxs_positive, pdf_positive, color='lightgreen', alpha=0.4,
                     label=f"True Positive {true_positive_rate * 100:0.1f}%")
    plt.fill_between(idxs_negative, pdf_negative, color='darkred', alpha=0.4,
                     label=f"False Positive {(1. - true_positive_rate) * 100:0.1f}%",
                     hatch="x")

    plt.title(
        f"audit of size {sample_size:,} success rate {observed_success_rate * 100:0.1f}%")
    plt.xlabel("model success rate")
    plt.ylabel("probability distribution function")

    d_plot = 0.1
    plt.xlim([observed_success_rate - d_plot, np.min([observed_success_rate + d_plot, 1.])])
    plt.legend()
    ax = plt.gca()

    ylim = ax.get_ylim()
    plt.vlines(success_rate_boundary, 0, ylim[-1], color="gray", linestyle="--",
               linewidth=4, alpha=0.5)