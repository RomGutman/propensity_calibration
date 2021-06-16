import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm


def get_variables(mean_, std_, n_, m_=1, noise_mean_=1, noise_std_=0):
    """ Generates normal distributed random variables for simulations.

    Draws M random variables from normal distribution, for N instances with given mean and std.
    Also draws N noise variables with given mean and std (possibly different from the variables).

    Args:
        mean_:
        std_:
        n_:
        m_:
        noise_mean_:
        noise_std_:

    Returns:

    """
    variables_ = np.random.normal(loc=mean_, scale=std_ ^ 2, size=[n_, m_])
    noise_ = np.random.normal(loc=noise_mean_, scale=noise_std_ ^ 2, size=n_)
    return variables_, noise_


def expit_transform(coef_, variables_):
    """

    Args:
        coef_:
        variables_:

    Returns:

    """
    exp = np.dot(coef_, np.transpose(variables_))
    return expit(exp)


def treatment_assigment(prop):
    """

    Args:
        prop:

    Returns:

    """
    return np.random.binomial(n=1, p=prop)


def get_potential_outcomes(variables_, coef_, noise_):
    """

    Args:
        variables_:
        coef_:
        noise_:

    Returns:

    """
    y_vars = np.concatenate((variables_, noise_[:, None]), axis=1)
    y0_vars = np.concatenate((np.zeros_like(noise_[:, None]), y_vars), axis=1)
    y1_vars = np.concatenate((np.ones_like(noise_[:, None]), y_vars), axis=1)
    return np.dot(coef_, y0_vars.transpose()), np.dot(coef_, y1_vars.transpose())


def calc_ipw(y, t, prop):
    y1 = np.mean(y*t / prop)
    y0 = np.mean(y*(1-t) / (1 - prop))
    return y1 - y0


def generate_calib_error_df(t, prop, idx=0):
    p_calib = lowess(t, prop, return_sorted=False)
    err = np.abs(prop - p_calib)
    err_s = pd.Series(err)

    brier_score = brier_score_loss(t, prop)
    brier = pd.Series(brier_score, index=['brier'])

    calib_res_df = (err_s
                    .describe(percentiles=[.1, .5, .9])
                    .append(brier)
                    .to_frame(idx).T
                    )
    return calib_res_df


def generate_simulation(m, mean, std, n, noise_mean, noise_std, coef, y_coef, num_of_experiments=1,
                        prop_func=expit_transform):
    """

    Args:
        m:
        mean:
        std:
        n:
        noise_mean:
        noise_std:
        coef:
        y_coef:
        num_of_experiments:
        prop_func:

    Returns:

    """
    variables, noise = get_variables(mean_=mean, std_=std, n_=n*num_of_experiments, m_=m, noise_mean_=noise_mean,
                                     noise_std_=noise_std)
    err_df_list = []
    for i in tqdm(range(num_of_experiments)):
        rel_vars = variables[i * n: (i + 1) * n]
        rel_noise = noise[i * n: (i + 1) * n]
        prop = prop_func(coef, rel_vars)
        t = treatment_assigment(prop)

        potential_outcomes_lst = get_potential_outcomes(rel_vars, y_coef, rel_noise)
        y = np.where(t == 1, potential_outcomes_lst[1], potential_outcomes_lst[0])
        ate_hat = calc_ipw(y, t, prop)
        err_df_list.append(generate_calib_error_df(t, prop, i).assign(ATE=ate_hat))
    return pd.concat(err_df_list)
