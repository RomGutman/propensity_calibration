import warnings
import pickle


import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from sklearn.calibration import _sigmoid_calibration, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


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


def expit_transform(coef_, variables_, exp_scale=1, func_scale=1):
    """

    Args:
        coef_:
        variables_:

    Returns:

    """
    exp = np.dot(coef_, np.transpose(variables_))
    return expit(exp * exp_scale) * func_scale


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


def calc_ipw(y, t, prop, eps=1E-7):
    prop = np.clip(prop, eps, 1 - eps)
    y1 = np.mean(y * t / prop)
    y0 = np.mean(y * (1 - t) / (1 - prop))
    return y1 - y0


def get_sacle_from_name(type_):
    return eval(type_.split('_')[-1])


def generate_calib_error_df(t, prop, idx=0, type_=None, scale=None, calibration_type=None):
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
    if type_ is not None:
        calib_res_df['type'] = type_
        calib_res_df['scale'] = scale
        calib_res_df['calibration_type'] = calibration_type
    return calib_res_df


def generate_simulation(m, mean, std, n, noise_mean, noise_std, coef, y_coef, num_of_experiments=1,
                        phi_func=expit_transform, experiments=None, post_colab_func=None, save=False):
    """

    Args:
        save: Whether to save the simulation run to present later
        calibration_flag:
        phi_func:
        m:
        mean:
        std:
        n:
        noise_mean:
        noise_std:
        coef:
        y_coef:
        num_of_experiments:
        experiments:

    Returns:

    """
    variables, noise = get_variables(mean_=mean, std_=std, n_=n * num_of_experiments, m_=m, noise_mean_=noise_mean,
                                     noise_std_=noise_std)
    if experiments is None:
        warnings.warn("No experiment were given, performing for identity only")
        experiments = {'Identity': None}
    err_df_list = []
    saved_dict = {}
    orig_save = save
    for i in tqdm(range(num_of_experiments)):
        rel_vars = variables[i * n: (i + 1) * n]
        rel_noise = noise[i * n: (i + 1) * n]
        prop = phi_func(coef, rel_vars)
        t = treatment_assigment(prop)
        potential_outcomes_lst = get_potential_outcomes(rel_vars, y_coef, rel_noise)
        y = np.where(t == 1, potential_outcomes_lst[1], potential_outcomes_lst[0])
        if save and i == 0:
            saved_dict['t'] = t
            saved_dict['prop'] = prop
            saved_dict['models'] = {}
        else:
            save = False
        for expr, prop_func in experiments.items():
            if callable(prop_func):
                prop_hat = prop_func(prop)
            else:
                prop_hat = prop
            ate_hat = calc_ipw(y, t, prop_hat)
            scale = get_sacle_from_name(expr)
            if save:
                saved_dict['models'][scale] = {'deformed': prop_hat}
            err_df_list.append(
                generate_calib_error_df(
                    t,
                    prop_hat,
                    i,
                    type_=expr,
                    scale=scale
                )
                .assign(ATE=ate_hat)
            )
            if callable(post_colab_func):
                new_label = f'{expr}_calibrated'
                prop_hat = post_colab_func(prop_hat, t)
                ate_hat = calc_ipw(y, t, prop_hat)
                err_df_list.append(
                    generate_calib_error_df(
                        t,
                        prop_hat,
                        i,
                        type_=new_label,
                        scale=scale,
                        calibration_type=post_colab_func.__name__
                    )
                    .assign(ATE=ate_hat)
                )
                if save:
                    saved_dict['models'][scale].update({'corrected': prop_hat})
    if orig_save: # because we make the upper flag to be false
        with open('models.pkl', 'wb') as f:
            pickle.dump(saved_dict, f)
    return pd.concat(err_df_list)


def scaled_for_experiments(scales):
    """ generation of experiments of re-scaled model

    Args:
        scales:

    Returns:

    """
    experiments = {
        f'scaled_expit_{i}': lambda x, coef=i: expit(logit(x) * coef)
        for i in scales
    }
    return experiments


def sigmoid_calib(pred, t):
    """ re-calibration of prediction, using Plat 2000

    using implantation by sci-kit learn

    Args:
        pred: the prediction
        t: the target variable

    Returns:

    """
    a, b = _sigmoid_calibration(pred, t)
    return expit(-(a * pred + b))


def plot_calibration(df, calib_metric, calib_metric_label, error='ATE_error', error_label='ATE Error',
                     upper_y_bound=3.5, upper_x_bound=0.3, cm=None):
    """

    Args:
        upper_y_bound:
        upper_x_bound:
        cm:
        df:
        calib_metric:
        calib_metric_label:
        error:
        error_label:

    Returns:

    """
    plt.figure(figsize=(10, 10))
    if cm is not None:
        hue_s = df.sort_values('scale').groupby('type')['scale'].max().sort_values().apply(np.log).map(cm)
        hue_order = hue_s.index.tolist()
        palette = sns.color_palette(hue_s.values.tolist(), hue_s.shape[0])
    else:
        hue_order = None
        palette = None
    sns.scatterplot(x=calib_metric, y=error, data=df.sort_values('scale'), hue='type',
                    hue_order=hue_order, palette=palette)
    plt.ylim(-0.01, upper_y_bound)
    plt.xlim(0, upper_x_bound)
    plt.xlabel(calib_metric_label, fontdict={'weight': 'bold', 'size': 17})
    plt.ylabel(error_label, fontdict={'weight': 'bold', 'size': 17})


def min_max_transform(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def plot_calibration_curve(res_dict, scale):
    t = res_dict['t']
    prop = res_dict['prop']
    models = res_dict['models'][scale]
    models.update({'identity': prop})
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, model in models.items():

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(t, model, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label=f"{name}")

        ax2.hist(model, range=(0, 1), bins=100, label=name,
                 histtype="bar", lw=2, alpha=0.5)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plots  (reliability curve) for scale: {scale}')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper left", ncol=2)
