import os.path
import warnings
import pickle
from copy import copy


import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from sklearn.calibration import _sigmoid_calibration, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from sklearn.isotonic import IsotonicRegression
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression

from causallib.datasets import load_nhefs
from causallib.evaluation.weight_evaluator import calculate_covariate_balance
from causallib.estimation.matching import  Matching

import utils


def get_variables(mean_, std_, n_, m_=1, treat_noise_mean_=0, treat_noise_std_=1,
                  outcome_noise_mean_=0, outcome_noise_std_=1, p_x='normal'):
    """ Generates normal distributed random variables for simulations.

    Draws M random variables from normal distribution, for N instances with given mean and std.
    Also draws N noise variables with given mean and std (possibly different from the variables).

    Args:
        mean_:
        std_:
        n_:
        m_:
        treat_noise_mean_:
        treat_noise_std_:
        outcome_noise_mean_:
        outcome_noise_std_:

    Returns:

    """
    if p_x == "normal":
        x_ = np.random.normal(loc=mean_, scale=std_, size=[n_, m_])
    elif p_x == "uniform":
        x_ = np.random.uniform(low=-std_, high=std_, size=[n_, m_])
    else:
        raise ValueError("Distribution of X must be either uniform or normal")
    treatment_noise_ = np.random.normal(loc=treat_noise_mean_, scale=treat_noise_std_, size=n_)
    outcome_noise_ = np.random.normal(loc=outcome_noise_mean_, scale=outcome_noise_std_, size=n_)
    return x_, treatment_noise_, outcome_noise_


def expit_transform(coef_, variables_, exp_scale=1, noise=0):
    """

    Args:
        coef_:
        variables_:

    Returns:

    """
    exp = np.dot(coef_, np.transpose(variables_)) + noise
    return expit(exp * exp_scale)


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


def calc_sipw(y, t, prop, eps=1E-7):
    prop = np.clip(prop, eps, 1 - eps)
    y1 = np.sum(y * t / prop)
    y1_weight = np.sum(t/prop)
    y1 = y1 * y1_weight
    y0 = np.sum(y * (1 - t) / (1 - prop))
    y0_weight = np.sum((1-t)/(1-prop))
    y0 = y0 * y0_weight
    return y1 - y0


def calc_matching(y, t, prop, eps=1E-7):
    """
    calculate propensity matching

    Args:
        y: outcome
        t: treatment
        prop: propensity
        eps: epsilon value for cliping the values

    Returns: ATE res

    """
    prop = np.clip(prop, eps, 1 - eps)
    matching = Matching(with_replacement=False, n_neighbors=1, matching_mode='control_to_treatment', metric='euclidean')
    X = pd.DataFrame(prop)
    a = pd.Series(t, index=X.index)
    y_s = pd.Series(y, index=X.index)
    matching.fit(X=X, a=a, y=y_s)
    match_df = matching.estimate_population_outcome(X=X, a=a, y=y_s)
    return match_df[1] - match_df[0]


def calc_stratification(y, t, prop, eps=1E-7, n_strata=20):
    prop = np.clip(prop, eps, 1 - eps)
    ps_qcut = pd.qcut(prop, q=n_strata)  # can use pd.cut instead
    outcome_model = LinearRegression()
    psa = pd.get_dummies(ps_qcut, drop_first=True).join(pd.Series(t, name='t'))
    outcome_model.fit(psa, y)
    return outcome_model.coef_[-1]  # The ATE is `a`’s coefficient


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


def is_scaled(label, label_prefix='scaled'):
    return label_prefix == label.split('_')[0]


def generate_simulation(n, variables, treatment_noise, outcome_noise, coef, y_coef, num_of_experiments=1,
                        phi_func=expit_transform, experiments=None, post_colab_func=None, calc_effect=utils.calc_ipw,
                        save=False,
                        nested_cv=False, save_dir='.'):
    """

    Args:
        calc_effect:
        outcome_noise:
        treatment_noise:
        variables:
        nested_cv:
        save_dir:
        save: Whether to save the simulation run to present later
        phi_func:
        post_colab_func:
        n:
        coef:
        y_coef:
        num_of_experiments:
        experiments:

    Returns:

    """
    if save:
        assert os.path.exists(save_dir), f"The directory {save_dir} doesn't exists."
        print(f"Saving to {save_dir}")
    if experiments is None:
        warnings.warn("No experiment were given, performing for identity only")
        experiments = {'Identity': None}
    err_df_list = []
    saved_dict = {}
    orig_save = save
    for i in tqdm(range(num_of_experiments), desc="Experiment", position=0):
        rel_vars = variables[i * n: (i + 1) * n]
        t_rel_noise = treatment_noise[i * n: (i + 1) * n]
        y_rel_noise = outcome_noise[i * n: (i + 1) * n]
        prop = phi_func(coef, rel_vars, noise=t_rel_noise)
        t = treatment_assigment(prop)
        potential_outcomes_lst = get_potential_outcomes(rel_vars, y_coef, y_rel_noise)
        y = np.where(t == 1, potential_outcomes_lst[1], potential_outcomes_lst[0])
        if save and i == 0:
            saved_dict['t'] = t
            saved_dict['prop'] = prop
            saved_dict['models'] = {}
        else:
            save = False
        for expr, prop_func in (pbar := tqdm(experiments.items(), desc="Func", position=1, leave=False)):
            pbar.set_description(f"Processing {expr}", refresh=True)
            flag = is_scaled(expr)
            if callable(prop_func):
                prop_hat = prop_func(prop)
            elif nested_cv and prop_func is not None:
                prop_func_temp = copy(prop_func)
                prop_hat = nested_cv_predict(prop_func_temp, rel_vars, t)
            elif isinstance(prop_func, ClassifierMixin) or isinstance(prop_func, BaseEstimator):
                prop_func_temp = copy(prop_func)
                prop_hat = fit_classifier(rel_vars, t, prop_func_temp)
            else:
                warnings.warn(f'for experiment {expr}, predicting identity')
                prop_hat = prop
            ate_hat = calc_effect(y, t, prop_hat)
            smd = calc_balancing(pd.DataFrame(rel_vars), pd.Series(t), prop_hat)
            scale = get_sacle_from_name(expr) if flag else f'{expr}_model'
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
                .assign(Balancing=smd)
            )
            if callable(post_colab_func):
                new_label = f'{expr}_calibrated'
                prop_hat = post_colab_func(prop_hat, t)
                ate_hat = calc_effect(y, t, prop_hat)
                smd = calc_balancing(pd.DataFrame(rel_vars), pd.Series(t), prop_hat)
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
                    .assign(Balancing=smd)
                )
                if save:
                    saved_dict['models'][scale].update({'corrected': prop_hat})
    scaled_df = pd.concat(err_df_list)
    if orig_save:   # because we make the upper flag to be false
        with open(os.path.join(save_dir, 'models.pkl'), 'wb') as f:
            pickle.dump(saved_dict, f)
        scaled_df.to_csv(os.path.join(save_dir, "simulation_df.csv"))
    return scaled_df


def calc_balancing(x, t, e):
    smd = calculate_covariate_balance(x, t, pd.Series(e), metric="abs_smd")
    return smd['weighted'].max()


def run_synthetic_experiments(var, t, e, potential_outcomes, experiments=None, post_colab_func=None,
                              save=False, id_=None, save_name=None, nested_cv=False):
    if experiments is None:
        warnings.warn("No experiment were given, performing for identity only")
        experiments = {'Identity': None}
    x = copy(var)
    y = np.where(t == 1, potential_outcomes[1], potential_outcomes[0])
    err_df_list = []
    saved_dict = {}
    if id_ is None:
        warnings.warn("No id was given to this run, defaulting to 0 instead")
        id_ = 0
    if save:
        saved_dict['t'] = t
        saved_dict['prop'] = e
        saved_dict['models'] = {}
    for expr, prop_func in experiments.items():
        if nested_cv and prop_func is not None:
            prop_hat = nested_cv_predict(prop_func, x, t)
        elif isinstance(prop_func, ClassifierMixin) or isinstance(prop_func, BaseEstimator):
            prop_hat = fit_classifier(x, t, prop_func)
        else:
            warnings.warn(f'for experiment {expr}, predicting identity')
            prop_hat = e
        smd = calc_balancing(x, t, prop_hat)
        ate_hat = calc_ipw(y, t, prop_hat)
        scale = f'{expr}_model'
        with open(f'{scale}_{id_}.pkl', 'wb') as f:
            pickle.dump(prop_func, f)
        if save:
            saved_dict['models'][scale] = {'uncalibrated': prop_hat}
        err_df_list.append(
            generate_calib_error_df(
                t,
                prop_hat,
                id_,
                type_=expr,
                scale=scale
            )
                .assign(ATE=ate_hat)
                .assign(Balancing=smd)
        )
        if callable(post_colab_func):
            new_label = f'{expr}_calibrated'
            prop_hat = post_colab_func(prop_hat, t)
            smd = calc_balancing(x, t, prop_hat)
            ate_hat = calc_ipw(y, t, prop_hat)
            err_df_list.append(
                generate_calib_error_df(
                    t,
                    prop_hat,
                    id_,
                    type_=new_label,
                    scale=scale,
                    calibration_type=post_colab_func.__name__
                )
                    .assign(ATE=ate_hat)
                    .assign(Balancing=smd)
            )
            if save:
                saved_dict['models'][scale].update({'calibrated': prop_hat})
    if save:   # because we make the upper flag to be false
        if save_name is None:
            save_name = f'model{id_}.pkl'
        elif os.path.splitext(save_name)[1] != '.pkl':
            warnings.warn("pickle file should be save with .pkl extension")
            save_name = f'{save_name}.pkl'
        with open(save_name, 'wb') as f:
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


def isotonic_reg(pred, t):
    """ re-calibration of prediction using isotonic regression

    Args:
        pred:
        t:

    Returns:

    """
    iso_reg = IsotonicRegression(increasing='auto')
    pred_hat = iso_reg.fit_transform(pred,t)
    return pred_hat


def plot_calibration(df, calib_metric, calib_metric_label, error='ATE_error', error_label='ATE Error',
                     upper_y_bound=3.5, upper_x_bound=0.3, lower_x_bound=0., cm=None, ax=None, return_ax=False):
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
    if ax is None:
        plt.figure(figsize=(10, 10))
    if cm is not None:
        hue_norm, hue_order, palette = get_palette_for_values(cm, df)
    else:
        hue_order = None
        palette = None
    ax = sns.scatterplot(x=calib_metric, y=error, data=df.sort_values('scale'), hue='type',
                         hue_order=hue_order, palette=palette, legend=None, edgecolor='black', ax=ax)
    ax.set_ylim(-0.01, upper_y_bound)
    ax.set_xlim(lower_x_bound, upper_x_bound)
    ax.set_xlabel(calib_metric_label, fontdict={'weight': 'bold', 'size': 17})
    ax.set_ylabel(error_label, fontdict={'weight': 'bold', 'size': 17})
    if cm is not None and not return_ax:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=hue_norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        # ax.get_legend().remove()
        ax.figure.colorbar(sm)
    if return_ax:
        return ax


def get_palette_for_values(cm, df):
    hue_s = df.sort_values('scale').groupby('type')['scale'].max().sort_values().apply(np.log).map(cm)
    hue_order = hue_s.index.tolist()
    hue_norm = TwoSlopeNorm(1, df['scale'].min(), df['scale'].max())
    palette = cm(hue_norm(df.sort_values('scale')['scale'].unique()))
    return hue_norm, hue_order, palette


def min_max_transform(arr):
    """

    Args:
        arr:

    Returns:

    """
    return (arr - arr.min()) / (arr.max() - arr.min())


def fix_legend_names(key):
    return (key
            .replace("corrected", "Calibrated")
            .replace("deformed", "Uncalibrated")
            .capitalize()
            )


def plot_calibration_curve(res_dict, scale, hist_ratio=3, ax1=None, model_name=None,
                           key_order=("Ground truth", "deformed", "corrected")):
    t = res_dict['t']
    prop = res_dict['prop']
    models = copy(res_dict['models'][scale])
    models.update({'Ground truth': prop})
    rel_model_ordered = {fix_legend_names(key): models[key] for key in key_order}
    if ax1 is None:
        ax1 = plt.figure(figsize=(3, 1)).add_subplot(111)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    ax2 = ax1.twinx()
    for name, model in rel_model_ordered.items():

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(t, model, n_bins=10)
        alpha = 0.4 if name == 'Ground truth' else 0.5
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label=f"{name}", alpha=alpha)
        ax2.hist(model, range=(0, 1), bins=100, label=name,
                 histtype="bar", lw=2, alpha=alpha)

    # ax1.set_ylabel("Fraction of positives", fontdict={'weight': 'bold', 'size': 16})
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left", prop={'weight': 'bold', 'size': 14})
    # label = model_name if model_name is not None else scale
    text = f"{model_name}" if model_name is not None else rf"Calibration deformation scale $(\gamma)$: {scale}"
    ax1.set_title(text,
                  fontdict={'weight': 'bold', 'size': 16})

    top_val = ax2.get_ybound()[1]
    ax2.set_ylim([0, top_val * hist_ratio])
    # print(top_val)
    ax2.set(yticklabels=[])
    ax2.tick_params(right=False)
    # ax2.set_ylabel("Count")
    # ax2.legend(loc="upper left", ncol=2)
    return ax2


def fit_classifier(variables, target, clf):
    """

    Args:
        clf:
        variables:
        target:

    Returns:

    """
    clf.fit(variables, target)
    return clf.predict_proba(variables)[:, 1]


def get_points(df, calib_metric, y_metric='ATE_error'):
    if df.shape[0] != 2:
        print(df)
        raise ValueError('The df should include two instances - before and after calibration')
    x_y_orig = df.loc[df['calibration_type'].isna(), [calib_metric, y_metric]]
    x_y_calib = df.loc[~df['calibration_type'].isna(), [calib_metric, y_metric]]
    return (x_y_orig[calib_metric].iloc[0],
            x_y_orig[y_metric].iloc[0],
            (x_y_calib[calib_metric].iloc[0] - x_y_orig[calib_metric].iloc[0]),
            (x_y_calib[y_metric].iloc[0] - x_y_orig[y_metric].iloc[0]))


def get_arrows(df, calib_metric, y_metric='ATE_error'):
    """

    Args:
        df:
        calib_metric:
        y_metric:

    Returns:

    """

    gb = df.reset_index().groupby('index')

    arrows = []
    for g in gb.groups:
        arrows.append(gb.get_group(g).groupby('scale').apply(get_points, calib_metric, y_metric))

    arrows = pd.concat(arrows)
    return arrows


# todo: fix arrow work
def plot_arrow(values, ax, color=None, alpha=0.4):
    # print(values[:2], tuple(np.array(values[2:]) + np.array(values[:2])))
    # print(values)
    if isinstance(color, dict):
        rel_color = color[values.name]
    else:
        rel_color = color
    if isinstance(values, pd.Series):
        values = values.item()
    tail = values[:2]
    head = tuple(np.array(values[2:]) + np.array(values[:2]))
    # print(alpha)
    arrow = FancyArrowPatch(tail, head, lw=2, alpha=alpha, linestyle='solid', edgecolor=rel_color,
                            arrowstyle='simple,tail_width=0.01', shrinkB=3, mutation_scale=15,
                            facecolor=(0, 0, 0, 0.05))
    ax.add_patch(arrow)
    return 1


def plot_overlap(t, prop):
    fig = plt.figure(figsize=(10, 10))
    plt.hist(prop[t == 1], bins=20, color='cornflowerblue', alpha=0.7, label='Treated');
    plt.hist(prop[t == 0], bins=20, color='red', alpha=0.7, label='Control');

    plt.legend();


def nested_cv_predict(model, variables, target, n_splits=10):

    cv_outer = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    final_score = cross_val_predict(model, variables, target, cv=cv_outer, n_jobs=-1, method='predict_proba')

    return final_score[:, 1]


def plot_comp_plot(big_df, metric='mean', y_metric='ATE_error',
                   x_label='Calibration error', y_label='Effect estimation error',
                   plot_legend=False, force_names=None, color_edges=False):
    fig = plt.figure(figsize=(10, 10))
    for type_, marker in zip(['model', 'calibrated'], ['s', 'o']):
        mask = big_df['calibration_type'].isna() if type_ == 'model' else ~big_df['calibration_type'].isna()
        temp_df = big_df.loc[mask]
        ax = sns.scatterplot(x=metric, y=y_metric, data=temp_df, hue='scale', marker=marker, legend=True, s=100)
    temp_df = big_df

    plt.xlabel(x_label, fontdict={'weight': 'bold', 'size': 17})
    plt.ylabel(y_label, fontdict={'weight': 'bold', 'size': 17})

    arrows = get_arrows(temp_df, calib_metric=metric, y_metric=y_metric)
    names = big_df['scale'].unique() if force_names is None else force_names
    colors = ax.get_children()[0]._facecolors[:len(names)]
    if color_edges:
        orig_names = [obj._label for obj in ax.get_children()[1:len(names) + 1]]
        colors_mapping = {name: color for name, color in zip(orig_names, colors)}
        arrows.to_frame("arrows").apply(plot_arrow, ax=ax, color=colors_mapping, alpha=0.3, axis=1);
    else:
        arrows.apply(plot_arrow, ax=ax);

    if plot_legend:
        # names = big_df['scale'].unique() if force_names is None else force_names
        # colors = ax.get_children()[0]._facecolors[:len(names)]
        legend_elements = [Patch(facecolor=fc, edgecolor='w',
                                 label=name.replace("_model", "").replace("_cv", "").capitalize())
                           for fc, name in zip(colors, names)]

        legend_elements.extend([
            Line2D([0], [0], marker='s', color='w', markeredgecolor='black', lw=4, alpha=1, label='Uncalibrated',
                   markerfacecolor='w', markersize=9),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='black', lw=4, alpha=1, label='Calibrated',
                   markerfacecolor='w', markersize=9),
        ]
        )
        plt.legend(handles=legend_elements, prop={'weight': 'bold', 'size': 13}, framealpha=0.2)
        plt.tight_layout()
    return ax


def plot_comp_simulation_plot(big_df, metric='mean', y_metric='ATE_error',
                              x_label='Calibration error', y_label='Effect estimation error', cm=None,
                              color_edges=False):
    fig = plt.figure(figsize=(10, 10))
    if cm is not None:
        hue_norm, _, palette = get_palette_for_values(cm, big_df)
        hue_order = big_df['scale'].sort_values().unique()
        palette = ListedColormap(palette, name='temp')
    else:
        hue_order = None
        palette = None
    for type_, marker in zip(['model', 'calibrated'], ['s', 'o']):
        mask = big_df['calibration_type'].isna() if type_ == 'model' else ~big_df['calibration_type'].isna()
        temp_df = big_df.loc[mask]
        legend = 'full' if type_ == 'model' else False
        ax = sns.scatterplot(x=metric, y=y_metric, data=temp_df.sort_values('scale'),
                             hue='scale', marker=marker, legend=legend, s=40,
                             hue_order=hue_order, palette=palette, edgecolor='black')
    temp_df = big_df
    plt.xlabel(x_label, fontdict={'weight': 'bold', 'size': 17})
    plt.ylabel(y_label, fontdict={'weight': 'bold', 'size': 17})

    arrows = get_arrows(temp_df, calib_metric=metric, y_metric=y_metric)

    if color_edges:
        colors_mapping = {float(obj._label): obj._facecolors.flatten()
                          for obj in ax.get_children()[1:big_df['scale'].nunique() + 1]}
        arrows.to_frame("arrows").apply(plot_arrow, ax=ax, color=colors_mapping, alpha=0.3, axis=1);
    else:
        arrows.apply(plot_arrow, ax=ax);
    # arrows.apply(plot_arrow, ax=ax);
    legend_elements = [
                   Line2D([0], [0], marker='s', color='w', markeredgecolor='black',
                          lw=4, alpha=1, label='Uncalibrated',
                          markerfacecolor='w', markersize=14),
                   Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
                          lw=4, alpha=1, label='Calibrated',
                          markerfacecolor='w', markersize=14),
                  ]
    plt.legend(handles=legend_elements, prop={'weight': 'bold', 'size': 17})#, title='Deformation', )
    if cm is not None:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=hue_norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        # ax.get_legend().remove()
        clb = ax.figure.colorbar(sm)
        clb.ax.set_ylabel("Calibration Deformation Scale", fontdict={'weight': 'bold',
                                                                    'size': 17}, labelpad=14)
    return ax


def get_row_slope(row):
    """
    get slope of each experiment

    Args:
        row: series of experiment

    Returns:

    """
    _, _, mx, my = row
    return -my / mx


def get_slopes(df, x_metric='mean', y_metric='ATE_error', describe=True):
    arrows = get_arrows(df, calib_metric=x_metric, y_metric=y_metric)
    slopes = arrows.apply(get_row_slope)
    temp_df = slopes.to_frame().reset_index().groupby('scale')
    if describe:
        return temp_df.describe()
    else:
        return temp_df


def make_run_dir(run_name):
    """
    make a dir for outputs experiments
    Args:
        run_name:

    Returns:

    """
    cur_dir = os.path.abspath(os.getcwd())

    outputs_dir = os.path.join(cur_dir, "outputs")
    # run_name = 'sim_nest_new_run_noises_t05'

    cur_run_dir = os.path.join(outputs_dir, run_name)
    if not os.path.exists(cur_run_dir):
        os.makedirs(cur_run_dir)
    return cur_run_dir

