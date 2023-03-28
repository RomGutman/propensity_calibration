import os
import warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import utils
warnings.filterwarnings('ignore')


# def make_run_dir(run_name):
#     """
#
#     Args:
#         run_name:
#
#     Returns:
#
#     """
#     cur_dir = os.path.abspath(os.getcwd())
#
#     outputs_dir = os.path.join(cur_dir, "outputs")
#     # run_name = 'sim_nest_new_run_noises_t05'
#
#     cur_run_dir = os.path.join(outputs_dir, run_name)
#     os.makedirs(cur_run_dir)
#     return cur_run_dir


def make_graphs_for_models(rel_models, force_names, cur_run_dir):
    """
    Reproduce scale graphs from paper, on simulation

    Args:
        rel_models: results df only with statistical estimators for propensity
        force_names: the model names
        cur_run_dir: the folder to save the figs in

    Returns:

    """

    utils.plot_comp_plot(rel_models, x_label=r"Calibration error",
                         plot_legend=True, force_names=force_names, color_edges=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'simulation_models_calibration_ate.jpg'), dpi=400)

    utils.plot_comp_plot(rel_models, y_metric='Balancing', y_label='Balancing error',
                         x_label=r"Calibration error",
                         plot_legend=True, force_names=force_names, color_edges=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'simulation_models_calibration_balancing.jpg'), dpi=400)

    utils.plot_comp_plot(rel_models, metric='Balancing', x_label='Balancing error',
                         plot_legend=True, force_names=force_names, color_edges=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'simulation_models_balancing_ate.jpg'), dpi=400)


def make_graphs_for_scales(temp_scale_df, cm, cur_run_dir):
    """
    Reproduce scale graphs from paper, on simulation

    Args:
        temp_scale_df: result df, only with for synthetic scaled propensity
        cm: color palates for graphs
        cur_run_dir: the folder to save the graphs

    Returns:

    """
    temp_scale_df['scale'] = temp_scale_df['scale'].astype('float64')
    # can be changed or generalized with more/less scales
    utils.plot_comp_simulation_plot(temp_scale_df.query('scale in [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2]'),
                                    cm=cm, color_edges=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'simulation_calibration_ate.jpg'), dpi=400)

    utils.plot_comp_simulation_plot(temp_scale_df.query('scale in [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2]'),
                                    cm=cm, metric='Balancing', x_label='Balancing error', color_edges=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'simulation_balancing_ate.jpg'), dpi=400)

    utils.plot_comp_simulation_plot(temp_scale_df.query('scale in [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2]'),
                                    cm=cm, y_metric='Balancing', y_label='Balancing error', color_edges=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'simulation_balancing_calibration.jpg'), dpi=400)


def get_res_dict(cur_run_dir):
    import pickle
    with open(os.path.join(cur_run_dir, "models.pkl"), 'rb') as f:
        res_dict = pickle.load(f)
    return res_dict


def make_calibration_graphs_models(res_dict, cur_run_dir):
    """
    make calibration graphs for statistical models

    Args:
        res_dict: the results dict
        cur_run_dir: the folder to save the figs

    Returns:

    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    utils.plot_calibration_curve(res_dict, 'rf_cv_model', ax1=axes[0][0], model_name="Random Forst")
    utils.plot_calibration_curve(res_dict, 'GBT_cv_model', ax1=axes[0][1], model_name="Gradient Boosting Trees")
    utils.plot_calibration_curve(res_dict, 'lr_model', ax1=axes[0][2], model_name="Logisitic Regression")
    utils.plot_calibration_curve(res_dict, 'lr_l1_model', ax1=axes[1][0], model_name="Lasso Logisitic Regression")
    utils.plot_calibration_curve(res_dict, 'lr_l2_model', ax1=axes[1][1], model_name="Ridge Logisitic Regression")
    fig.supxlabel('Predicted probability', fontweight="bold", fontsize=30)
    fig.supylabel('Actual probability', fontweight="bold", fontsize=30, x=0.01)
    fig.suptitle("Calibration curves of statistical estimators", fontweight="bold", fontsize=25)

    fig.legend(*axes[0][1].get_legend_handles_labels(), loc=(0.73, 0.2), prop={'weight': 'bold', 'size': 18})
    for x in range(2):
        for y in range(3):
            if x == 1 and y == 2:
                continue
            axes[x][y].get_legend().remove()
    axes[1][2].remove()
    plt.tight_layout()

    plt.savefig(os.path.join(cur_run_dir, 'simulation_models_calibration_curves.jpg'), dpi=400)


def make_calibration_graphs_scales(res_dict, scales, row_limit, cur_run_dir):
    fig, axes = plt.subplots(row_limit, row_limit, figsize=(20, 15))
    i, j = 0, 0
    for idx, scale in enumerate(scales):
        utils.plot_calibration_curve(res_dict, scale, ax1=axes[i][j])
        j += 1
        if j >= row_limit:
            i += 1
            j = 0
    fig.supxlabel('Predicted probability', fontweight="bold", fontsize=30)
    fig.supylabel('Actual probability', fontweight="bold", fontsize=30, x=0.01)
    fig.suptitle("Calibration curves of synthetic propensity scores", fontweight="bold", fontsize=25)

    plt.tight_layout()

    plt.savefig(os.path.join(cur_run_dir, 'simulation_deforming_calibration_curves.jpg'), dpi=400)


if __name__ == '__main__':
    cur_run_dir = utils.make_run_dir("sim_only_n03_t05_t05")

    amount_of_vars = 4
    intercept = 0
    mean = 0
    std = 3
    n = 10000
    p_x = 'normal'

    noise_mean = 0
    noise_std = 1

    coef = np.array([-0.1, .05, .2, -.05])
    y_coef = np.array([5, 1.2, 3.6, 1.2, 1.2, 1])
    t_noise_mean = 0
    t_noise_std = .5

    outcome_noise_mean = 0
    outcome_noise_std = .5

    rf_tuned_parameters = [{'max_depth': [1, 2, 3],
                            'n_estimators': [1, 5, 10, 100, 200, 400, 1000]}]

    gb_tuned_parameters = [{'max_depth': [1, 2, 3, 4],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'n_estimators': [1, 3, 5, 10, 15]}]

    scores = 'neg_brier_score'

    cv_inner = KFold(n_splits=10, shuffle=True, random_state=42)

    model_experiments = {
        'lr': LogisticRegression(random_state=42, n_jobs=-1, penalty='none'),
        'lr_l1': LogisticRegressionCV(random_state=42, n_jobs=-1, cv=10, solver='saga', penalty='l1', max_iter=1e4),
        'lr_l2': LogisticRegressionCV(random_state=42, n_jobs=-1, cv=10, solver='saga', penalty='l2', max_iter=1e4),
        'GBT_cv': GridSearchCV(GradientBoostingClassifier(random_state=42), gb_tuned_parameters, scoring=scores,
                               n_jobs=-1, cv=cv_inner),
        'rf_cv': GridSearchCV(RandomForestClassifier(random_state=42), rf_tuned_parameters, scoring=scores, n_jobs=-1,
                              cv=cv_inner),
    }
    scaling_range = [0.125, 0.25, 1 / 3, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3]

    experiments = utils.scaled_for_experiments(scaling_range)

    experiments.update(model_experiments)

    num_of_experiments = 1000

    calib_df = utils.generate_simulation(
        m=amount_of_vars,
        mean=mean, std=std, n=n,
        treatment_noise_mean=t_noise_mean,
        treatment_noise_std=t_noise_std,
        outcome_noise_mean=outcome_noise_mean,
        outcome_noise_std=outcome_noise_std,
        coef=coef,
        y_coef=y_coef,
        num_of_experiments=num_of_experiments,
        experiments=experiments,
        post_colab_func=utils.sigmoid_calib,
        save=True,
        nested_cv=True,
        save_dir=cur_run_dir,
        p_x=p_x
    )

    calib_df['ATE_error'] = (calib_df['ATE'] - y_coef[0]).pipe(lambda x: np.sqrt(x ** 2))
    calib_df['ATE_error_l1'] = (calib_df['ATE'] - y_coef[0]).pipe(lambda x: np.abs(x))

    calib_df.to_csv(os.path.join(cur_run_dir, "calib_df.csv"))
    print(calib_df)

    model_rows = pd.to_numeric(calib_df['scale'], errors='coerce').isna()
    # rel_models = calib_df[model_rows].copy()

    force_names = ['Logistic Regression', 'Lasso Logistic Regression', 'Ridge Logistic Regression',
                   "Gradient Boosting Trees", "Random Forest"]
    make_graphs_for_models(rel_models=calib_df[model_rows].copy(), force_names=force_names, cur_run_dir=cur_run_dir)

    cm = sns.diverging_palette(240, 50, s=80, l=70,
                               n=calib_df['scale'].nunique(),
                               as_cmap=True,
                               center='light'
                               )

    make_graphs_for_scales(temp_scale_df=calib_df[~model_rows].copy(), cm=cm, cur_run_dir=cur_run_dir)

    res_dict = get_res_dict(cur_run_dir)

    make_calibration_graphs_models(res_dict, cur_run_dir)
    scales = [0.125, 0.25, 0.5, 0.75, 1, 1.5, 1.75, 2, 3]
    row_limit = 3
    make_calibration_graphs_scales(res_dict, scales, row_limit, cur_run_dir)
