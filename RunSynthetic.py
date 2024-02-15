from glob import glob
import re
import os
from copy import deepcopy

from tqdm import tqdm
from causallib.datasets import load_acic16
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

import utils


def get_expr_id(s, r_pattern):
    """
    get the id of experiment from the list

    Args:
        s: file name
        r_pattern: regex pattern to find

    Returns:
        the id of the acic experiment
    """
    return re.findall(r_pattern, s)[0]


def get_data_files(reg):
    """
    Retrieves the data files of the acic experiments

    Args:
        reg:

    Returns:

    """
    cur_dir = os.path.abspath(os.getcwd())
    data_dir = os.path.join(cur_dir, "data")
    pattern = 'var*_*.csv'
    target_files_list = glob(os.path.join(data_dir, pattern))
    rel_target_list = sorted([f for f in target_files_list if get_expr_id(f, reg).split('_')[0] == '42'])
    return rel_target_list


def run_multiple_experiments(experiment_files, model_experiments, reg, x_acic):
    """
    Runs the experiments, as detailed in the paper for the synthetic data

    Args:
        experiment_files: the experiments files, with the acic data
        model_experiments: the statistical models to be used
        reg: the regularization pattern to extract the data number
        x_acic: the covarites of the ACIC data

    Returns:
        Dataframe with results, one row per model, per experiment
    """
    df_list = []
    for instance in tqdm(experiment_files):
        models = deepcopy(model_experiments)
        id_ = get_expr_id(instance, reg)
        instance_df = pd.read_csv(instance, index_col=[0]).reset_index(drop=True)
        po = instance_df[['y.0', 'y.1']].copy()
        po.columns = [0, 1]
        e = instance_df['e'].copy()
        c_df_1 = utils.run_synthetic_experiments(var=x_acic,
                                                 t=instance_df['z'],
                                                 e=e,
                                                 potential_outcomes=po,
                                                 experiments=models,
                                                 post_colab_func=utils.sigmoid_calib,
                                                 save=True, save_name=f'models_test_new/acic{id_}_new_sig.pkl', id_=id_,
                                                 nested_cv=False
                                                 )
        ate = (po[1] - po[0]).mean()
        c_df_1['ATE_error'] = (c_df_1['ATE'] - ate).pipe(lambda x: np.sqrt(x ** 2))
        df_list.append(c_df_1)
    df_res = pd.concat(df_list)
    df_res.to_csv('nested_sig_with_lr.csv')
    return df_res


def get_res_dict():
    """
    get result dict of the experiments

    Returns:
        dict of the results
    """
    import pickle
    with open("models_test/acic42_2_new_sig.pkl", 'rb') as f:
        res_dict = pickle.load(f)
    return res_dict


def make_calibration_graph_synthetic(res_dict, cur_run_dir):
    """
    Create the calibration graphs from the paper

    Args:
        res_dict: the results dict
        cur_run_dir: the folder to save the figs

    Returns:

    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    utils.plot_calibration_curve(res_dict, 'rf_cv_model', ax1=axes[0][0], model_name="Random Forest",
                                 key_order=("Ground truth", "uncalibrated", "calibrated"))
    utils.plot_calibration_curve(res_dict, 'GBT_cv_model', ax1=axes[0][1], model_name="Gradient Boosting Trees",
                                 key_order=("Ground truth", "uncalibrated", "calibrated"))
    utils.plot_calibration_curve(res_dict, 'lr_model', ax1=axes[0][2], model_name="Logistic Regression",
                                 key_order=("Ground truth", "uncalibrated", "calibrated"))
    utils.plot_calibration_curve(res_dict, 'lr_l1_model', ax1=axes[1][0], model_name="Lasso Logistic Regression",
                                 key_order=("Ground truth", "uncalibrated", "calibrated"))
    utils.plot_calibration_curve(res_dict, 'lr_l2_model', ax1=axes[1][1], model_name="Ridge Logistic Regression",
                                 key_order=("Ground truth", "uncalibrated", "calibrated"))
    fig.supxlabel('Predicted probability', fontweight="bold", fontsize=30)
    fig.supylabel('Actual probability', fontweight="bold", fontsize=30, x=0.01)
    fig.suptitle("Calibration curves of statistical estimators (ACIC)", fontweight="bold", fontsize=25)

    fig.legend(*axes[0][1].get_legend_handles_labels(), loc=(0.73, 0.2), prop={'weight': 'bold', 'size': 18})
    for x in range(2):
        for y in range(3):
            if x == 1 and y == 2:
                continue
            axes[x][y].get_legend().remove()
    axes[1][2].remove()
    plt.tight_layout()

    utils.save_figure_in_format(figure=fig, save_dir=cur_run_dir, filename='calibration_acic')

def make_graphs_for_synthetic(df, force_names, cur_run_dir):
    """
    make graphs for synthetic data, as shown in paper

    Args:
        df: dataframe of models per experiment
        force_names: names of the model to be shown
        cur_run_dir: the dir to save graphs

    Returns:

    """

    utils.plot_comp_plot(df.query("scale != 'identity_model'"), x_label=r"Calibration error",
                         plot_legend=True, force_names=force_names, color_edges=True)
    # plt.

    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'acic_models_calibration_ate.jpg'), dpi=400)

    utils.plot_comp_plot(df.query("scale != 'identity_model'"), y_metric='Balancing',
                         y_label='Balancing', x_label=r"Calibration error",
                         plot_legend=True, force_names=force_names, color_edges=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'acic_models_calibration_balancing.jpg'), dpi=400)

    utils.plot_comp_plot(df.query("scale != 'identity_model'"), metric='Balancing', x_label='Balancing error',
                         plot_legend=True, force_names=force_names, color_edges=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cur_run_dir, 'acis_models_balancing_ate.jpg'), dpi=400)


if __name__ == '__main__':
    regex_term = r'(?<=var)\d{1,2}_\d{1,2}'
    rel_target_list = get_data_files(reg=regex_term)

    X = load_acic16(instance=1, raw=False)['X']

    rf_tuned_parameters = [{'max_depth': [5, 10, 20, 30],
                            'n_estimators': [50, 100, 200, 400, 1000]}]

    gb_tuned_parameters = [{'max_depth': [1, 2, 3, 6],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'n_estimators': [30, 50, 100, 250, 500, 1000]}]

    scores = 'neg_brier_score'

    cv_inner = KFold(n_splits=10, shuffle=True, random_state=42)

    model_experiments = {
        'lr': LogisticRegression(random_state=42, n_jobs=-1, penalty='none'),
        'lr_l1': LogisticRegressionCV(random_state=42, n_jobs=-1, cv=10, solver='saga', penalty='l1', max_iter=1e4),
        'lr_l2': LogisticRegressionCV(random_state=42, n_jobs=-1, cv=10, solver='saga', penalty='l2', max_iter=1e4),
        'GBT_cv': GridSearchCV(GradientBoostingClassifier(random_state=42), gb_tuned_parameters, scoring=scores,
                               n_jobs=-1, cv=cv_inner),
        'rf_cv': GridSearchCV(RandomForestClassifier(random_state=42, oob_score=True), rf_tuned_parameters,
                              scoring=scores, n_jobs=-1, cv=cv_inner),
        'identity': None
    }

    df_42_nested_sig_new = run_multiple_experiments(rel_target_list, model_experiments, reg=regex_term, x_acic=X)

    print(utils.get_slopes(df_42_nested_sig_new.query("scale != 'identity_model'")))
    print(utils.get_slopes(df_42_nested_sig_new.query("scale != 'identity_model'"),
                           x_metric='mean', y_metric='Balancing'))
    print(utils.get_slopes(df_42_nested_sig_new.query("scale != 'identity_model'"), x_metric='Balancing'))

    run_dir = utils.make_run_dir("sig_nested_42_lr")

    res_dict = get_res_dict()
    make_calibration_graph_synthetic(res_dict, cur_run_dir=run_dir)

    model_names = ['Logistic Regression', 'Lasso Logistic Regression', 'Ridge Logistic Regression',
                   "Gradient Boosting Trees", "Random Forest"]

    make_graphs_for_synthetic(df=df_42_nested_sig_new, force_names=model_names, cur_run_dir=run_dir)