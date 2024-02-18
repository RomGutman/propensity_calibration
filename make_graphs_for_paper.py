import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

# import utils

from utils import (plot_comp_plot, plot_comp_simulation_plot, save_figure_in_format, make_run_dir,
                   make_calibration_graphs_scales, get_res_dict)

from RunSynthetic import make_calibration_graph_synthetic
from RunSimulation import make_calibration_graphs_models
# mpl.use('TkAgg')

run_dir = make_run_dir("figs_for_paper")

# Make graph 1

figs = plt.figure(figsize=(21, 10), constrained_layout=True)
subfigs = figs.subfigures(1, 2, wspace=0.01)

sim_df = pd.read_csv(os.path.join('outputs', 'sim_nest_norm3_noises_t05_o05', 'simulation_df.csv'), index_col=[0])
y_coef = np.array([5, 1.2, 3.6, 1.2, 1.2, 1])

sim_df['ATE_error'] = (sim_df['ATE'] - y_coef[0]).pipe(lambda x: np.sqrt(x ** 2))
sim_df['ATE_error_l1'] = (sim_df['ATE'] - y_coef[0]).pipe(lambda x: np.abs(x))

scales_for_paper = [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2]

cm = sns.diverging_palette(240, 50, s=80, l=70,
                           n=len(scales_for_paper),
                           as_cmap=True,
                           center='light'
                           )

model_rows = pd.to_numeric(sim_df['scale'], errors='coerce').isna()
temp_scale_df = sim_df[~model_rows].copy()
temp_scale_df['scale'] = temp_scale_df['scale'].astype('float64')
# can be changed or generalized with more/fewer scales
ax = plot_comp_simulation_plot(temp_scale_df.query('scale in [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2]'),
                               cm=cm, color_edges=True, fig=subfigs[0])
# plt.tight_layout()

ax.text(0., 1.01, "A)", fontsize=28, fontweight='bold', transform=ax.transAxes)
# plt.show()

synthetic_df = pd.read_csv(os.path.join('models_test', 'nested_sig_with_lr.csv'), index_col=[0])

model_names = ['Logistic Regression', 'Lasso Logistic Regression', 'Ridge Logistic Regression',
               "Gradient Boosting Trees", "Random Forest"]

ax = plot_comp_plot(synthetic_df.query("scale != 'identity_model'"), x_label=r"Calibration error",
                    plot_legend=True, force_names=model_names, color_edges=True, fig=subfigs[1])
# plt.tight_layout()
# plt.show()
ax.text(0., 1.01, "B)", fontsize=28, fontweight='bold', transform=ax.transAxes)
# ax.text(0.014, 0.73, "B)", fontsize=28, fontweight='bold')

save_figure_in_format(figs, run_dir, filename="figure1")


# Separating figure 1

fig = plt.figure(figsize=(10, 10))

ax = plot_comp_simulation_plot(temp_scale_df.query('scale in [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2]'),
                               cm=cm, color_edges=True, fig=fig)
plt.tight_layout()
save_figure_in_format(fig, run_dir, filename="simulation_calibration_ate")

fig = plt.figure(figsize=(10, 10))

ax = plot_comp_plot(synthetic_df.query("scale != 'identity_model'"), x_label=r"Calibration error",
                    plot_legend=True, force_names=model_names, color_edges=True, fig=fig)
plt.tight_layout()
save_figure_in_format(fig, run_dir, filename="acic_models_calibration_ate")


## Make graph 2

scales = [0.125, 0.25, 0.5, 0.75, 1, 1.5, 1.75, 2, 3]
row_limit = 3

res_dict_simulation = get_res_dict(cur_run_dir=os.path.join('outputs', 'sim_nest_norm3_noises_t05_o05'))

make_calibration_graphs_scales(res_dict_simulation, scales, row_limit, run_dir)

res_dict_acic = get_res_dict(cur_run_dir='models_test', file_name='acic42_2_new_sig.pkl')

make_calibration_graph_synthetic(res_dict=res_dict_acic, cur_run_dir=run_dir)


## Make eFigure 4:

# figs = plt.figure(figsize=(21, 10), constrained_layout=True)
# subfigs = figs.subfigures(1, 2, wspace=0.01)

fig = plt.figure(figsize=(10, 10))

rel_models = sim_df[model_rows].copy()

ax = plot_comp_plot(rel_models, x_label=r"Calibration error",
                    plot_legend=True, force_names=model_names, color_edges=True, fig=fig,
                    legend_loc='upper left')

plt.tight_layout()
save_figure_in_format(fig, run_dir, filename="simulation_models_calibration_ate")

fig = plt.figure(figsize=(10, 10))

ax = plot_comp_plot(rel_models, y_metric='Balancing', y_label='Balancing error',
                    x_label=r"Calibration error",
                    plot_legend=True, force_names=model_names, color_edges=True, fig=fig,
                    legend_loc='lower right')

plt.tight_layout()
save_figure_in_format(fig, run_dir, filename="simulation_models_calibration_balancing")



# Make eFigure 5

make_calibration_graphs_models(res_dict=res_dict_simulation, cur_run_dir=run_dir)

# make eFigure 6

outputs = os.path.basename(run_dir)
run_names_dict = {
    'sim_only_n03_t05_t05_matching': 'matching',
    'sim_only_n03_t05_t05_strata_10': 'qstart_10',
    'sim_only_n03_t05_t05_strata_nq_10': 'start_10',
    'sim_only_n03_t05_t05_strata_20': 'qstart_20',
    'sim_only_n03_t05_t05_strata_nq_20': 'strat_20',
    'sim_only_n03_t05_t05_strata_30': 'qstart_30',
    'sim_only_n03_t05_t05_strata_nq_30': 'strat_30'
         }

res_type = ['models', 'defrec']


def plot_sim_graphs(run_names, cm, run_dir, only_models=False, plot_only_balancing=False):
    for run in run_names.keys():
        temp_sim_df = pd.read_csv(os.path.join('outputs', run, 'calib_df.csv'), index_col=[0])
        temp_model_rows = pd.to_numeric(temp_sim_df['scale'], errors='coerce').isna()
        temp_rel_models = temp_sim_df[temp_model_rows].copy()
        if not only_models:
            temp_scale_df = temp_sim_df[~temp_model_rows].copy()
            temp_scale_df['scale'] = temp_scale_df['scale'].astype('float64')

            # for eFigure 7
            fig = plt.figure(figsize=(10, 10))
            ax = plot_comp_simulation_plot(temp_scale_df.query('scale in @scales_for_paper'),
                                           cm=cm, color_edges=True, fig=fig)
            plt.tight_layout()
            save_figure_in_format(fig, run_dir, filename=f"sim_defrec-{run_names[run]}-calibration_ate")

        # for eFigure 6
        if not plot_only_balancing:
            fig = plt.figure(figsize=(10, 10))
            ax = plot_comp_plot(temp_rel_models, x_label=r"Calibration error",
                                plot_legend=True, force_names=model_names, color_edges=True, fig=fig,
                                legend_loc='upper right')
            plt.tight_layout()
            save_figure_in_format(fig, run_dir, filename=f"sim_models-{run_names[run]}-calibration_ate")

        else:
            fig = plt.figure(figsize=(10, 10))
            ax = plot_comp_plot(temp_rel_models, y_metric='Balancing', y_label='Balancing error',
                                x_label=r"Calibration error",
                                plot_legend=True, force_names=model_names, color_edges=True, fig=fig,
                                legend_loc='lower right')

            plt.tight_layout()
            save_figure_in_format(fig, run_dir, filename=f"sim_models-{run_names[run]}-calibration_balancing")


plot_sim_graphs(run_names_dict, cm=cm, run_dir=run_dir)

# for eFigure 8:

confounding_runs = {
    'sim_only_n03_t05_t05_try_misspec_ipw': 'unconf',
    'sim_only_n03_t05_t05_try_additive_2_misspec_ipw': 'proxy-nonzero_mean',
    'sim_only_n03_t05_t05_try_additive_0_misspec_ipw': 'proxy-zero_mean',
}
plot_sim_graphs(run_names=confounding_runs, cm=cm, run_dir=run_dir, only_models=True)


# for eFigure 9:

fig9_names_dict = {
    'sim_only_n03_t05_t05_try_misspec_ipw': 'unconf',
    'sim_only_n03_t05_t05_matching': 'matching',
    'sim_only_n03_t05_t05_strata_30': 'qstart_30',
    'sim_only_n03_t05_t05_strata_nq_30': 'strat_30'
         }


fig = plt.figure(figsize=(10, 10))

plot_comp_plot(synthetic_df.query("scale != 'identity_model'"), y_metric='Balancing',
                     y_label='Balancing', x_label=r"Calibration error",
                     plot_legend=True, force_names=model_names, color_edges=True, fig=fig)
plt.tight_layout()
save_figure_in_format(fig, run_dir, filename=f"acic_models_calibration_balancing")


plot_sim_graphs(run_names=fig9_names_dict, cm=cm, run_dir=run_dir, only_models=True, plot_only_balancing=True)
