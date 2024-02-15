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

# plt.show()

synthetic_df = pd.read_csv(os.path.join('models_test', 'nested_sig_with_lr.csv'), index_col=[0])

model_names = ['Logistic Regression', 'Lasso Logistic Regression', 'Ridge Logistic Regression',
               "Gradient Boosting Trees", "Random Forest"]

ax = plot_comp_plot(synthetic_df.query("scale != 'identity_model'"), x_label=r"Calibration error",
                    plot_legend=True, force_names=model_names, color_edges=True, fig=subfigs[1])
# plt.tight_layout()
# plt.show()

save_figure_in_format(figs, run_dir, filename="figure1")


## Make graph 2

scales = [0.125, 0.25, 0.5, 0.75, 1, 1.5, 1.75, 2, 3]
row_limit = 3

res_dict_simulation = get_res_dict(cur_run_dir=os.path.join('outputs', 'sim_nest_norm3_noises_t05_o05'))

make_calibration_graphs_scales(res_dict_simulation, scales, row_limit, run_dir)

res_dict_acic = get_res_dict(cur_run_dir='models_test', file_name='acic42_2_new_sig.pkl')

make_calibration_graph_synthetic(res_dict=res_dict_acic, cur_run_dir=run_dir)