# %%
import os 
import numpy as np 
import pandas as pd 
from absl import app 
from absl import flags 
import compress_pickle
from typing import Sequence 
import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt

# %%
params_list = [] 
all_params_list = [] 
ks = [x for x in range(1,11)] 
# ks = [x for x in range(120,501, 20)]
# ks.extend([x for x in range(20,1001, 20)])
# ks.extend([x for x in range(20,501, 20)])
ks.extend([x for x in range(20,101, 20)])

def get_n_k_for_num_ratings(all_values, num_ratings=5000): 
    values=[] 
    for x in all_values:
        if x==0:
            x=1 
        n = int(np.floor(num_ratings/x)) 
        if n > 0:
            values.append((n, int(x))) 
    return values


# vals = get_n_k_for_num_ratings(ks) 
# [x[1] for x in vals], [x[0] for x in vals]
ratings_list = [] 
# nk_list = [2500]
# nk_list = [2500, 5000, 10000, 25000, 50000]
# nk_list = [1000, 2500, 5000, 10000, 25000, 50000]
# nk_list = [100, 250, 500, 1000, 2500]
nk_list = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
# for r in range(1000, 5001, 1000):
# for r in range(5000, 5001, 500): 
for r in nk_list:
    params = get_n_k_for_num_ratings(ks, r)
    params_list.append(params)
    all_params_list.extend(params)
    ratings_list.extend([(r-x[0]*x[1]) for x in params])
    print(params)

# %%
len(params_list), len(ratings_list), len(all_params_list)

# %%
def plot_ci(alt_ci_nk_list, nk_list, x_list, metric, distortion, dataset, col, base_path):

    plt.figure(figsize=(12, 9))

    title_fontsize = 30
    label_fontsize = 30
    tick_fontsize = 30
    legend_fontsize = 30


    for idx, nk in enumerate(nk_list):
        if len(alt_ci_nk_list) > idx:
            cis = alt_ci_nk_list[idx][:len(x_list)]
            ci_lower = [x[0][0] for x in cis]
            ci_upper = [x[0][1] for x in cis]
            stat_means = [x[1] for x in cis]
            plt.plot(x_list, stat_means, marker='o', label=f'NxK={nk}')
            plt.fill_between(x_list, ci_lower, ci_upper, alpha=0.2)

            # plt.plot(x_list[idx], stat_means, marker='o', label=f'NxK={nk}')
            # # plt.fill_between([x[1] for x in params_list[0][:35]], ci_lower, ci_upper, color='blue', alpha=0.2, label=f'{int(confidence_level*100)}% Confidence Interval')
            # plt.fill_between(x_list[idx], ci_lower, ci_upper, alpha=0.2)
            # break

    plt.xlabel(col, fontsize=label_fontsize)
    plt.ylabel('CI', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(f"CI - {dataset} ({metric}, $\\epsilon$={distortion})", fontsize=title_fontsize)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncols=3, fontsize=legend_fontsize)
    plt.savefig(f"{base_path}/{dataset}_CI_{metric}_{col}_100_e_{distortion}.pdf", bbox_inches='tight')
    plt.close()

# %%
def plot_ci_width(alt_ci_nk_list, nk_list, x_list, metric, distortion, dataset, col, base_path):

    plt.figure(figsize=(12, 9))

    title_fontsize = 30
    label_fontsize = 30
    tick_fontsize = 30
    legend_fontsize = 30

    for idx, nk in enumerate(nk_list):
        if len(alt_ci_nk_list) > idx:
            cis = alt_ci_nk_list[idx][:len(x_list)]
            ci_lower = [x[0][0] for x in cis]
            ci_upper = [x[0][1] for x in cis]
            if metric=='Wins':
                nks = params_list[idx][:len(x_list)]
                ns = [x[0] for x in nks]
                ci_width = (np.array(ci_upper) - np.array(ci_lower))/np.array(ns)
            else:
                ci_width = np.array(ci_upper) - np.array(ci_lower)
            plt.plot(x_list, ci_width, marker='o', label=f'NxK={nk}')
            # plt.plot(x_list[idx], ci_width, marker='o', label=f'NxK={nk}')
            # break

    plt.xlabel(col, fontsize=label_fontsize)
    plt.ylabel('CI-width', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    if metric=='MAE':
        plt.title(f"CI-width - {dataset} (TV, $\\epsilon$={distortion})", fontsize=title_fontsize)
    else:
        plt.title(f"CI-width - {dataset} ({metric}, $\\epsilon$={distortion})", fontsize=title_fontsize)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncols=3, fontsize=legend_fontsize)
    plt.savefig(f"{base_path}/{dataset}_CI_width_{metric}_{col}_100_e_{distortion}.pdf", bbox_inches='tight')
    plt.close()

# %%
datasets = [{"exp_dir": "../../../../data/ptest_arr_toxicity/", "dataset": "Toxicity", "num_categories": "2",},
            {"exp_dir": "../../../../data/ptest_arr_dices/", "dataset": "DICES", "num_categories": "3",},
            {"exp_dir": "../../../../data/ptest_arr_d3code/", "dataset": "D3code", "num_categories": "2",},
            {"exp_dir": "../../../../data/ptest_arr_jobsQ1/", "dataset": "JobsQ1", "num_categories": "5",},
            {"exp_dir": "../../../../data/ptest_arr_jobsQ3/", "dataset": "JobsQ3", "num_categories": "12",},]
# datasets = [
#             {"exp_dir": "../../../../data/ptest_arr_d3code/", "dataset": "D3code", "num_categories": "2",},]

# %%
col = "K"
# distortion = 0.2
distortion_values = [0.1, 0.2, 0.3, 0.4]
# distortion_values = [0.3]
metrics_list = ['Accuracy', 'MAE', 'Wins', 'KL-Div']

# %%
confidence_level = 0.95

for dataset_info in datasets:
    _M_CATEGORIES = dataset_info['num_categories']
    exp_dir = dataset_info['exp_dir']
    dataset = dataset_info['dataset']

    # base_path = f"output/K100/ci_plots/{dataset}"
    base_path = f"output/K100_new/ci_plots/{dataset}"
    # base_path = f"output"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    ci_path_1 = f"{exp_dir}ci_1000"
    ci_path_2 = f"{exp_dir}ci"

    for distortion in distortion_values:
        alt_ci_nk_dict_1 = compress_pickle.load(f"{ci_path_1}/ci_level={confidence_level}_col={col}_500_dist={distortion}.pkl.lz4")
        # print(len(alt_ci_nk_dict_1), len(alt_ci_nk_dict_1['Accuracy']), len(alt_ci_nk_dict_1['Accuracy'][0]))

        alt_ci_nk_dict_2 = compress_pickle.load(f"{ci_path_2}/ci_level={confidence_level}_col={col}_500_dist={distortion}.pkl.lz4")
        # print(len(alt_ci_nk_dict_2), len(alt_ci_nk_dict_2['Accuracy']), len(alt_ci_nk_dict_2['Accuracy'][0]))

        for metric in metrics_list:
            if metric in alt_ci_nk_dict_1 and metric in alt_ci_nk_dict_2:
                alt_ci_nk_list = alt_ci_nk_dict_1[metric] + alt_ci_nk_dict_2[metric]
                # print(len(alt_ci_nk_list))

                # plot_ci(alt_ci_nk_list, nk_list, ks, metric, distortion, dataset, col, base_path)
                plot_ci_width(alt_ci_nk_list, nk_list, ks, metric, distortion, dataset, col, base_path)
            
    #         break
    #     break
    # break

# %%



