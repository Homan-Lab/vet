# %%
import os
import datetime
import numpy as np
import pandas as pd
from absl import app
from tqdm import tqdm
import compress_pickle
from absl import flags
from absl import logging
from typing import Sequence
from scipy.stats import bootstrap
import pathos.multiprocessing as mp
import parameterized_sample_lib as psample
import cat_machine_contest_metrics as cmcm

import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt

# %%
params_list = [] 
all_params_list = [] 
ks = [x for x in range(1,11)] 
ks.extend([x for x in range(20,1001, 20)])

def get_n_k_for_num_ratings(all_values, num_ratings=5000): 
    values=[] 
    for x in all_values:
        if x==0:
            x=1 
        n = int(np.floor(num_ratings/x)) 
        # if n > 30:
        values.append((n, int(x))) 
    return values
# vals = get_n_k_for_num_ratings(ks) 
# [x[1] for x in vals], [x[0] for x in vals]
ratings_list = [] 
# nk_list = [5000]
nk_list = [2500, 5000, 10000, 25000, 50000]
# nk_list = [1000, 2500, 5000, 10000, 25000, 50000]
# for r in range(1000, 5001, 1000):
# for r in range(5000, 5001, 500): 
for r in nk_list:
    params = get_n_k_for_num_ratings(ks, r) 
    params_list.append(params)
    all_params_list.extend(params) 
    ratings_list.extend([(r-x[0]*x[1]) for x in params])
    print(params)

# %%

def get_metric_scores(response_sets, metric):

    null_scores = []
    alt_scores = []
    for i in range(1000):
        null_score=(0.0,0.0)
        alt_score=(0.0,0.0)

        null_simulated_data_gold = response_sets.null_data_list[i].gold
        null_simulated_data_preds1 = response_sets.null_data_list[i].preds1
        null_simulated_data_preds2 = response_sets.null_data_list[i].preds2

        alt_simulated_data_gold = response_sets.alt_data_list[i].gold
        alt_simulated_data_preds1 = response_sets.alt_data_list[i].preds1
        alt_simulated_data_preds2 = response_sets.alt_data_list[i].preds2

        if metric=="Accuracy":
            null_score = cmcm.cat_accuracy(null_simulated_data_gold, null_simulated_data_preds1, null_simulated_data_preds2)
            alt_score = cmcm.cat_accuracy(alt_simulated_data_gold, alt_simulated_data_preds1, alt_simulated_data_preds2)
        if metric=="F1-score":
            null_score = cmcm.cat_f1_score(null_simulated_data_gold, null_simulated_data_preds1, null_simulated_data_preds2)
            alt_score = cmcm.cat_f1_score(alt_simulated_data_gold, alt_simulated_data_preds1, alt_simulated_data_preds2)
        if metric=="MAE":
            null_score = cmcm.cat_mean_absolute_error(null_simulated_data_gold, null_simulated_data_preds1, null_simulated_data_preds2)
            alt_score = cmcm.cat_mean_absolute_error(alt_simulated_data_gold, alt_simulated_data_preds1, alt_simulated_data_preds2)
        if metric=="Wins":
            null_score = cmcm.cat_wins_mae(null_simulated_data_gold, null_simulated_data_preds1, null_simulated_data_preds2)
            alt_score = cmcm.cat_wins_mae(alt_simulated_data_gold, alt_simulated_data_preds1, alt_simulated_data_preds2)
        if metric=="KL-Div":
            null_score = cmcm.cat_kl_div(null_simulated_data_gold, null_simulated_data_preds1, null_simulated_data_preds2)
            alt_score = cmcm.cat_kl_div(alt_simulated_data_gold, alt_simulated_data_preds1, alt_simulated_data_preds2)
        
        null_scores.append(null_score)
        alt_scores.append(alt_score)
    
    return  null_scores, alt_scores


def process_scores(null_scores, alt_scores):
    null_scores_m1 = [x[0] for x in null_scores]
    null_scores_m2 = [x[1] for x in null_scores]
    alt_scores_m1 = [x[0] for x in alt_scores]
    alt_scores_m2 = [x[1] for x in alt_scores]

    gamma_null_scores = np.array(null_scores_m1) - np.array(null_scores_m2)
    gamma_alt_scores = np.array(alt_scores_m1) - np.array(alt_scores_m2)
    
    return gamma_null_scores, gamma_alt_scores

# %%
def calculate_ci(scores, confidence_level=0.95):
    alpha = 1-confidence_level
    g_hat = scores[0]
    lower = np.percentile(scores, 100*alpha/2)
    upper = np.percentile(scores, 100*(1-alpha/2))
    return 2*g_hat-upper, 2*g_hat-lower

def calculate_per_ci(scores, confidence_level=0.95):
    alpha = 1-confidence_level
    lower = np.percentile(scores, 100*alpha/2)
    upper = np.percentile(scores, 100*(1-alpha/2))
    return lower, upper


# %%
def get_bootstrap_result(gamma_null_scores, gamma_alt_scores, confidence_level=0.95, method='BCa'):
    
    null_data = (gamma_null_scores,)
    alt_data = (gamma_alt_scores,)
    null_res = bootstrap(null_data, np.mean, confidence_level=confidence_level, method=method)
    alt_res = bootstrap(alt_data, np.mean, confidence_level=confidence_level, method=method)

    return null_res, alt_res

# %%
def plot_ci(alt_ci_nk_list, nk_list, x_list, metric, distortion, dataset, col, base_path):

    plt.figure(figsize=(12, 8))

    title_fontsize = 30
    label_fontsize = 30
    tick_fontsize = 30
    legend_fontsize = 30


    for idx, nk in enumerate(nk_list):
        if len(alt_ci_nk_list) > idx:
            ci_lower = [x[0].low for x in alt_ci_nk_list[idx]]
            ci_upper = [x[0].high for x in alt_ci_nk_list[idx]]
            stat_means = [x[1] for x in alt_ci_nk_list[idx]]
            plt.plot(x_list[idx], stat_means, marker='o', label=f'NxK={nk}')
            # plt.fill_between([x[1] for x in params_list[0][:35]], ci_lower, ci_upper, color='blue', alpha=0.2, label=f'{int(confidence_level*100)}% Confidence Interval')
            plt.fill_between(x_list[idx], ci_lower, ci_upper, alpha=0.2)
            # break

    plt.xlabel(col, fontsize=label_fontsize)
    plt.ylabel('CI', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(f"CI - {dataset} ({metric}, $\\epsilon$={distortion})", fontsize=title_fontsize)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncols=3, fontsize=legend_fontsize)
    plt.savefig(f"{base_path}/{dataset}_CI_{metric}_{col}_500_e_{distortion}.pdf", bbox_inches='tight')
    plt.close()

# %%
def plot_ci_width(alt_ci_nk_list, nk_list, x_list, metric, distortion, dataset, col, base_path):

    plt.figure(figsize=(12, 8))

    title_fontsize = 30
    label_fontsize = 30
    tick_fontsize = 30
    legend_fontsize = 30

    for idx, nk in enumerate(nk_list):
        if len(alt_ci_nk_list) > idx:
            ci_lower = [x[0].low for x in alt_ci_nk_list[idx]]
            ci_upper = [x[0].high for x in alt_ci_nk_list[idx]]
            ci_width = np.array(ci_upper) - np.array(ci_lower)
            plt.plot(x_list[idx], ci_width, marker='o', label=f'NxK={nk}')
            # break

    plt.xlabel(col, fontsize=label_fontsize)
    plt.ylabel('CI', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(f"CI - {dataset} ({metric}, $\\epsilon$={distortion})", fontsize=title_fontsize)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncols=3, fontsize=legend_fontsize)
    plt.savefig(f"{base_path}/{dataset}_CI_width_{metric}_{col}_500_e_{distortion}.pdf", bbox_inches='tight')
    plt.close()

# %%
def write_ci_to_file(ci, output_filename):
    write_start_time = datetime.datetime.now()
    
    with open(output_filename, "wb") as f:
        compress_pickle.dump(ci, f)
        
    elapsed_time = datetime.datetime.now() - write_start_time
    logging.info("File writing time=%f", elapsed_time.total_seconds())

# %%

def run_experiment_with_dist(distortion, params_list, nk_list, dataset, exp_dir, _M_CATEGORIES, col, base_path, ci_path, confidence_level):
    x_list = []
    alt_ci_nk_dict = {}
    for i, nks in tqdm(enumerate(params_list), desc="Calculating CIs", leave=False):
        xs = []
        alt_ci_dict = {}
        for n_items, k_responses in tqdm(nks[:35], desc="Processing", leave=False):

            response_sets = psample.read_samples_from_file(f"{exp_dir}cat_responses_simulated_distr_dist={distortion}_gen_N={n_items}_K={k_responses}_M={_M_CATEGORIES}_num_samples=1000.pkl.lz4", True)
            response_sets.truncate(n_items, k_responses)

            for metric in metrics_list:
                null_scores, alt_scores = get_metric_scores(response_sets, metric)
                gamma_null_scores, gamma_alt_scores = process_scores(null_scores, alt_scores)
                null_res, alt_res = get_bootstrap_result(gamma_null_scores, gamma_alt_scores)
                
                alt_ci = alt_res.confidence_interval
                
                if metric in alt_ci_dict and isinstance(alt_ci_dict[metric], list):
                    # alt_ci_dict[metric].append(alt_ci)
                    alt_ci_dict[metric].append((alt_ci, np.mean(gamma_alt_scores)))
                    # alt_ci_dict[metric].append((alt_ci, np.mean(alt_res.bootstrap_distribution)))
                else:
                    alt_ci_dict[metric] = [(alt_ci, np.mean(gamma_alt_scores))]
                
            xs.append(k_responses)
            # break
        
        for metric in alt_ci_dict:
            if metric in alt_ci_nk_dict and isinstance(alt_ci_nk_dict[metric], list):
                alt_ci_nk_dict[metric].append(alt_ci_dict[metric])
            else:
                alt_ci_nk_dict[metric] = [alt_ci_dict[metric]]
        
        x_list.append(xs)
        # break

    for metric in alt_ci_nk_dict:
        plot_ci(alt_ci_nk_dict[metric], nk_list, x_list, metric, distortion, dataset, col, base_path)

        plot_ci_width(alt_ci_nk_dict[metric], nk_list, x_list, metric, distortion, dataset, col, base_path)

    write_ci_to_file(alt_ci_nk_dict, f"{ci_path}/ci_level={confidence_level}_col={col}_500_dist={distortion}.pkl.lz4")

# %%
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_dices/", "dataset": "DICES", "num_categories": "3",}
dataset_info = {"exp_dir": "../../../../data/ptest_arr_d3code/", "dataset": "D3code", "num_categories": "2",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_jobsQ1/", "dataset": "JobsQ1", "num_categories": "5",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_jobsQ3/", "dataset": "JobsQ3", "num_categories": "12",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_toxicity/", "dataset": "Toxicity", "num_categories": "2",}


_M_CATEGORIES = dataset_info['num_categories']
exp_dir = dataset_info['exp_dir']
dataset = dataset_info['dataset']


# distortion=0.1
distortion_values = [0.1, 0.2, 0.3, 0.4]

confidence_level = 0.95
col='K'

metrics_list = ['Accuracy', 'MAE', 'Wins', 'KL-Div']
# metric = metrics_list[1]

base_path = f"output/ci_plots/{dataset}"
if not os.path.exists(base_path):
    os.mkdir(base_path)

ci_path = f"{exp_dir}ci"
if not os.path.exists(ci_path):
    os.mkdir(ci_path)


start_time = datetime.datetime.now()

args = [(distortion, params_list, nk_list, dataset, exp_dir, _M_CATEGORIES, col, base_path, ci_path, confidence_level) for distortion in distortion_values]
with mp.Pool(mp.cpu_count()) as pool:
    pool.starmap(run_experiment_with_dist, args)

elapsed_time = datetime.datetime.now() - start_time
print("File writing time=%f", elapsed_time.total_seconds())
