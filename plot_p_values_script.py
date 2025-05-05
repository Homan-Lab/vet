# %%
import os 
import numpy as np 
import pandas as pd 
from absl import app 
from absl import flags 
from typing import Sequence 
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
def get_bash_params_list(all_params_list):
    # print(all_params_list)
    params_str = str(all_params_list)
    params_str = params_str.replace('[','')
    params_str = params_str.replace(']','')
    params_str = params_str.replace(',','')
    params_str = params_str.replace('(','"')
    params_str = params_str.replace(')','"')
    params_str = "(" + params_str + ")"
    return params_str

# params_str = get_bash_params_list(all_params_list)
# print(params_str)

# %%
len(params_list), len(ratings_list), len(all_params_list)

# %%
def gather_data(_N_ITEMS, _K_RESPONSES, distortion_values, exp_dir, metrics_list, _M_CATEGORIES, actual_p=False): 
    final_table = pd.DataFrame() 
    for distortion in distortion_values:
        file_path = f'{exp_dir}results_N={_N_ITEMS}_K={_K_RESPONSES}_cat_responses_simulated_distr_dist={distortion}_gen_N={_N_ITEMS}_K={_K_RESPONSES}_M={_M_CATEGORIES}_num_samples=1000.pkl.csv'
        if actual_p:
            file_path = f'{exp_dir}results_N={_N_ITEMS}_K={_K_RESPONSES}_cat_actual_responses_simulated_distr_dist={distortion}_gen_N={_N_ITEMS}_K={_K_RESPONSES}_M={_M_CATEGORIES}_num_samples=1000.pkl.csv'
        experiment_results = pd.read_csv(file_path)
        
        intermediate_table = pd.DataFrame() 
        intermediate_table['$\\Delta$'] = (experiment_results['M2 GT Alt'] - experiment_results['M1 GT Alt']).abs() 
        intermediate_table['p-value'] = experiment_results['GT_Pvalue'] 
        # intermediate_table['Metric'] = ['$\\Gamma_{\\rm Accuracy}$', '$\\Gamma_{\\rm F1-score}$']  
        intermediate_table['Metric'] = metrics_list 
        # intermediate_table['Metric'] = ['Accuracy'] 
        intermediate_table[f'$\\epsilon$'] = distortion 
        final_table = pd.concat([final_table, intermediate_table]) 
 
    final_table = final_table.melt(["Metric", "$\\epsilon$"]).sort_values(by=["Metric","variable"]).pivot(index = "$\\epsilon$", columns=["Metric","variable"]) 
    # final_table = final_table.reset_index(drop=True) 
    final_table = final_table.reset_index() 
    final_table.columns = pd.MultiIndex.from_tuples([(j,k) for i,j,k in final_table.columns]) 
    final_table.columns = ['_'.join(col) for col in final_table.columns] 
    final_table["N"] = pd.Series([_N_ITEMS]*len(distortion_values)) 
    final_table["K"] = pd.Series([_K_RESPONSES]*len(distortion_values)) 
    # final_table["NxK"] = final_table["N"]*final_table["K"] 
 
    return final_table

# %%
def plot_p_values(all_dfs, data_nk_list, metric, distortion, dataset, col, base_path):
    if all_dfs and data_nk_list:
        plt.figure(figsize=(10, 6))

        for idx, nk in enumerate(data_nk_list):
            plt.plot(all_dfs[idx][col], all_dfs[idx][f"{metric}_p-value"], marker='o', label=f'NxK={nk}') 

        plt.xlabel(col) 
        plt.ylabel('p-value') 
        # plt.ylim((0,0.6))
        # plt.ylim((0,0.015))
        # plt.ylim((0,1))
        plt.title(f"{dataset} ({metric}, $\\epsilon$={distortion})") 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncols=3) 
        # plt.savefig(f"{base_path}/{dataset}_nk_5000_{metric}_{col}_e_{distortion}.png")
        # plt.savefig(f"{base_path}/{dataset}_p_vals_{metric}_{col}_e_{distortion}.png")
        plt.savefig(f"{base_path}/{dataset}_p_vals_{metric}_{col}_500_e_{distortion}.png")
        plt.close()

# %%
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_dices/", "dataset": "DICES", "num_categories": "3",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_d3code/", "dataset": "D3code", "num_categories": "2",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_jobsQ1/", "dataset": "JobsQ1", "num_categories": "5",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_jobsQ3/", "dataset": "JobsQ3", "num_categories": "12",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_toxicity/", "dataset": "Toxicity", "num_categories": "2",}

# dataset_info = {"exp_dir": "../ptest_arr_uniform/", "dataset": "uniform", "num_categories": "2",}
dataset_info = {"exp_dir": "../ptest_arr_gamma/", "dataset": "gamma", "num_categories": "3",}

# dataset_info = {"exp_dir": "../../../../data/ptest_arr_dices_actual_p/", "dataset": "DICES actual p-vals", "num_categories": "3",}


# %%
_M_CATEGORIES = dataset_info['num_categories']
exp_dir = dataset_info['exp_dir']
dataset = dataset_info['dataset']

distortion = 0.1
distortion_values = [0.1, 0.2, 0.3, 0.4]
# distortion_values = [0.15, 0.2, 0.3, 0.4]
# distortion_values = [0.1, 0.15, 0.2, 0.3, 0.4]

metrics_list = ['Accuracy', 'MAE', 'Wins', 'KL-Div']

col = "K"
# col = "N"

actual_p = False
# base_path = "output/actual_p"

# base_path = "output/artificial"
base_path = "output"
if not os.path.exists(base_path):
    os.mkdir(base_path)

# _N_ITEMS = 5000 
# _K_RESPONSES = 1
# final_table = gather_data(_N_ITEMS, _K_RESPONSES, [distortion], exp_dir, metrics_list, _M_CATEGORIES, actual_p) 
# print(_N_ITEMS, _K_RESPONSES) 
# final_table

# %%
for distortion in distortion_values:
    all_dfs = []
    errors = []
    data_nk_list = []
    for i,nks in enumerate(params_list):
        data_df_list = []
        for n,k in nks[:35]:
        # for n,k in nks:
            try:
                data_df = gather_data(n, k, [distortion], exp_dir, metrics_list, _M_CATEGORIES, actual_p)
                data_df_list.append(data_df)
            except FileNotFoundError:
                errors.append((n,k))
                # print(f"File not found!: {n,k}")
            except:
                errors.append((n,k))
                print(f"Some exception occured!: {n,k}")

        if data_df_list:
            df_ratings_nk = pd.concat(data_df_list)
            all_dfs.append(df_ratings_nk)
            data_nk_list.append(nk_list[i])

    print(f"Distortion: {distortion}, Error list len: {len(errors)}")

    for metric in metrics_list:
        plot_p_values(all_dfs, data_nk_list, metric, distortion, dataset, col, base_path)
        # break

    # break

# %%
# params_str = get_bash_params_list(errors)
# print(params_str)

# %%



