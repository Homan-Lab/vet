# %%
import os 
import numpy as np 
import pandas as pd 
from absl import app 
from absl import flags 
import compress_pickle
from typing import Sequence 
import matplotlib.pyplot as plt

# %%
params_list = [] 
all_params_list = [] 
ks = [x for x in range(1,11)] 
# ks = [x for x in range(120,501, 20)]
# ks.extend([x for x in range(20,1001, 20)])
ks.extend([x for x in range(20,501, 20)])
# ks.extend([x for x in range(20,101, 20)])

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
# nk_list = [5000]
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
        intermediate_table['M1 GT Alt'] = experiment_results['M1 GT Alt']
        intermediate_table['M2 GT Alt'] = experiment_results['M2 GT Alt']
        # intermediate_table['Metric'] = ['$\\Gamma_{\\rm Accuracy}$', '$\\Gamma_{\\rm F1-score}$']  
        intermediate_table['Metric'] = metrics_list 
        # intermediate_table['Metric'] = ['Accuracy'] 
        intermediate_table[f'$\\epsilon$'] = distortion 
        final_table = pd.concat([final_table, intermediate_table]) 
 
    final_table = final_table.melt(["Metric", "$\\epsilon$"]).sort_values(by=["Metric","variable"]).pivot(index = "$\\epsilon$", columns=["Metric","variable"]) 
    # final_table = final_table.reset_index(drop=True) 
    final_table = final_table.reset_index() 
    final_table.columns = pd.MultiIndex.from_tuples([(j,k) for i,j,k in final_table.columns]) 
    final_table.columns = ['\\_'.join(col) for col in final_table.columns] 
    final_table["N"] = pd.Series([_N_ITEMS]*len(distortion_values)) 
    final_table["K"] = pd.Series([_K_RESPONSES]*len(distortion_values)) 
    # final_table["NxK"] = final_table["N"]*final_table["K"] 
 
    return final_table

# %%
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_dices/", "dataset": "DICES", "num_categories": "3",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_d3code/", "dataset": "D3code", "num_categories": "2",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_jobsQ1/", "dataset": "JobsQ1", "num_categories": "5",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_jobsQ3/", "dataset": "JobsQ3", "num_categories": "12",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_toxicity/", "dataset": "Toxicity", "num_categories": "2",}

# dataset_info = {"exp_dir": "../ptest_arr_uniform/", "dataset": "uniform", "num_categories": "2",}
# dataset_info = {"exp_dir": "../ptest_arr_gamma/", "dataset": "gamma", "num_categories": "3",}

# datasets = [{"exp_dir": "../../../../data/ptest_arr_toxicity/", "dataset": "Toxicity", "num_categories": "2",},
#             {"exp_dir": "../../../../data/ptest_arr_dices/", "dataset": "DICES", "num_categories": "3",},
#             {"exp_dir": "../../../../data/ptest_arr_d3code/", "dataset": "D3code", "num_categories": "2",},
#             {"exp_dir": "../../../../data/ptest_arr_jobsQ1/", "dataset": "JobsQ1", "num_categories": "5",},
#             {"exp_dir": "../../../../data/ptest_arr_jobsQ3/", "dataset": "JobsQ3", "num_categories": "12",},]

datasets = [{"exp_dir": "../ptest_arr_uniform/", "dataset": "Balanced (M=2)", "num_categories": "2",},
            {"exp_dir": "../ptest_arr_uniform/", "dataset": "Balanced (M=3)", "num_categories": "3",},
            {"exp_dir": "../ptest_arr_uniform/", "dataset": "Balanced (M=4)", "num_categories": "4",},
            {"exp_dir": "../ptest_arr_uniform/", "dataset": "Balanced (M=5)", "num_categories": "5",},
            {"exp_dir": "../../../../shared/rc/population/ptest_arr_uniform/", "dataset": "Balanced (M=12)", "num_categories": "12",},
            {"exp_dir": "../ptest_arr_gamma/", "dataset": "Unbalanced (M=2)", "num_categories": "2",},
            {"exp_dir": "../ptest_arr_gamma/", "dataset": "Unbalanced (M=3)", "num_categories": "3",},
            {"exp_dir": "../ptest_arr_gamma/", "dataset": "Unbalanced (M=4)", "num_categories": "4",},
            {"exp_dir": "../ptest_arr_gamma/", "dataset": "Unbalanced (M=5)", "num_categories": "5",},
            {"exp_dir": "../../../../shared/rc/population/ptest_arr_gamma/", "dataset": "Unbalanced (M=12)", "num_categories": "12",},]

# %%
col = "K"
distortion = 0.3
metrics_list = ['Accuracy', 'MAE', 'Wins', 'KL-Div']

confidence_level = 0.95
nk_list_idx=0

# %%
# all_dfs = {}
all_dfs = []

for dataset_info in datasets:
    _M_CATEGORIES = dataset_info['num_categories']
    exp_dir = dataset_info['exp_dir']
    dataset = dataset_info['dataset']

    errors = []
    dfs_nk_list = []
    for i, nks in enumerate(params_list):
        data_df_list = []
        # for n,k in params_list[nk_list_idx][:35]:
        for n,k in nks[:35]:
            try:
                data_df = gather_data(n, k, [distortion], exp_dir, metrics_list, _M_CATEGORIES)
                data_df_list.append(data_df)
            except FileNotFoundError:
                errors.append((n,k))
                # print(f"File not found!: {n,k}")
            except:
                errors.append((n,k))
                print(f"Some exception occured!: {n,k}")

        if data_df_list:
            df_ratings_nk = pd.concat(data_df_list)
            df_ratings_nk = df_ratings_nk.reset_index(drop=True)
            df_ratings_nk["NK"] = nk_list[i]
            dfs_nk_list.append(df_ratings_nk)

    # # all_dfs[dataset] = dfs_nk_list
    # all_dfs[dataset] = pd.concat(dfs_nk_list, ignore_index=True)

    all_dfs.append(pd.concat(dfs_nk_list, ignore_index=True))

    print(f"Distortion: {distortion}, Error list len: {len(errors)}")

    # break

print(len(all_dfs))

# %%
# all_dfs[0]

# %%
# metric = 'Accuracy'
# ls = all_dfs[-1][f'{metric}\\_p-value']
# met_df = all_dfs[-1].iloc[ls.index[ls<=0.05]]
# min_nk = met_df['NK'].min()
# nk_df = met_df[met_df['NK']==min_nk]
# min_k = nk_df['K'].min()
# nk_df

# %%
ci_dfs = []

for dataset_info in datasets:
    _M_CATEGORIES = dataset_info['num_categories']
    exp_dir = dataset_info['exp_dir']
    dataset = dataset_info['dataset']

    ci_path_1 = f"{exp_dir}ci_1000"
    ci_path_2 = f"{exp_dir}ci"

    
    alt_ci_nk_dict_1 = compress_pickle.load(f"{ci_path_1}/ci_level={confidence_level}_col={col}_500_dist={distortion}.pkl.lz4")
    # print(len(alt_ci_nk_dict_1), len(alt_ci_nk_dict_1['Accuracy']), len(alt_ci_nk_dict_1['Accuracy'][0]))
    alt_ci_nk_dict_2 = compress_pickle.load(f"{ci_path_2}/ci_level={confidence_level}_col={col}_500_dist={distortion}.pkl.lz4")
    # print(len(alt_ci_nk_dict_2), len(alt_ci_nk_dict_2['Accuracy']), len(alt_ci_nk_dict_2['Accuracy'][0]))

    ci_metric_dfs_dict = {}

    for metric in metrics_list:
        if metric in alt_ci_nk_dict_1 and metric in alt_ci_nk_dict_2:
            alt_ci_nk_list = alt_ci_nk_dict_1[metric] + alt_ci_nk_dict_2[metric]
            # print(len(alt_ci_nk_list))

            dfs_nk_list = []

            for nk_list_idx, nks in enumerate(params_list):
                ci_rows = []
                for idx, (n_items, k_responses) in enumerate(nks[:15]):
                    # print(idx, n_items, k_responses)
                    ci_lower, ci_upper = alt_ci_nk_list[nk_list_idx][idx][0]
                    mean_score = alt_ci_nk_list[nk_list_idx][idx][1]
                    ci_rows.append({'ci_lower':ci_lower, 'ci_upper':ci_upper, 'ci_width':ci_upper-ci_lower, 'mean_score':mean_score, 'N':n_items, 'K':k_responses, 'NK':nk_list[nk_list_idx]})
                    # break

                dfs_nk_list.append(pd.DataFrame(ci_rows))
            
            ci_metric_dfs_dict[metric] = pd.concat(dfs_nk_list, ignore_index=True)
    #     break
    ci_dfs.append(ci_metric_dfs_dict)
    # break

print(len(ci_dfs))


# %%
base_path = "output/tables_min_nk"
# base_path = f"output/tables_{nk_list[nk_list_idx]}"
if not os.path.exists(base_path):
    os.makedirs(base_path)

# table_name = os.path.join(base_path, "table.tex")
# df.to_latex(table_name, index=False, float_format="%.4f")
# for idx, dataset_info in enumerate(datasets):
#     dataset = dataset_info['dataset']
#     sub_df = all_dfs[idx][['N', 'K', 'Accuracy\\_p-value', 'Accuracy\\_$\\Delta$', 'Accuracy\\_M1 GT Alt', 'Accuracy\\_M2 GT Alt']]
#     sub_df = sub_df[sub_df['K']<=100]
#     table_name = os.path.join(base_path, f"{dataset}_accuracy_table_k_100.tex")
#     sub_df.to_latex(table_name, index=False, float_format="%.4f")
# %%
results_list = []
results_list_2 = []
for idx, df in enumerate(all_dfs):
    results_nk = {'Dataset': f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'NK'}
    results_k = {'Dataset': f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'K'}
    results_delta = {'Dataset': f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'$\\Delta$'}

    results_nk_2 = {'Dataset': f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'NK'}
    results_pval_2 = {'Dataset':f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'p-value'}
    results_k_2 = {'Dataset': f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'K'}
    results_delta_2 = {'Dataset': f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'$\\Delta$'}
    
    for metric in metrics_list:
        ls = df[f'{metric}\\_p-value']
        met_df = df.iloc[ls.index[ls<=0.05]]
        rows_nk = met_df['NK']
        
        if len(rows_nk)==0:
            results_nk[metric] = '-'
            results_k[metric] = '-'
            results_delta[metric] = '-'
        else:
            min_nk = rows_nk.min()
            nk_df = met_df[met_df['NK']==min_nk]
            min_k = nk_df['K'].min()

            results_nk[metric] = int(min_nk)
            results_k[metric] = int(min_k)
            results_delta[metric] = df.iloc[nk_df['K'].idxmin()][f'{metric}\\_$\\Delta$']

            results_pval_2[metric] = nk_df[f'{metric}\\_p-value'].min()
            min_p_index = nk_df[f'{metric}\\_p-value'].idxmin()
            k_value = df.iloc[min_p_index]['K']
            results_k_2[metric] = int(k_value)
            results_nk_2[metric] = int(min_nk)
            results_delta_2[metric] = df.iloc[min_p_index][f'{metric}\\_$\\Delta$']
            
    results_list.append(results_nk)
    results_list.append(results_k)
    results_list.append(results_delta)

    results_list_2.append(results_nk_2)
    results_list_2.append(results_pval_2)
    results_list_2.append(results_k_2)
    results_list_2.append(results_delta_2)
    # print(results)

res_df = pd.DataFrame(results_list)
res_df_2 = pd.DataFrame(results_list_2)
# res_df

# %%
table_name = os.path.join(base_path, "low_k_for_p_lte_05_nk.tex")
res_df.to_latex(table_name, index=False, float_format="%.4f")

# %%
results_list = []
for idx, df in enumerate(all_dfs):
    results_pval = {'idx':idx, 'Dataset':f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'p-value'}
    results_nk = {'idx':idx, 'Dataset': f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'NK'}
    results_k = {'idx':idx, 'Dataset':f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']}", 'Stat':'K'}
    results_delta = {'idx':idx, 'Dataset':f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']}", 'Stat':'$\\Delta$'}

    for metric in metrics_list:
        results_pval[metric] = df[f'{metric}\\_p-value'].min()
        min_p_index = df[f'{metric}\\_p-value'].idxmin()
        k_value = df.iloc[min_p_index]['K']
        nk_value = df.iloc[min_p_index]['NK']
        results_k[metric] = int(k_value)
        results_nk[metric] = int(nk_value)
        results_delta[metric] = df.iloc[min_p_index][f'{metric}\\_$\\Delta$']
    results_list.append(results_pval)
    results_list.append(results_nk)
    results_list.append(results_k)
    results_list.append(results_delta)
    # print(results)

res_df = pd.DataFrame(results_list)
res_df = res_df.set_index('idx')
# res_df

table_name = os.path.join(base_path, "k_delta_low_p_nk.tex")
res_df.to_latex(table_name, index=False, float_format="%.4f", multirow=True)

results_list = []
for idx, ci_dict in enumerate(ci_dfs):
    results_ci = {'idx':idx, 'Dataset':f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'ci-width'}
    results_nk = {'idx':idx, 'Dataset': f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'NK'}
    results_k = {'idx':idx, 'Dataset':f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'K'}
    results_delta = {'idx':idx, 'Dataset':f"{datasets[idx]['dataset']} (M={datasets[idx]['num_categories']})", 'Stat':'$\\Delta$'}
    p_df = all_dfs[idx]

    for metric in metrics_list:
        df = ci_dict[metric]
        # print(df)
        results_ci[metric] = df['ci_width'].min()
        min_p_index = df['ci_width'].idxmin()
        k_value = df.iloc[min_p_index]['K']
        results_k[metric] = int(k_value)
        results_nk[metric] = df.iloc[min_p_index]['NK']
        results_delta[metric] = p_df.iloc[min_p_index][f'{metric}\\_$\\Delta$']
        # break
    results_list.append(results_ci)
    results_list.append(results_nk)
    results_list.append(results_k)
    results_list.append(results_delta)
    # print(results)
    # break

res_df = pd.DataFrame(results_list)
res_df = res_df.set_index('idx')
# res_df

table_name = os.path.join(base_path, f"k_low_ci_nk.tex")
res_df.to_latex(table_name, index=False, float_format="%.4f", multirow=True)