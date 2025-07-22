# %%
import os
from tqdm import tqdm
import pickle
import compress_pickle

# %%
# dataset_info = {"exp_dir": "../ptest/",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr/",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_dices/", "dataset": "DICES", "num_categories": "3",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_d3code/", "dataset": "D3code", "num_categories": "2",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_jobsQ1/", "dataset": "JobsQ1", "num_categories": "5",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_jobsQ3/", "dataset": "JobsQ3", "num_categories": "12",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_toxicity/", "dataset": "toxicity", "num_categories": "2",}

# dataset_info = {"exp_dir": "../ptest_arr_uniform/", "dataset": "uniform", "num_categories": "2",}
# dataset_info = {"exp_dir": "../ptest_arr_gamma/", "dataset": "gamma", "num_categories": "3",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_uniform/", "dataset": "uniform", "num_categories": "2",}
# dataset_info = {"exp_dir": "../../../../data/ptest_arr_gamma/", "dataset": "gamma", "num_categories": "3",}
dataset_info = {"exp_dir": "../../../../data/ptest_arr_dices_actual_p/", "dataset": "DICES actual p-vals", "num_categories": "3",}

# %%
# # exp_dir = "../ptest/" 
# # exp_dir = "../../../../data/ptest_arr/"
# # file_names = [(os.path.join(dataset_info['exp_dir'],x), os.path.getsize(os.path.join(dataset_info['exp_dir'],x))/(1024*1024)) for x in os.listdir(dataset_info['exp_dir']) if x.endswith('.pkl')]
# file_names = [os.path.join(dataset_info['exp_dir'],x) for x in os.listdir(dataset_info['exp_dir']) if x.endswith('.pkl')]
# file_names, len(file_names)

# %%
def read_pickle(file_name, open_mode="rb"):
    with open(file_name, open_mode) as f:
        data = pickle.load(f)
    return data

# %%
def write_compress_pickle(data, file_name, open_mode="wb"):
    with open(file_name, open_mode) as f:
        compress_pickle.dump(data, f)

# %%
def read_compress_pickle(file_name, open_mode="rb"):
    with open(file_name, open_mode) as f:
        data = compress_pickle.load(f)
    return data

# %%
# data = read_pickle(file_names[0])
# len(data)

# %%
# output_exp_dir = "../../../../data/test/"
# # output_filename = os.path.join(output_exp_dir,"test.pkl.gz")
# output_filename = os.path.join(output_exp_dir,"test.pkl.bz2")
# # output_filename = os.path.join(output_exp_dir,"test.pkl.xz")
# # output_filename = os.path.join(output_exp_dir,"test.pkl.lz4")

# write_compress_pickle(data, output_filename)

# %%
# 570M
# 13m 28.4s - 46.2M
# 2m 13s - 40.1M
# 6m 33.4s - 38.8M
# 4s - 180M

# %%
compression = "lz4"
file_list = os.listdir(dataset_info['exp_dir'])
for x in tqdm(file_list):
    if x.endswith('.pkl'):
        file_name = os.path.join(dataset_info['exp_dir'], x)
        try:
            data = read_pickle(file_name)
            output_filename = os.path.join(dataset_info['exp_dir'],f"{x}.{compression}")
            write_compress_pickle(data, output_filename)
            os.remove(file_name)
        except:
            print(f"Error: {file_name}, Size: {os.path.getsize(file_name)}")
        # break

# %%
# compressed_data = read_compress_pickle(output_filename)
# # data==compressed_data

# %%
# 192


