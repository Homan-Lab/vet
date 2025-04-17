pairs=("5000 1" "1000 5" "500 10" "333 15" "250 20" "200 25" "166 30" "142 35" "125 40" "111 45" "100 50" "90 55" "83 60" "76 65" "71 70" "66 75" "62 80" "58 85" "55 90" "52 95" "50 100")

# python categorical_sample.py --n_items=5000 --k_responses=100 --num_samples=1000 --distortion=0.1 --use_pickle=true --exp_dir=../../../../data/ptest_arr/

# Loop over the pairs and extract each value
for pair in "${pairs[@]}"; do
    # Split the pair into two values
    IFS=' ' read -r n_items k_responses <<< "$pair"
    
	n_items=${n_items%.*}
	k_responses=${k_responses%.*}
    echo "Value1: $n_items, Value2: $k_responses"
	for d in 0.1 0.2 0.4
	do
		# python categorical_sample.py --n_items=${n_items} --k_responses=${k_responses} --num_samples=1000 --distortion=${d} --use_pickle=true --exp_dir=../ptest/
		# python response_resampler.py --line_num=-1 --n_items=${n_items} --k_responses=${k_responses} --config_file=example_config.csv --use_pickle=true --exp_dir=../../../../data/ptest_arr/ --input_response_file=cat_responses_simulated_distr_dist=${d}_gen_N=5000_K=100_M=3_num_samples=1000.pkl
		# python response_resampler.py --line_num=-1 --n_items=${n_items} --k_responses=${k_responses} --config_file=example_config.csv --use_pickle=true --exp_dir=../../../../data/ptest_arr_d3code/ --input_response_file=cat_responses_simulated_distr_dist=${d}_gen_N=5000_K=100_M=2_num_samples=1000.pkl
		python response_resampler.py --line_num=-1 --n_items=${n_items} --k_responses=${k_responses} --config_file=example_config.csv --use_pickle=true --exp_dir=../../../../data/ptest_arr_jobsQ1/ --input_response_file=cat_responses_simulated_distr_dist=${d}_gen_N=5000_K=100_M=5_num_samples=1000.pkl
	done
	# break
done