for n_items in 10 25 50 100 250 500 1000 2000
do
	for k_responses in 1 5 10 25 50 100
	do
		for d in 0.005 0.01 0.02 0.1 0.2
		do
			# python categorical_sample.py --n_items=${n_items} --k_responses=${k_responses} --num_samples=1000 --distortion=${d} --use_pickle=true --exp_dir=../ptest/
			python response_resampler.py --line_num=-1 --n_items=${n_items} --k_responses=${k_responses} --config_file=example_config.csv --use_pickle=true --exp_dir=../ptest/ --input_response_file=cat_responses_simulated_distr_dist=${d}_gen_N=${n_items}_K=${k_responses}_M=3_num_samples=1000.pkl
		done
	done
done
