# python3 categorical_sample.py \
#   --n_items=5000 \
#   --k_responses=100 \
#   --num_samples=1000 \
#   --distortion=0.1 \
#   --use_pickle=true \
#   --exp_dir=../../../../data/ptest_arr/

# python3 categorical_sample.py \
#   --n_items=350 \
#   --k_responses=123 \
#   --num_samples=1 \
#   --distortion=0.1 \
#   --use_pickle=true \
#   --exp_dir=../../../../data/ptest_arr/

# python3 categorical_sample.py \
#   --n_items=5000 \
#   --k_responses=100 \
#   --m_categories=2 \
#   --alpha="6.08113935,2.88368607"\
#   --noise_parameters="0.5,0.5"\
#   --num_samples=1000 \
#   --distortion=0.1 \
#   --use_pickle=true \
#   --exp_dir=../../../../data/ptest_arr_d3code/

# python3 categorical_sample.py \
#   --n_items=4554 \
#   --k_responses=23 \
#   --m_categories=2 \
#   --alpha="6.08113935,2.88368607"\
#   --noise_parameters="0.5,0.5"\
#   --num_samples=1 \
#   --distortion=0.1 \
#   --use_pickle=true \
#   --exp_dir=../../../../data/ptest_arr_d3code/

# python3 categorical_sample.py \
#   --n_items=5000 \
#   --k_responses=100 \
#   --m_categories=5 \
#   --alpha="1039.76103107,38.24111517,35.57247128,310.28721781,46.02098984"\
#   --noise_parameters="0.2,0.2,0.2,0.2,0.2"\
#   --num_samples=1000 \
#   --distortion=0.1 \
#   --use_pickle=true \
#   --exp_dir=../../../../data/ptest_arr_jobsQ1/

  python3 categorical_sample.py \
  --n_items=500 \
  --k_responses=10 \
  --m_categories=3 \
  --dir_prior_prob_dist="uniform"\
  --dir_prior_prob_dist_params="0,1"\
  --num_samples=1 \
  --distortion=0.1 \
  --use_pickle=true \
  --exp_dir=../../../../data/ptest_arr_uniform/
  