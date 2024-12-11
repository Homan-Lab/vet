"""
Example usage:

python categorical_sample --exp_dir=/data_dir/path --distortion=.02
"""
import datetime
import os
import random as rand
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np
from parameterized_sample_lib import write_samples_to_file
import categorical_sample_lib as cat_sample

_DISTORTION = flags.DEFINE_float(
    "distortion", 0.3, "Amount of distortion between machines."
)
_EXP_DIR = flags.DEFINE_string(
    "exp_dir", "../ptest/",
    "The file path where the experiment input and output data are located."
)
_N_ITEMS = flags.DEFINE_integer(
    "n_items", 1000, "Number of items per response set."
)
_K_RESPONSES = flags.DEFINE_integer(
    "k_responses", 10, "Number of responses per item."
)
_M_CATEGORIES = flags.DEFINE_integer(
    "m_categories", 3, "Number of categories."
)
_ALPHA = flags.DEFINE_list(
    "alpha",
    [5.21765954, 0.85824731, 2.74833849],
    "Parameters for the dirichlet distribution",
)
_NOISE_PARAMS = flags.DEFINE_list(
    "noise_parameters",
    [0.333, 0.333, 0.334],
    "Parameters for the noise distribution",
)
_NUM_SAMPLES = flags.DEFINE_integer(
    "num_samples", 1000, "Number of sample sets per experiment."
)
_USE_PICKLE = flags.DEFINE_boolean(
    "use_pickle",
    False,
    "If true use pickle to save data. Otherwise use json."
    "Pickle is much faster as it saves the data in binary format.",
)
_RANDOM_SEED = flags.DEFINE_integer(
    "random_seed",
    None,
    "When set, it generates the data in deterministically across runs.",
)

# for how to use this library.
def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  # Set random seeds for deterministic data generation.
  if _RANDOM_SEED.value:
    rand.seed(_RANDOM_SEED.value)
    np.random.seed(_RANDOM_SEED.value)

  generation_start_time = datetime.datetime.now()
  response_sets = cat_sample.simulate_response_tables_cat(
      _N_ITEMS.value,
      _K_RESPONSES.value,
      _M_CATEGORIES.value,
      _ALPHA.value,
      _NOISE_PARAMS.value,
      _DISTORTION.value,
      _NUM_SAMPLES.value,
  )
  elapsed_time = datetime.datetime.now() - generation_start_time
  logging.info("Data generation time=%f", elapsed_time.total_seconds())

  file_extension = "pkl" if _USE_PICKLE.value else "json"
  if not os.path.exists(_EXP_DIR.value):
    os.mkdir(_EXP_DIR.value)
  output_filename = os.path.join(
      _EXP_DIR.value,
      f"cat_responses_simulated_distr_dist={_DISTORTION.value}_gen_N="
      f"{_N_ITEMS.value}_K={_K_RESPONSES.value}_M={_M_CATEGORIES.value}"
      f"_num_samples={_NUM_SAMPLES.value}.{file_extension}",
  )
  write_samples_to_file(
      response_sets, output_filename, _USE_PICKLE.value
  )

if __name__ == "__main__":
  app.run(main)
