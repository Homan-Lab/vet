"""Copyright 2022 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Annotator P-Value Experimentation Suite.

Run args:
  --n_items: Number of items.
  --k_responses: Number of responses per item. Must be no greater than number
        of responses per items in the input_data dataset.
  --line_num: Line of configuration file to run, or (default) -1 for all lines.
  --exp_dir: Path to experiment directory for input, configuration, and output
        files.
  --config_file: Filename of configuration parameters for running experiments.
  --response_file: Filename containing responses generated by
      parameterized_sample over which to run simulation.
"""
import random as rand

from absl import app
from absl import flags
from absl import logging
import response_resampler_lib as resampler_lib

_N_ITEMS = flags.DEFINE_integer("n_items", 100, "Number of items.")

_K_RESPONSES = flags.DEFINE_integer(
    "k_responses", 5, "Number of responses per item. Must be no greater than "
    "number of responses per items in the input_data dataset.")
_LINE_NUM = flags.DEFINE_integer("line_num", -1, "Line of experiment file.")
_CONFIG_FILE = flags.DEFINE_string(
    "config_file",
    "config_N=1000_K=5_n_trials=1000.csv",
    "Config file for running experiments located in exp_dir/config/.",
)
_INPUT_RESPONSE_FILE = flags.DEFINE_string(
    "input_response_file",
    "responses_simulated_distr_dist=0.3_gen_N=1000_K=5_n_samples=1000.json",
    "File name in <exp_dir> containing the output of parameterized_sample to"
    "resample from.",
)
_EXP_DIR = flags.DEFINE_string(
    "exp_dir",
    "/tmp/ptest/",
    "Path name to experiment directory.",
)
_USE_PICKLE = flags.DEFINE_boolean(
    "use_pickle",
    False,
    "If true load the data using pickle. Otherwise load using json."
    "Pickle is much faster as it saves the data in binary format.",
)
_RANDOM_SEED = flags.DEFINE_integer(
    "random_seed",
    None,
    "When set, it generates the data in deterministically across runs.",
)

def main(_):
  logging.info(
      "Running ptest experiments with command line arguments:"
      "n_items = %d, k_responses = %d, "
      "line = %d, config = %s,"
      " input_data = %s",
      _N_ITEMS.value,
      _K_RESPONSES.value,
      _LINE_NUM.value,
      _CONFIG_FILE.value,
      _INPUT_RESPONSE_FILE.value,
  )

  # Set random seeds for deterministic data generation.
  if _RANDOM_SEED.value:
    rand.seed(_RANDOM_SEED.value)

  experiments_manager = resampler_lib.ExperimentsManager(
      exp_dir=_EXP_DIR.value,
      input_response_file=_INPUT_RESPONSE_FILE.value,
      use_pickle=_USE_PICKLE.value,
      line=_LINE_NUM.value,
      config_file_name=_CONFIG_FILE.value,
      n_items=_N_ITEMS.value,
      k_responses=_K_RESPONSES.value,
  )

  logging.info("Experiments set up. Getting ready to run.")
  experiments_manager.run_experiments()

if __name__ == "__main__":
  app.run(main)
