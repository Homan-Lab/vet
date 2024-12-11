import datetime
import enum
import functools
import json
import pickle
import random as rand
from typing import Any, Callable, List, Tuple

from absl import logging
import numpy as np
import datatypes


def gen_dirichlet_samples(
    alpha: List[float],
    n: int,
) -> np.ndarray:
  """Generates n samples from a dirichlet distribution with parameters alpha.

  Args:
      alpha (List[float]): Parameters of the dirichlet distribution.
      n (int): Number of samples to generate.

  Returns:
      np.ndarray: Samples generated from a dirichlet distribution 
  """
  rng = np.random.default_rng()

  samples = rng.dirichlet(alpha, n)
  return samples

def distort_parameters_cat(
    categorical_params: np.ndarray,
    noise_params: np.ndarray,
    distortion: float,
) -> np.ndarray:
  """Distorts the parameters (linear combination)

  Args:
      categorical_params (np.ndarray): Categorical parameters
      noise_params (np.ndarray): Noise parameters
      distortion (float): distortion

  Returns:
      np.ndarray: Distorted parameters
  """
  distorted_params = (1 - distortion) * categorical_params + distortion * noise_params
  return distorted_params

def gen_alt_responses_cat(
    categorical_params: np.ndarray,
    distorted_params: np.ndarray,
    k_responses: int,
) -> Tuple[np.ndarray, np.ndarray,]:
  """Generate responses for alt hypothesis

  Args:
      categorical_params (np.ndarray): Categorical parameters
      distorted_params (np.ndarray): Distorted parameters
      k_responses (int): Number of responses

  Returns:
      Tuple[np.ndarray, np.ndarray, np.ndarray,]: Generated data for gold and two machines
  """
  rng = np.random.default_rng()
  responses_gold = np.column_stack(
    [rng.multinomial(1, categorical_params).argmax(axis=-1) for _ in range(k_responses)]
  )
  responses_y = np.column_stack(
    [rng.multinomial(1, categorical_params).argmax(axis=-1) for _ in range(k_responses)]
  )
  responses_z = np.column_stack(
    [rng.multinomial(1, distorted_params).argmax(axis=-1) for _ in range(k_responses)]
  )
  return responses_gold, responses_y, responses_z

def mix_arrays(
    array_1: np.ndarray,
    array_2: np.ndarray,
) -> np.ndarray:
  """Mixes two numpy arrays

  Args:
      array1 (np.ndarray): array one
      array2 (np.ndarray): array two

  Returns:
      np.ndarray: mixed arrays
  """
  rng = np.random.default_rng()
  choices =  rng.integers(2, size=(array_1.shape[0]))

  mixed_arrays = np.zeros(array_1.shape)
  mixed_arrays[choices==0] = array_1[choices==0]
  mixed_arrays[choices==1] = array_2[choices==1]

  return mixed_arrays

def gen_null_responses_cat(
    categorical_params_null: np.ndarray,
    distorted_params_null: np.ndarray,
    k_responses: int,
) -> Tuple[np.ndarray, np.ndarray,]:
  """Generate responses for null hypothesis

  Args:
      categorical_params_null (np.ndarray): Categorical parameters
      distorted_params_null (np.ndarray): Distorted parameters
      k_responses (int): Number of responses

  Returns:
      Tuple[np.ndarray, np.ndarray, np.ndarray,]: Generated data for gold and two machines
  """
  rng = np.random.default_rng()

  responses_gold = np.column_stack(
    [rng.multinomial(1, categorical_params_null).argmax(axis=-1) for _ in range(k_responses)]
  )
  responses_y = np.column_stack(
    [rng.multinomial(1, mix_arrays(categorical_params_null, distorted_params_null)).argmax(axis=-1) for _ in range(k_responses)]
  )
  responses_z = np.column_stack(
    [rng.multinomial(1, mix_arrays(distorted_params_null, categorical_params_null)).argmax(axis=-1) for _ in range(k_responses)]
  )
  return responses_gold, responses_y, responses_z

def simulate_response_tables_cat(
    n_items: int = 1000,
    k_responses: int = 5,
    m_categories: int = 3,
    alpha: List[float] = [0.6, 0.1, 0.3],
    noise_parameters: List[float] = [0.333, 0.333, 0.334],
    distortion: float = 0.1,
    num_samples: int = 1000,
) -> datatypes.ResponseSets:
  """Generates a collection of machine responses.

  Generates tables ("sets"), for null and alternative hypotheses

  Args:
      n_items (int, optional): Number of items per set. Defaults to 1000.
      k_responses (int, optional): Number of responses per item. Defaults to 5.
      m_categories (int, optional): Number of categories. Defaults to 3.
      alpha (List[float], optional): Categorical parameters. Defaults to [0.6, 0.1, 0.3].
      noise_parameters (List[float], optional): Noise parameters. Defaults to [0.333, 0.333, 0.333].
      distortion (float, optional): Distortion value. Defaults to 0.1.
      num_samples (int, optional): Number of samples of size n_items x k_responses. Defaults to 1000.

  Returns:
      datatypes.ResponseSets: _description_
  """
  responses_alt = []
  responses_null = []

  for _ in range(num_samples):
    categorical_params = gen_dirichlet_samples(alpha=alpha, n=n_items)
    noise_params = gen_dirichlet_samples(alpha=noise_parameters, n=n_items)

    distorted_params = distort_parameters_cat(categorical_params, noise_params, distortion)

    responses_gold, responses_y, responses_z = gen_alt_responses_cat(categorical_params, distorted_params, k_responses)

    responses_alt.append(
        datatypes.ResponseData(
            gold=responses_gold, preds1=responses_y, preds2=responses_z
        )
    )
    # print(responses_gold.shape, responses_y.shape, responses_z.shape)
    # print(responses_gold[:3], responses_y[:3], responses_z[:3])

    categorical_params_null = gen_dirichlet_samples(alpha=alpha, n=n_items)
    noise_params_null = gen_dirichlet_samples(alpha=noise_parameters, n=n_items)

    distorted_params_null = distort_parameters_cat(categorical_params_null, noise_params_null, distortion)

    responses_gold_null, responses_y_null, responses_z_null = gen_null_responses_cat(categorical_params_null, distorted_params_null, k_responses)
    
    responses_null.append(
        datatypes.ResponseData(
            gold=responses_gold_null, preds1=responses_y_null, preds2=responses_z_null
        )
    )
    # print(responses_gold_null.shape, responses_y_null.shape, responses_z_null.shape)
    # print(responses_gold_null[:3], responses_y_null[:3], responses_z_null[:3])

    # break
  # print(len(responses_alt))
  # print(len(responses_null))

  response_sets = datatypes.ResponseSets(
      alt_data_list=responses_alt, null_data_list=responses_null
  )

  return response_sets

if __name__ == "__main__":
  simulate_response_tables_cat(n_items=3,k_responses=5,m_categories=3)
