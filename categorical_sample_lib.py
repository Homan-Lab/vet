import datetime
from typing import Any, List, Optional, Tuple

from absl import logging
import numpy as np
import datatypes
import cat_machine_contest_metrics as cmcm

def gen_dirichlet_samples(
    alpha: List[float],
    num_samples: int,
) -> np.ndarray:
  """Generates n samples from a dirichlet distribution with parameters alpha.

  Args:
      alpha (List[float]): Parameters of the dirichlet distribution.
      num_samples (int): Number of samples to generate.

  Returns:
      np.ndarray: Samples generated from a dirichlet distribution
  """
  rng = np.random.default_rng()

  samples = rng.dirichlet(alpha, num_samples)
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
  distorted_params = (1 - distortion) * \
    categorical_params + distortion * noise_params
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
      Tuple[np.ndarray, np.ndarray, np.ndarray,]: Generated data for gold
        and two machines
  """
  rng = np.random.default_rng()
  responses_gold = np.column_stack(
    [rng.multinomial(1, categorical_params).argmax(axis=-1)
     for _ in range(k_responses)]
  )
  responses_y = np.column_stack(
    [rng.multinomial(1, categorical_params).argmax(axis=-1)
     for _ in range(k_responses)]
  )
  responses_z = np.column_stack(
    [rng.multinomial(1, distorted_params).argmax(axis=-1)
     for _ in range(k_responses)]
  )
  return responses_gold, responses_y, responses_z

def mix_arrays(
    array_1: np.ndarray,
    array_2: np.ndarray,
) -> np.ndarray:
  """Mixes two arrays by randomly choosing elements from each array.

  Args:
      array1 (np.ndarray): array one
      array2 (np.ndarray): array two

  Returns:
      np.ndarray: mixed arrays
  """
  rng = np.random.default_rng()
  random_mask = rng.choice([True, False], size=(array_1.shape[0]))

  mixed_arrays = np.where(random_mask[:, np.newaxis], array_1, array_2)

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
      Tuple[np.ndarray, np.ndarray, np.ndarray,]: Generated data for gold
        and two machines
  """
  rng = np.random.default_rng()

  responses_gold = np.column_stack(
    [rng.multinomial(1, categorical_params_null).argmax(axis=-1)
     for _ in range(k_responses)]
  )
  responses_y = np.column_stack(
    [rng.multinomial(1, mix_arrays(categorical_params_null, distorted_params_null)).argmax(
      axis=-1) for _ in range(k_responses)]
  )
  responses_z = np.column_stack(
    [rng.multinomial(1, mix_arrays(distorted_params_null, categorical_params_null)).argmax(
      axis=-1) for _ in range(k_responses)]
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
    dir_prior_prob_dist: Optional[str] = None,
    dir_prior_prob_dist_params: Optional[List[Any]] = None,
    compute_actual_p_values: bool = False,
) -> datatypes.ResponseSets:
  """Generates a collection of machine responses.

  Generates tables ("sets"), for null and alternative hypotheses

  Args:
      n_items (int, optional): Number of items per set. Defaults to 1000.
      k_responses (int, optional): Number of responses per item. Defaults to 5.
      m_categories (int, optional): Number of categories. Defaults to 3.
      alpha (List[float], optional): Categorical parameters. Defaults to [0.6, 0.1, 0.3].
      noise_parameters (List[float], optional): Noise parameters. Defaults to [0.333, 0.333, 0.334].
      distortion (float, optional): Distortion value. Defaults to 0.1.
      num_samples (int, optional): Number of samples of size n_items x k_responses. Defaults to 1000.

  Returns:
      datatypes.ResponseSets: _description_
  """

  prior_distributions = {
      "beta": np.random.default_rng().beta,
      "exponential": np.random.default_rng().exponential,
      "gamma": np.random.default_rng().gamma,
      "lognormal": np.random.default_rng().lognormal,
      "normal": np.random.default_rng().normal,
      "uniform": np.random.default_rng().uniform,
  }

  responses_alt = []
  responses_null = []

  if len(noise_parameters) != m_categories:
    logging.info(
      f"Length of noise parameters: {len(noise_parameters)} does \
      not match the number of categories: {m_categories}")
    noise_parameters = [1.0/m_categories]*m_categories
    logging.info(f"Using uniform noise parameters: {noise_parameters}")

  if compute_actual_p_values:
    actual_responses_alt = []
    actual_responses_null = []
  
  for _ in range(num_samples):
    if dir_prior_prob_dist is not None:
      dir_prior_dist = prior_distributions.get(dir_prior_prob_dist)
      if dir_prior_dist:
        if dir_prior_prob_dist_params is not None:
          alpha = dir_prior_dist(
            *list(map(float, dir_prior_prob_dist_params)), size=m_categories)
        else:
          alpha = dir_prior_dist(size=m_categories)
        if len(alpha) != m_categories:
          logging.info(
            f"Length of alpha parameters: {len(alpha)} do not match \
              the number of categories: {m_categories}")
          print(
            f"Length of alpha parameters: {len(alpha)} do not match \
              the number of categories: {m_categories}")
          break
      else:
        logging.info("Prior distribution not supported")
        print("Prior distribution not supported")
        break

    categorical_params = gen_dirichlet_samples(
      alpha=alpha, num_samples=n_items)
    noise_params = gen_dirichlet_samples(
      alpha=noise_parameters, num_samples=n_items)

    distorted_params = distort_parameters_cat(
      categorical_params, noise_params, distortion)

    responses_gold, responses_y, responses_z = gen_alt_responses_cat(
      categorical_params, distorted_params, k_responses)

    responses_alt.append(
        datatypes.ResponseData(
            gold=responses_gold, preds1=responses_y, preds2=responses_z
        )
    )

    if compute_actual_p_values:
      actual_responses_alt.append(
        datatypes.ResponseData(
            gold=categorical_params,
            preds1=categorical_params,
            preds2=distorted_params
        )
      )

    categorical_params_null = gen_dirichlet_samples(
      alpha=alpha, num_samples=n_items)
    noise_params_null = gen_dirichlet_samples(
      alpha=noise_parameters, num_samples=n_items)

    distorted_params_null = distort_parameters_cat(
      categorical_params_null, noise_params_null, distortion)

    responses_gold_null, responses_y_null, responses_z_null = \
      gen_null_responses_cat(
        categorical_params_null, distorted_params_null, k_responses)
    
    responses_null.append(
        datatypes.ResponseData(
            gold=responses_gold_null,
            preds1=responses_y_null,
            preds2=responses_z_null
        )
    )

    if compute_actual_p_values:
      actual_responses_null.append(
          datatypes.ResponseData(
              gold=categorical_params_null,
              preds1=categorical_params_null,
              preds2=distorted_params_null
          )
      )

  response_sets = datatypes.ResponseSets(
      alt_data_list=responses_alt, null_data_list=responses_null
  )

  if compute_actual_p_values:
    actual_response_sets = datatypes.ResponseSets(
        alt_data_list=actual_responses_alt,
        null_data_list=actual_responses_null
    )
    return response_sets, actual_response_sets

  return response_sets

if __name__ == "__main__":
  start_time = datetime.datetime.now()
  response_sets = simulate_response_tables_cat(
    n_items=3,
    k_responses=5,
    m_categories=3,
    dir_prior_prob_dist="uniform",
    dir_prior_prob_dist_params=[0,1])
  
  logging.info(len(response_sets.alt_data_list))
  print(len(response_sets.alt_data_list))
  
  elapsed_time = datetime.datetime.now() - start_time
  logging.info("Data generation time=", elapsed_time.total_seconds())
  print(f"Data generation time = {elapsed_time.total_seconds()}")

  start_time = datetime.datetime.now()
  print(cmcm.cat_accuracy(
    response_sets.alt_data_list[0].gold,
    response_sets.alt_data_list[0].preds1,
    response_sets.alt_data_list[0].preds2))
  print(
    f"Elapsed time = {(datetime.datetime.now() - start_time).total_seconds()}")
