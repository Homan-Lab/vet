from typing import Any

import numpy as np
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
import sklearn.metrics

def binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
  """Convert an array of scores into float scores in [0, 1] based on a
    threshold.

  Args:
    scores (np.ndarray): An array of scores to be binarized.
    threshold (float): The threshold value for binarization.
      Scores below this value will be set to 0,
      and scores equal to or above this value will be set to 1.

  Returns:
    np.ndarray: The binarized array.
  """
  return np.where(scores < threshold, 0, 1)

def freq_agg(arr: np.ndarray, num_categories: int = 0
) -> np.ndarray[Any, np.dtype[np.int_]]:
  """Convert a 2D-array of category labels into a 2D-array of category frequencies.

  Args:
    arr (np.ndarray): A 2D array of integer category labels.
    num_categories (int, optional): The number of categories. Defaults to 0.

  Returns:
    np.ndarray: An array of category frequencies for each row in `arr`.
  """
  return np.apply_along_axis(
    lambda x: np.bincount(x, minlength=num_categories),
    axis=1,
    arr=arr)

def majority_vote(arr: np.ndarray, num_categories: int = 0) -> np.ndarray:
  """Convert matrix of integer categories into array of row-wise pluralities
    (plurality along the column).

  Args:
    arr (np.ndarray): A matrix of integer category labels.
    num_categories (int, optional): The number of categories. Defaults to 0.
  
  Returns:
    np.ndarray: A matrix of plurality category labels.
  """
  counts = freq_agg(arr=arr, num_categories=num_categories)
  return np.argmax(counts, axis=1)

def get_num_categories(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> int:
  """Get the number of categories from human, machine1, and machine2
    matrices of integer category labels.

  Args:
    human (np.ndarray): A matrix of integer category labels for human.
    machine1 (np.ndarray): A matrix of integer category labels for machine 1.
    machine2 (np.ndarray): A matrix of integer category labels for machine 2.
  Returns:
    int: The number of categories, which is one more than the maximum category
      label across all three matrices.
  """
  return np.max(np.stack((human, machine1, machine2))) + 1

def cat_accuracy(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute accuracy relative to human labels.

  Args:
    human: A matrix of human scores.
    machine1: A matrix of machine scores.
    machine2: Another matrix of machine scores.

  Returns:
    A pair of accuracy scores, for machines 1 and 2, relative to
    human scores.
  """

  num_categories = get_num_categories(human, machine1, machine2)
  human_majorities = majority_vote(human, num_categories)
  machine1_majorities = majority_vote(machine1, num_categories)
  machine2_majorities = majority_vote(machine2, num_categories)

  return (
      sklearn.metrics.accuracy_score(human_majorities, machine1_majorities),
      sklearn.metrics.accuracy_score(human_majorities, machine2_majorities),
  )

def cat_actual_accuracy(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute accuracy relative to human labels based on
    categorical parameters directly.

  Args:
    human: A matrix of categorical parameters for human.
    machine1: A matrix of categorical parameters for machine 1.
    machine2: Another matrix of categorical parameters for machine 2.

  Returns:
    A pair of accuracy scores, for machines 1 and 2, relative to
    human.
  """

  human = np.argmax(human, axis=-1)
  machine1 = np.argmax(machine1, axis=-1)
  machine2 = np.argmax(machine2, axis=-1)

  return (
      sklearn.metrics.accuracy_score(human, machine1),
      sklearn.metrics.accuracy_score(human, machine2),
  )

def cat_auc(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute ROC AUC relative to human labels.

  Args:
    human: A 2D array of human scores.
    machine1: A 2D array of machine scores.
    machine2: Another 2D array of machine scores.

  Returns:
    A pair of receiver operater characteristic (ROC) area under the curve (AUC)
    scores, for machines 1 and 2, relative to human scores.
  """

  return (
      sklearn.metrics.roc_auc_score(human, machine1),
      sklearn.metrics.roc_auc_score(human, machine2),
  )

def cat_f1_score(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute f1-score relative to human labels.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the f-score, i.e., the harmonic mean of precision and recall,
    between one machine and the human responses, and of the other machine at the
    human responses.
  """

  num_categories = get_num_categories(human, machine1, machine2)
  human_majorities = majority_vote(human, num_categories)
  machine1_majorities = majority_vote(machine1, num_categories)
  machine2_majorities = majority_vote(machine2, num_categories)

  return (
      sklearn.metrics.f1_score(human_majorities, machine1_majorities, average='micro'),
      sklearn.metrics.f1_score(human_majorities, machine2_majorities, average='micro'),
  )

def cat_precision(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute precision relative to human labels.

  Args:
    human: A 2D array of human scores.
    machine1: A 2D array of machine scores.
    machine2: Another 2D array of machine scores.

  Returns:
    A pair of precision scores, for machines 1 and 2, relative to
    human scores.
  """

  num_categories = get_num_categories(human, machine1, machine2)
  human_majorities = majority_vote(human, num_categories)
  machine1_majorities = majority_vote(machine1, num_categories)
  machine2_majorities = majority_vote(machine2, num_categories)

  return (
      sklearn.metrics.precision_score(human_majorities, machine1_majorities),
      sklearn.metrics.precision_score(human_majorities, machine2_majorities),
  )

def cat_recall(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute recall relative to human labels.

  Args:
    human: A 2D array of human scores.
    machine1: A 2D array of machine scores.
    machine2: Another 2D array of machine scores.

  Returns:
    A pair of recall scores, for machines 1 and 2, relative to
    human scores.
  """
  
  num_categories = get_num_categories(human, machine1, machine2)
  human_majorities = majority_vote(human, num_categories)
  machine1_majorities = majority_vote(machine1, num_categories)
  machine2_majorities = majority_vote(machine2, num_categories)

  return (
      sklearn.metrics.recall_score(human_majorities, machine1_majorities),
      sklearn.metrics.recall_score(human_majorities, machine2_majorities),
  )

def cat_mean_absolute_error(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute (L1) itemwise distance mean.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the itemwise distance mean between one machine and the human
    responses, and of the other machine and the human responses.
  """
  num_categories = get_num_categories(human, machine1, machine2)
  human_frequencies = freq_agg(human, num_categories)
  machine1_frequencies = freq_agg(machine1, num_categories)
  machine2_frequencies = freq_agg(machine2, num_categories)

  norm_human_frequencies = human_frequencies/np.sum(
    human_frequencies, axis=-1, keepdims=True)
  norm_machine1_frequencies = machine1_frequencies/np.sum(
    machine1_frequencies, axis=-1, keepdims=True)
  norm_machine2_frequencies = machine2_frequencies/np.sum(
    machine2_frequencies, axis=-1, keepdims=True)

  return (np.mean(abs(norm_human_frequencies - norm_machine1_frequencies)),
          np.mean(abs(norm_human_frequencies - norm_machine2_frequencies)))

def cat_actual_mean_absolute_error(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute (L1) itemwise distance mean based on
    categorical parameters directly.

  Args:
    human: A 2D array of categorical parameters for human.
    machine1: A 2D array of categorical parameters for machine.
    machine2: A 2D array of categorical parameters for another machine.

  Returns:
    A 2-tuple of the itemwise distance mean between one machine and the human
    responses, and of the other machine and the human responses.
  """

  norm_human = human/np.sum(human, axis=-1, keepdims=True)
  norm_machine1 = machine1/np.sum(machine1, axis=-1, keepdims=True)
  norm_machine2 = machine2/np.sum(machine2, axis=-1, keepdims=True)

  return (np.mean(abs(norm_human - norm_machine1)),
          np.mean(abs(norm_human - norm_machine2)))

def cat_wins_mae(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute number of wins relative to distance from human labels.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: Another 2D array of machine responses.

  Returns:
    A 2-tuple of the itemwise distance wins between one machine and the human
    responses, and of the other machine and the human responses.
  """

  num_categories = get_num_categories(human, machine1, machine2)
  human_frequencies = freq_agg(human, num_categories)
  machine1_frequencies = freq_agg(machine1, num_categories)
  machine2_frequencies = freq_agg(machine2, num_categories)

  machine1_results = np.mean(abs(human_frequencies - machine1_frequencies),
                             axis=-1)
  machine2_results = np.mean(abs(human_frequencies - machine2_frequencies),
                             axis=-1)

  return (
      np.sum(machine1_results < machine2_results),
      np.sum(machine1_results > machine2_results),
  )

def cat_actual_wins_mae(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute number of wins relative to distance from human labels
    based on categorical parameters directly.

  Args:
    human: A 2D array of categorical parameters for human.
    machine1: A 2D array of categorical parameters for machine.
    machine2: Another 2D array of categorical parameters for another machine.

  Returns:
    A 2-tuple of the itemwise distance wins between one machine and the human
    responses, and of the other machine and the human responses.
  """

  machine1_results = np.mean(abs(human - machine1), axis=-1)
  machine2_results = np.mean(abs(human - machine2), axis=-1)

  return (
      np.sum(machine1_results < machine2_results),
      np.sum(machine1_results > machine2_results),
  )

def cat_kl_div(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute KL divergence.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the itemwise KL divergence between one machine and the human
    responses, and of the other machine and the human responses.
  """
  num_categories = get_num_categories(human, machine1, machine2)
  human_frequencies = freq_agg(human, num_categories) + 1e-12
  machine1_frequencies = freq_agg(machine1, num_categories) + 1e-12
  machine2_frequencies = freq_agg(machine2, num_categories) + 1e-12

  return (
    np.mean(st.entropy(
      human_frequencies, machine1_frequencies, axis=-1, nan_policy='omit')),
    np.mean(st.entropy(
      human_frequencies, machine2_frequencies, axis=-1, nan_policy='omit')),
  )

def cat_actual_kl_div(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute KL divergence based on categorical parameters directly.

  Args:
    human: A 2D array of categorical parameters for human.
    machine1: A 2D array of categorical parameters for machine.
    machine2: A 2D array of categorical parameters for another machine.

  Returns:
    A 2-tuple of the itemwise KL divergence between one machine and the human
    responses, and of the other machine and the human responses.
  """
  
  human = human + 1e-12
  machine1 = machine1 + 1e-12
  machine2 = machine2 + 1e-12

  return (
    np.mean(st.entropy(human, machine1, axis=-1, nan_policy='omit')),
    np.mean(st.entropy(human, machine2, axis=-1, nan_policy='omit')),
  )

def cat_jsd(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute Jensen-Shannon distance.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the itemwise Jensen-Shannon distance between one machine and
    the human responses, and of the other machine and the human responses.
  """
  num_categories = get_num_categories(human, machine1, machine2)
  human_frequencies = freq_agg(human, num_categories) + 1e-12
  machine1_frequencies = freq_agg(machine1, num_categories) + 1e-12
  machine2_frequencies = freq_agg(machine2, num_categories) + 1e-12

  return (
    np.mean(jensenshannon(human_frequencies, machine1_frequencies, axis=-1)),
    np.mean(jensenshannon(human_frequencies, machine2_frequencies, axis=-1)),
  )
