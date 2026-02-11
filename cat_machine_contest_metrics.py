
import numpy as np
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
import sklearn.metrics

def binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
  return np.where(scores < threshold, 0, 1)

def freq_agg(arr: np.ndarray, num_categories: int = 3) -> np.ndarray:
  return np.apply_along_axis(lambda x: np.bincount(x, minlength=num_categories), axis=1, arr=arr)

def majority_vote(arr: np.ndarray, num_categories: int = 3) -> np.ndarray:
  counts = freq_agg(arr=arr, num_categories=num_categories)
  return np.argmax(counts, axis=1)

def cat_accuracy(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute accuracy relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A pair of accuracy scores, for machines 1 and 2, relative to
    human scores.
  """

  num_categories = np.max(np.concatenate((human.flatten(), machine1.flatten(), machine2.flatten()))) + 1
  human = majority_vote(human, num_categories)
  machine1 = majority_vote(machine1, num_categories)
  machine2 = majority_vote(machine2, num_categories)

  return (
      sklearn.metrics.accuracy_score(human, machine1),
      sklearn.metrics.accuracy_score(human, machine2),
  )

def cat_actual_accuracy(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute accuracy relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A pair of accuracy scores, for machines 1 and 2, relative to
    human scores.
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

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

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

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A 2D array of human responses.
    machine1: A 2D array of machine responses.
    machine2: A 2D array of responses from another machine.

  Returns:
    A 2-tuple of the f-score, i.e., the harmonic mean of precision and recall,
    between one machine and the human responses, and of the other machine at the
    human responses.
  """

  num_categories = np.max(np.concatenate((human.flatten(), machine1.flatten(), machine2.flatten()))) + 1
  human = majority_vote(human, num_categories)
  machine1 = majority_vote(machine1, num_categories)
  machine2 = majority_vote(machine2, num_categories)

  return (
      sklearn.metrics.f1_score(human, machine1, average='micro'),
      sklearn.metrics.f1_score(human, machine2, average='micro'),
  )

def cat_precision(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute precision relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A pair of precision scores, for machines 1 and 2, relative to
    human scores.
  """

  num_categories = np.max(np.concatenate((human.flatten(), machine1.flatten(), machine2.flatten()))) + 1
  human = majority_vote(human, num_categories)
  machine1 = majority_vote(machine1, num_categories)
  machine2 = majority_vote(machine2, num_categories)

  return (
      sklearn.metrics.precision_score(human, machine1),
      sklearn.metrics.precision_score(human, machine2),
  )

def cat_recall(
    human: np.ndarray,
    machine1: np.ndarray,
    machine2: np.ndarray,
) -> tuple[float, float]:
  """Compute recall relative to human labels.

  The params ht, mt1 and mt2 can be specified via a config string,
  so we use short names for them.

  Args:
    human: A list of human scores.
    machine1: A list of machine scores.
    machine2: Another list of machine scores.

  Returns:
    A pair of recall scores, for machines 1 and 2, relative to
    human scores.
  """
  
  num_categories = np.max(np.concatenate((human.flatten(), machine1.flatten(), machine2.flatten()))) + 1
  human = majority_vote(human, num_categories)
  machine1 = majority_vote(machine1, num_categories)
  machine2 = majority_vote(machine2, num_categories)

  return (
      sklearn.metrics.recall_score(human, machine1),
      sklearn.metrics.recall_score(human, machine2),
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
  num_categories = np.max(np.concatenate((human.flatten(), machine1.flatten(), machine2.flatten()))) + 1
  human = freq_agg(human, num_categories)
  machine1 = freq_agg(machine1, num_categories)
  machine2 = freq_agg(machine2, num_categories)

  human = human/np.sum(human, axis=-1, keepdims=True)
  machine1 = machine1/np.sum(machine1, axis=-1, keepdims=True)
  machine2 = machine2/np.sum(machine2, axis=-1, keepdims=True)

  return (np.mean(abs(human - machine1)), np.mean(abs(human - machine2)))

def cat_actual_mean_absolute_error(
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

  human = human/np.sum(human, axis=-1, keepdims=True)
  machine1 = machine1/np.sum(machine1, axis=-1, keepdims=True)
  machine2 = machine2/np.sum(machine2, axis=-1, keepdims=True)

  return (np.mean(abs(human - machine1)), np.mean(abs(human - machine2)))

def cat_wins_mae(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute number of wins relative to distance from human labels.

  Args:
    human: A list of human responses.
    machine1: A list of machine responses.
    machine2: Another list of machine responses.

  Returns:
    A 2-tuple of the itemwise distance wins between one machine and the human
    responses, and of the other machine and the human responses.
  """

  num_categories = np.max(np.concatenate((human.flatten(), machine1.flatten(), machine2.flatten()))) + 1
  human = freq_agg(human, num_categories)
  machine1 = freq_agg(machine1, num_categories)
  machine2 = freq_agg(machine2, num_categories)

  machine1_results = np.mean(abs(human - machine1), axis=-1)
  machine2_results = np.mean(abs(human - machine2), axis=-1)

  return (
      np.sum(machine1_results < machine2_results),
      np.sum(machine1_results > machine2_results),
  )

def cat_actual_wins_mae(
    human: np.ndarray, machine1: np.ndarray, machine2: np.ndarray
) -> tuple[float, float]:
  """Compute number of wins relative to distance from human labels.

  Args:
    human: A list of human responses.
    machine1: A list of machine responses.
    machine2: Another list of machine responses.

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
  num_categories = np.max(np.concatenate((human.flatten(), machine1.flatten(), machine2.flatten()))) + 1
  human = freq_agg(human, num_categories) + 1e-12
  machine1 = freq_agg(machine1, num_categories) + 1e-12
  machine2 = freq_agg(machine2, num_categories) + 1e-12

  return (
    np.mean(st.entropy(human, machine1, axis=-1, nan_policy='omit')),
    np.mean(st.entropy(human, machine2, axis=-1, nan_policy='omit')),
  )

def cat_actual_kl_div(
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
    A 2-tuple of the itemwise Jensen-Shannon distance between one machine and the human
    responses, and of the other machine and the human responses.
  """
  num_categories = np.max(np.concatenate((human.flatten(), machine1.flatten(), machine2.flatten()))) + 1
  human = freq_agg(human, num_categories) + 1e-12
  machine1 = freq_agg(machine1, num_categories) + 1e-12
  machine2 = freq_agg(machine2, num_categories) + 1e-12

  return (
    np.mean(jensenshannon(human, machine1, axis=-1)),
    np.mean(jensenshannon(human, machine2, axis=-1)),
  )
