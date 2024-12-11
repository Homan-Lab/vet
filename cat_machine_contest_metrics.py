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

Metrics used for machine vs. machine tests.

Contains metrics that compare the performance of one machine to another based on
their responses relative to a human-labeled response set. They were created
to support a simulator used to model response variance in machine learning
testing, but may be useful other settings where two sets of responses are
compared to a third.
"""

import math

import numpy as np
import scipy.spatial
import scipy.stats
import sklearn.metrics

def binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
  return np.where(scores < threshold, 0, 1)

def freq_agg(arr: np.ndarray, num_categories: int = 3):
  return np.array([np.sum(arr == x) for x in range(num_categories)])

def majority_vote(arr: np.ndarray, num_categories: int = 3):
  return np.argmax(np.array([np.sum(arr == x) for x in range(num_categories)]))

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

  # human = np.argmax(freq_agg(human))
  # machine1 = np.argmax(freq_agg(machine1))
  # machine2 = np.argmax(freq_agg(machine2))

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

  # human = np.argmax(freq_agg(human))
  # machine1 = np.argmax(freq_agg(machine1))
  # machine2 = np.argmax(freq_agg(machine2))

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
  # human = freq_agg(human)
  # machine1 = freq_agg(machine1)
  # machine2 = freq_agg(machine2)

  return (np.mean(abs(human - machine1)), np.mean(abs(human - machine2)))
