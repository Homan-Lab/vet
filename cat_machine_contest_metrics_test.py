import numpy as np
from absl.testing import absltest
import cat_machine_contest_metrics


class CatMachineContestMetricsTest(absltest.TestCase):

  def test_binarize(self):
    scores = np.array([0.1, 0.5, 0.8])
    out = cat_machine_contest_metrics.binarize(scores, 0.5)
    expected = np.array([0, 1, 1])
    np.testing.assert_array_equal(out, expected)

  def test_freq_and_majority(self):
    arr = np.array([[0, 1, 1], [2, 2, 0]])
    freq = cat_machine_contest_metrics.freq_agg(arr, num_categories=3)
    np.testing.assert_array_equal(freq, np.array([[1, 2, 0], [1, 0, 2]]))
    maj = cat_machine_contest_metrics.majority_vote(arr, num_categories=3)
    np.testing.assert_array_equal(maj, np.array([1, 2]))

  def test_cat_accuracy_and_f1_precision_recall(self):
    # human and machines as vote matrices -> majority voting
    human = np.array([[0, 0, 1], [1, 1, 1]])
    machine1 = np.array([[0, 1, 0], [1, 1, 1]])
    machine2 = np.array([[1, 1, 1], [0, 0, 0]])

    acc1, acc2 = cat_machine_contest_metrics.cat_accuracy(
      human, machine1, machine2)
    self.assertAlmostEqual(acc1, 1.0)
    self.assertLess(acc2, acc1)

    f1_1, f1_2 = cat_machine_contest_metrics.cat_f1_score(
      human, machine1, machine2)
    self.assertAlmostEqual(f1_1, 1.0)

    # actual (one-hot) accuracy
    human_act = np.array([[1, 0], [0, 1]])
    m1_act = np.array([[1, 0], [0, 1]])
    m2_act = np.array([[0, 1], [1, 0]])
    a1, a2 = cat_machine_contest_metrics.cat_actual_accuracy(
      human_act, m1_act, m2_act)
    self.assertAlmostEqual(a1, 1.0)
    self.assertAlmostEqual(a2, 0.0)

    # auc expects binary labels and scores
    human_bin = np.array([0, 1, 0, 1])
    m1_scores = np.array([0.1, 0.9, 0.2, 0.8])
    m2_scores = np.array([0.6, 0.4, 0.7, 0.3])
    auc1, auc2 = cat_machine_contest_metrics.cat_auc(
      human_bin, m1_scores, m2_scores)
    self.assertGreater(auc1, auc2)

  def test_mean_absolute_error_and_wins(self):
    # human and machines as vote matrices -> distribution based MAE
    human = np.array([[0, 0, 1], [1, 0, 0]])
    machine1 = np.array([[0, 0, 1], [1, 0, 0]])
    machine2 = np.array([[1, 1, 1], [0, 0, 1]])

    mae1, mae2 = cat_machine_contest_metrics.cat_mean_absolute_error(
      human, machine1, machine2)
    self.assertAlmostEqual(mae1, 0.0)
    self.assertGreater(mae2, 0.0)

    # actual MAE with probability vectors
    human_act = np.array([[1.0, 0.0], [0.5, 0.5]])
    m1_act = np.array([[1.0, 0.0], [0.5, 0.5]])
    m2_act = np.array([[0.0, 1.0], [0.0, 1.0]])
    a_mae1, a_mae2 = cat_machine_contest_metrics.cat_actual_mean_absolute_error(
      human_act, m1_act, m2_act)
    self.assertAlmostEqual(a_mae1, 0.0)
    self.assertGreater(a_mae2, 0.0)

    wins1, wins2 = cat_machine_contest_metrics.cat_wins_mae(
      human, machine1, machine2)
    self.assertGreaterEqual(wins1, 0)
    self.assertGreaterEqual(wins2, 0)
    self.assertGreater(wins1, wins2)

    a_w1, a_w2 = cat_machine_contest_metrics.cat_actual_wins_mae(
      human_act, m1_act, m2_act)
    self.assertGreaterEqual(a_w1, 0)

  def test_kl_and_jsd(self):
    # For cat_kl_div and cat_jsd.
    human_votes = np.array([[0, 0, 0], [1, 1, 1]])
    m1_votes = human_votes.copy()
    m2_votes = np.array([[0, 1, 1], [1, 1, 1]])

    kl1, kl2 = cat_machine_contest_metrics.cat_kl_div(
      human_votes, m1_votes, m2_votes)
    self.assertAlmostEqual(kl1, 0.0, places=6)
    self.assertGreater(kl2, kl1)

    # cat_actual_kl_div operates on probability vectors directly
    human_prob = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    m1_prob = human_prob.copy()
    m2_prob = np.array([[0.5, 0.5, 0.0], [0.2, 0.7, 0.1]])
    akt1, akt2 = cat_machine_contest_metrics.cat_actual_kl_div(
      human_prob, m1_prob, m2_prob)
    self.assertAlmostEqual(akt1, 0.0, places=6)

    jsd1, jsd2 = cat_machine_contest_metrics.cat_jsd(
      human_votes, m1_votes, m2_votes)
    self.assertAlmostEqual(jsd1, 0.0, places=6)
    self.assertGreaterEqual(jsd2, jsd1)


if __name__ == '__main__':
  absltest.main()
