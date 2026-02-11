import numpy as np
from absl.testing import absltest
import categorical_sample_lib as csl
import datatypes


class GenDirichletSamplesTest(absltest.TestCase):
  """Tests for gen_dirichlet_samples function."""

  def test_correct_shape(self):
    """Test that output has correct shape."""
    alpha = [0.6, 0.1, 0.3]
    n = 100
    samples = csl.gen_dirichlet_samples(alpha, n)
    self.assertEqual(samples.shape, (n, len(alpha)))

  def test_values_normalized(self):
    """Test that samples sum to approximately 1."""
    alpha = [1.0, 1.0, 1.0]
    n = 50
    samples = csl.gen_dirichlet_samples(alpha, n)
    sums = np.sum(samples, axis=1)
    np.testing.assert_array_almost_equal(sums, np.ones(n), decimal=5)

  def test_values_in_valid_range(self):
    """Test that all samples are in [0, 1]."""
    alpha = [0.5, 2.0, 1.5]
    n = 100
    samples = csl.gen_dirichlet_samples(alpha, n)
    self.assertTrue(np.all(samples >= 0))
    self.assertTrue(np.all(samples <= 1))

  def test_single_sample(self):
    """Test with n=1."""
    alpha = [0.5, 0.5]
    samples = csl.gen_dirichlet_samples(alpha, 1)
    self.assertEqual(samples.shape, (1, 2))
    self.assertAlmostEqual(np.sum(samples), 1.0, places=5)

  def test_with_many_categories(self):
    """Test with many categories."""
    alpha = [1.0] * 10
    n = 50
    samples = csl.gen_dirichlet_samples(alpha, n)
    self.assertEqual(samples.shape, (n, 10))


class DistortParametersTest(absltest.TestCase):
  """Tests for distort_parameters_cat function."""

  def test_distortion_zero(self):
    """When distortion=0, result should equal categorical_params."""
    cat_params = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2]])
    noise_params = np.array([[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]])
    distorted = csl.distort_parameters_cat(cat_params, noise_params, 0.0)
    np.testing.assert_array_almost_equal(distorted, cat_params)

  def test_distortion_one(self):
    """When distortion=1, result should equal noise_params."""
    cat_params = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2]])
    noise_params = np.array([[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]])
    distorted = csl.distort_parameters_cat(cat_params, noise_params, 1.0)
    np.testing.assert_array_almost_equal(distorted, noise_params)

  def test_distortion_half(self):
    """When distortion=0.5, result should be average."""
    cat_params = np.array([[0.6, 0.2, 0.2]])
    noise_params = np.array([[0.4, 0.4, 0.2]])
    expected = np.array([[0.5, 0.3, 0.2]])
    distorted = csl.distort_parameters_cat(cat_params, noise_params, 0.5)
    np.testing.assert_array_almost_equal(distorted, expected)

  def test_shape_preservation(self):
    """Test that output shape matches input shape."""
    cat_params = np.ones((10, 5)) / 5
    noise_params = np.ones((10, 5)) / 5
    distorted = csl.distort_parameters_cat(cat_params, noise_params, 0.3)
    self.assertEqual(distorted.shape, cat_params.shape)

  def test_values_in_valid_range(self):
    """Test that distorted values remain non-negative."""
    cat_params = np.random.dirichlet([1] * 3, 100)
    noise_params = np.random.dirichlet([1] * 3, 100)
    distorted = csl.distort_parameters_cat(cat_params, noise_params, 0.3)
    self.assertTrue(np.all(distorted >= -1e-10))


class GenAltResponsesTest(absltest.TestCase):
  """Tests for gen_alt_responses_cat function."""

  def test_shapes(self):
    """Test that output shapes are correct."""
    cat_params = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2]])
    dist_params = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    k = 5
    gold, y, z = csl.gen_alt_responses_cat(cat_params, dist_params, k)
    
    self.assertEqual(gold.shape, (2, k))
    self.assertEqual(y.shape, (2, k))
    self.assertEqual(z.shape, (2, k))

  def test_values_are_integers(self):
    """Test that responses are categorical indices (integers)."""
    cat_params = np.array([[0.6, 0.2, 0.2]])
    dist_params = np.array([[0.5, 0.3, 0.2]])
    gold, y, z = csl.gen_alt_responses_cat(cat_params, dist_params, 10)
    
    self.assertTrue(np.all(np.mod(gold, 1) == 0))
    self.assertTrue(np.all(np.mod(y, 1) == 0))
    self.assertTrue(np.all(np.mod(z, 1) == 0))

  def test_values_in_valid_range(self):
    """Test that values are valid category indices."""
    cat_params = np.array([[0.25, 0.25, 0.25, 0.25]])
    dist_params = np.array([[0.25, 0.25, 0.25, 0.25]])
    gold, y, z = csl.gen_alt_responses_cat(cat_params, dist_params, 100)
    
    self.assertTrue(np.all(gold >= 0) and np.all(gold < 4))
    self.assertTrue(np.all(y >= 0) and np.all(y < 4))
    self.assertTrue(np.all(z >= 0) and np.all(z < 4))

  def test_single_item_single_response(self):
    """Test with minimal parameters."""
    cat_params = np.array([[0.5, 0.5]])
    dist_params = np.array([[0.5, 0.5]])
    gold, y, z = csl.gen_alt_responses_cat(cat_params, dist_params, 1)
    
    self.assertEqual(gold.shape, (1, 1))


class MixArraysTest(absltest.TestCase):
  """Tests for mix_arrays function."""

  def test_shape_preservation(self):
    """Test that output shape matches input shape."""
    arr1 = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])
    arr2 = np.array([[0.3, 0.3, 0.4], [0.4, 0.4, 0.2]])
    mixed = csl.mix_arrays(arr1, arr2)
    self.assertEqual(mixed.shape, arr1.shape)

  def test_values_from_inputs(self):
    """Test that mixed array values come from input arrays."""
    arr1 = np.array([[1.0, 0.0], [2.0, 0.0]])
    arr2 = np.array([[3.0, 0.0], [4.0, 0.0]])
    
    for _ in range(10):
      mixed = csl.mix_arrays(arr1, arr2)
      for i in range(mixed.shape[0]):
        # Each row should be either from arr1 or arr2
        is_from_arr1 = np.allclose(mixed[i], arr1[i])
        is_from_arr2 = np.allclose(mixed[i], arr2[i])
        self.assertTrue(is_from_arr1 or is_from_arr2)

  def test_random_distribution(self):
    """Test that mixing creates a distribution of choices."""
    arr1 = np.ones((1000, 3)) * 1.0
    arr2 = np.ones((1000, 3)) * 2.0
    mixed = csl.mix_arrays(arr1, arr2)
    
    # Count rows that match arr1 or arr2
    count_arr1 = np.sum(np.all(np.isclose(mixed, arr1), axis=1))
    count_arr2 = np.sum(np.all(np.isclose(mixed, arr2), axis=1))
    
    # Both should appear at least once out of 1000
    self.assertGreater(count_arr1, 0)
    self.assertGreater(count_arr2, 0)


class GenNullResponsesTest(absltest.TestCase):
  """Tests for gen_null_responses_cat function."""

  def test_shapes(self):
    """Test that output shapes are correct."""
    cat_params = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2]])
    dist_params = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    k = 5
    gold, y, z = csl.gen_null_responses_cat(cat_params, dist_params, k)
    
    self.assertEqual(gold.shape, (2, k))
    self.assertEqual(y.shape, (2, k))
    self.assertEqual(z.shape, (2, k))

  def test_values_are_integers(self):
    """Test that responses are categorical indices."""
    cat_params = np.array([[0.6, 0.2, 0.2]])
    dist_params = np.array([[0.5, 0.3, 0.2]])
    gold, y, z = csl.gen_null_responses_cat(cat_params, dist_params, 10)
    
    self.assertTrue(np.all(np.mod(gold, 1) == 0))
    self.assertTrue(np.all(np.mod(y, 1) == 0))
    self.assertTrue(np.all(np.mod(z, 1) == 0))

  def test_values_in_valid_range(self):
    """Test that values are valid category indices."""
    cat_params = np.array([[0.25, 0.25, 0.25, 0.25]])
    dist_params = np.array([[0.25, 0.25, 0.25, 0.25]])
    gold, y, z = csl.gen_null_responses_cat(cat_params, dist_params, 100)
    
    self.assertTrue(np.all(gold >= 0) and np.all(gold < 4))
    self.assertTrue(np.all(y >= 0) and np.all(y < 4))
    self.assertTrue(np.all(z >= 0) and np.all(z < 4))


class SimulateResponseTablesTest(absltest.TestCase):
  """Tests for simulate_response_tables_cat function."""

  def test_default_parameters(self):
    """Test with default parameters."""
    response_sets = csl.simulate_response_tables_cat(
        n_items=10, k_responses=3, m_categories=3, num_samples=2
    )
    
    self.assertIsInstance(response_sets, datatypes.ResponseSets)
    self.assertEqual(len(response_sets.alt_data_list), 2)
    self.assertEqual(len(response_sets.null_data_list), 2)

  def test_response_data_structure(self):
    """Test that ResponseData objects have correct structure."""
    response_sets = csl.simulate_response_tables_cat(
        n_items=5, k_responses=3, m_categories=3, num_samples=1
    )
    
    alt_data = response_sets.alt_data_list[0]
    null_data = response_sets.null_data_list[0]
    
    self.assertIsInstance(alt_data, datatypes.ResponseData)
    self.assertIsInstance(null_data, datatypes.ResponseData)
    self.assertEqual(alt_data.gold.shape, (5, 3))
    self.assertEqual(alt_data.preds1.shape, (5, 3))
    self.assertEqual(alt_data.preds2.shape, (5, 3))

  def test_different_num_samples(self):
    """Test with different number of samples."""
    for num_samples in [1, 3, 5]:
      response_sets = csl.simulate_response_tables_cat(
          n_items=5,
          k_responses=2,
          m_categories=2,
          alpha=[0.5, 0.5],
          noise_parameters=[0.5, 0.5],
          num_samples=num_samples
      )
      self.assertEqual(len(response_sets.alt_data_list), num_samples)
      self.assertEqual(len(response_sets.null_data_list), num_samples)

  def test_different_dimensions(self):
    """Test with various dimensions."""
    n_items = 20
    k_responses = 5
    m_categories = 4
    alpha = [0.25, 0.25, 0.25, 0.25]
    noise_params = [0.25, 0.25, 0.25, 0.25]
    
    response_sets = csl.simulate_response_tables_cat(
        n_items=n_items,
        k_responses=k_responses,
        m_categories=m_categories,
        alpha=alpha,
        noise_parameters=noise_params,
        num_samples=1
    )
    
    data = response_sets.alt_data_list[0]
    self.assertEqual(data.gold.shape, (n_items, k_responses))
    self.assertEqual(data.preds1.shape, (n_items, k_responses))
    self.assertEqual(data.preds2.shape, (n_items, k_responses))

  def test_with_custom_alpha(self):
    """Test with custom alpha parameters."""
    alpha = [0.5, 0.3, 0.2]
    response_sets = csl.simulate_response_tables_cat(
        n_items=5, k_responses=3, m_categories=3, alpha=alpha, num_samples=1
    )
    
    self.assertEqual(len(response_sets.alt_data_list), 1)

  def test_with_custom_distortion(self):
    """Test with custom distortion value."""
    for distortion in [0.0, 0.2, 0.5, 1.0]:
      response_sets = csl.simulate_response_tables_cat(
          n_items=5,
          k_responses=3,
          m_categories=3,
          distortion=distortion,
          num_samples=1
      )
      
      self.assertEqual(len(response_sets.alt_data_list), 1)

  def test_with_prior_distribution(self):
    """Test with prior distribution specified."""
    response_sets = csl.simulate_response_tables_cat(
        n_items=5,
        k_responses=3,
        m_categories=3,
        num_samples=1,
        dir_prior_prob_dist="uniform",
        dir_prior_prob_dist_params=[0, 1]
    )
    
    self.assertEqual(len(response_sets.alt_data_list), 1)

  def test_compute_actual_p_values_false(self):
    """Test that without compute_actual_p_values, only response_sets returned."""
    result = csl.simulate_response_tables_cat(
        n_items=5,
        k_responses=3,
        m_categories=3,
        num_samples=1,
        compute_actual_p_values=False
    )
    
    self.assertIsInstance(result, datatypes.ResponseSets)

  def test_compute_actual_p_values_true(self):
    """Test that with compute_actual_p_values, tuple returned."""
    result = csl.simulate_response_tables_cat(
        n_items=5,
        k_responses=3,
        m_categories=3,
        num_samples=1,
        compute_actual_p_values=True
    )
    
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)
    response_sets, actual_response_sets = result
    self.assertIsInstance(response_sets, datatypes.ResponseSets)
    self.assertIsInstance(actual_response_sets, datatypes.ResponseSets)

  def test_noise_parameters_matching(self):
    """Test with matching noise parameters to categories."""
    alpha = [0.5, 0.3, 0.2]
    noise_params = [0.25, 0.25, 0.5]
    response_sets = csl.simulate_response_tables_cat(
        n_items=5,
        k_responses=3,
        m_categories=3,
        alpha=alpha,
        noise_parameters=noise_params,
        num_samples=1
    )
    
    self.assertEqual(len(response_sets.alt_data_list), 1)

  def test_noise_parameters_mismatched(self):
    """Test that mismatched noise parameters get handled."""
    alpha = [0.5, 0.3, 0.2]
    noise_params = [0.5, 0.5]  # Only 2, but m_categories=3
    response_sets = csl.simulate_response_tables_cat(
        n_items=5,
        k_responses=3,
        m_categories=3,
        alpha=alpha,
        noise_parameters=noise_params,
        num_samples=1
    )
    
    # Should still generate valid response sets
    self.assertEqual(len(response_sets.alt_data_list), 1)


class IntegrationTest(absltest.TestCase):
  """Integration tests combining multiple operations."""

  def test_end_to_end_simulation(self):
    """Test complete simulation workflow."""
    n_items = 50
    k_responses = 5
    m_categories = 3
    num_samples = 2
    
    response_sets = csl.simulate_response_tables_cat(
        n_items=n_items,
        k_responses=k_responses,
        m_categories=m_categories,
        num_samples=num_samples
    )
    
    # Verify structure
    self.assertEqual(len(response_sets.alt_data_list), num_samples)
    self.assertEqual(len(response_sets.null_data_list), num_samples)
    
    # Verify each data object
    for alt_data, null_data in zip(
        response_sets.alt_data_list, response_sets.null_data_list
    ):
      self.assertEqual(alt_data.gold.shape, (n_items, k_responses))
      self.assertEqual(null_data.gold.shape, (n_items, k_responses))
      
      # Verify values are valid
      self.assertTrue(np.all(alt_data.gold >= 0))
      self.assertTrue(np.all(alt_data.gold < m_categories))


if __name__ == '__main__':
  absltest.main()
