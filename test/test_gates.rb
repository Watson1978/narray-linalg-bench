# frozen_string_literal: true

require_relative 'test_helper'

class TestGates < Test::Unit::TestCase
  include TestHelper

  Gates = LinalgBench::Gates

  def test_solve_residual_is_zero_for_an_exact_solution
    a = XF[[4, 1], [1, 3]]
    x = XF[1, 2]
    assert_equal(0.0, Gates.solve_residual(a, x, a.dot(x)))
    assert_operator(Gates.solve_residual(a, x + 1e-3, a.dot(x)), :>, 1e-5)
  end

  def test_orthogonality_of_a_rotation_and_of_a_scaled_matrix
    c = Math.cos(0.3)
    s = Math.sin(0.3)
    assert_in_delta(0.0, Gates.orthogonality(XF[[c, -s], [s, c]]), 1e-15)
    assert_in_delta(3.0, Gates.orthogonality(XF[[2, 0], [0, 1]]), 1e-15)
  end

  def test_sorted_comparison_ignores_order
    assert_equal(0.0, Gates.max_rel_diff_sorted(XF[1, 3, 2], XF[3, 2, 1]))
    assert_in_delta(0.1 / 3, Gates.max_rel_diff_sorted(XF[1.1, 3, 2], XF[3, 2, 1]), 1e-12)
  end

  def test_gate_pass_and_report
    gates = [Gates.le('small', 1e-12, 1e-10), Gates.le('large', 1e-8, 1e-10), Gates.eq('rank', 6, 6)]
    assert_equal([true, false, true], gates.map(&:pass?))
    assert_equal({ 'value' => 1e-8, 'tol' => 1e-10, 'op' => 'le', 'pass' => false }, Gates.report(gates)['large'])
  end

  def test_default_tol_follows_the_dtype
    assert_equal('fp64', LinalgBench.dtype_name)
    assert_equal(1e-10, Gates.tol)
    assert_equal(1e-4, Gates.tol('fp32'))
  end

  def test_timer_returns_rounds_and_per_step
    calls = 0
    result = LinalgBench::Timer.measure(rounds: 3, warmup: 2, inner: 4) { calls += 1 }
    assert_equal(3, result.times.size)
    assert_equal(20, calls)
    assert_in_delta(result.best / 4, result.per_step, 1e-15)
  end
end
