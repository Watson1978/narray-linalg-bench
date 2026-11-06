# frozen_string_literal: true

require_relative 'test_helper'

class TestKnownFailures < Test::Unit::TestCase
  include TestHelper

  KF = LinalgBench::KnownFailures
  Gates = LinalgBench::Gates

  def setup
    @saved = KF.instance_variable_get(:@table)
    KF.instance_variable_set(:@table, { 'probe' => { 'flaky' => { 'backends' => ['numo'] } } })
  end

  def teardown
    KF.instance_variable_set(:@table, @saved)
  end

  def test_only_recorded_failures_make_a_known_failure
    gates = [Gates.le('flaky', 1.0, 0.1), Gates.le('solid', 0.0, 0.1)]
    assert_equal(['known_failure', { 'known_failures' => ['flaky'] }], KF.classify('probe', gates, 'numo'))
  end

  def test_an_unrecorded_failure_is_a_failure
    gates = [Gates.le('flaky', 1.0, 0.1), Gates.le('solid', 1.0, 0.1)]
    assert_equal('fail', KF.classify('probe', gates, 'numo').first)
  end

  def test_a_record_is_per_backend
    gates = [Gates.le('flaky', 1.0, 0.1)]
    assert_equal('fail', KF.classify('probe', gates, 'cumo').first)
  end

  def test_a_recorded_gate_that_passes_is_a_failure
    gates = [Gates.le('flaky', 0.0, 0.1)]
    assert_equal(['fail', { 'unexpected_passes' => ['flaky'] }], KF.classify('probe', gates, 'numo'))
  end

  def test_the_python_side_reads_the_same_records
    out = run_python('-c', <<~PY)
      import json, sys
      sys.path.insert(0, "python")
      import known_failures
      print(json.dumps(known_failures.table()))
    PY
    assert_equal(JSON.parse(File.read(KF::PATH)), JSON.parse(out))
  end
end
