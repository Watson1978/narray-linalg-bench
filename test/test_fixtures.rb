# frozen_string_literal: true

require_relative 'test_helper'
require 'digest'

class TestFixtures < Test::Unit::TestCase
  include TestHelper

  def test_round_trip_keeps_bytes_shapes_and_attrs
    Dir.mktmpdir do |dir|
      a = HM::DFloat.new(3, 4).seq / 7
      b = HM::DFloat.new(2, 2, 2).seq - 3.5
      LinalgBench::Fixtures.write('probe', 'small', { 'a' => a, 'b' => b }, attrs: { 'x' => 0.1, 'n' => 3 }, dir: dir)
      fixture = LinalgBench::Fixtures.read('probe', 'small', dir: dir)
      assert_equal(a.to_binary, fixture['a'].to_binary)
      assert_equal([2, 2, 2], fixture['b'].shape)
      assert_equal(b.to_binary, fixture['b'].to_binary)
      assert_equal(0.1, fixture.attr('x'))
      assert_equal(3, fixture.attr('n'))
    end
  end

  def test_read_rejects_a_truncated_file
    Dir.mktmpdir do |dir|
      LinalgBench::Fixtures.write('probe', 'small', { 'a' => HM::DFloat.new(8).seq }, dir: dir)
      path = File.join(dir, 'probe_small.bin')
      File.binwrite(path, File.binread(path, 40))
      assert_raise(LinalgBench::Error) { LinalgBench::Fixtures.read('probe', 'small', dir: dir) }
    end
  end

  def test_device_casts_to_the_working_dtype
    fixture = require_fixture('kkt', 'small')
    a = fixture.device('A')
    assert_kind_of(XF, a)
    assert_equal(fixture['A'].shape, a.shape)
    assert_equal(0.0, LinalgBench::Gates.max_abs(LinalgBench.device(LinalgBench.host(a), XM::DFloat) - fixture.device('A', XM::DFloat)))
  end

  PROBLEMS.each do |problem|
    define_method("test_#{problem}_small_matches_its_manifest") do
      fixture = require_fixture(problem, 'small')
      fixture.manifest['arrays'].each do |entry|
        array = fixture[entry['name']]
        assert_equal(entry['shape'], array.shape, entry['name'])
        assert_equal(entry['sha256'], Digest::SHA256.hexdigest(array.to_binary), entry['name'])
      end
    end

    define_method("test_#{problem}_small_reads_the_same_in_python") do
      fixture = require_fixture(problem, 'small')
      python = JSON.parse(run_python('python/fixtures.py', 'digest', problem, 'small'))
      ruby = fixture.arrays.transform_values do |array|
        { 'shape' => array.shape, 'sha256' => Digest::SHA256.hexdigest(array.to_binary) }
      end
      assert_equal(ruby, python)
    end
  end
end
