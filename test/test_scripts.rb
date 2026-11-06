# frozen_string_literal: true

require_relative 'test_helper'

class TestScripts < Test::Unit::TestCase
  include TestHelper

  def test_every_problem_passes_its_gates_on_numo_and_numpy_with_the_same_values
    Dir.mktmpdir do |dir|
      ruby_json = File.join(dir, 'numo.json')
      python_json = File.join(dir, 'numpy.json')
      _, err, status = run_ruby('script/gate.rb', '--out', ruby_json)
      assert(status.success?, err)
      run_python('python/gate.py', '--out', python_json)
      ruby = JSON.parse(File.read(ruby_json))
      python = JSON.parse(File.read(python_json))
      assert_equal(%w[ruby numo fp64 small], ruby.values_at('impl', 'backend', 'dtype', 'size'))
      assert_equal(%w[python numpy fp64 small], python.values_at('impl', 'backend', 'dtype', 'size'))
      assert_equal(LinalgBench::Problem.registry.keys.sort, ruby['problems'].keys.sort)
      assert_equal(ruby['problems'].keys.sort, python['problems'].keys.sort)
      [ruby, python].each do |report|
        report['problems'].each do |name, entry|
          assert_include(%w[pass known_failure], entry['status'], "#{report['backend']} #{name}: #{entry}")
        end
      end
      out, err, status = run_ruby('script/compare_gates.rb', ruby_json, python_json)
      assert(status.success?, "#{out}#{err}")
    end
  end

  def test_aggregate_reads_what_run_sh_writes
    omit("#{PYTHON} not found") unless File.executable?(PYTHON)
    Dir.mktmpdir do |dir|
      File.write(File.join(dir, 'conditions.tsv'), "# header\nprobe\truby x\tpython y\n")
      rows = (0..3).flat_map do |round|
        [[round, 'ruby', 10.0], [round, 'python', 20.0 + round], [round, 'control', 10.0 + ((-1)**round)]]
      end
      File.write(File.join(dir, 'probe.tsv'), rows.map { |r| "#{r.join("\t")}\n" }.join)
      out, err, status = Open3.capture3('timeout', '60', PYTHON, 'bench/aggregate.py',
                                        File.join(dir, 'conditions.tsv'), dir, chdir: ROOT)
      assert(status.success?, err)
      assert_equal('| probe | 10.000 | 22.000 | 2.200 3/3 p=0.25 | 0.900 1/3 p=1 またぐ | 0.0% | 9.5% 要確認 |', out.lines.last.chomp)
    end
  end
end
