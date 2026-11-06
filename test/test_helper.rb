# frozen_string_literal: true

ENV['DTYPE'] ||= 'fp64'
$LOAD_PATH.unshift(File.expand_path('../lib', __dir__))

require 'open3'
require 'tmpdir'
require 'test/unit'
require 'linalg_bench'

module TestHelper
  ROOT = File.expand_path('..', __dir__)
  PYTHON = ENV.fetch('PYTHON') { File.join(ROOT, 'python/.venv/bin/python') }
  PROBLEMS = %w[hartree_fock gaussian_process kalman_riccati procrustes_pca vandermonde_lstsq
                markov_diffusion kkt].freeze

  def require_fixture(problem, size)
    omit("#{problem}_#{size} not found; run `rake fixtures`") unless LinalgBench::Fixtures.exist?(problem, size)
    LinalgBench::Fixtures.read(problem, size)
  end

  # GPU stays off for the child: the Python side is checked on numpy here.
  def run_python(*args)
    omit("#{PYTHON} not found; see README") unless File.executable?(PYTHON)
    out, err, status = Open3.capture3({ 'GPU' => nil }, 'timeout', '300', PYTHON, *args, chdir: ROOT)
    assert(status.success?, "python #{args.join(' ')} failed:\n#{err}")
    out
  end

  def run_ruby(*args)
    out, err, status = Open3.capture3({ 'GPU' => nil, 'DTYPE' => nil }, 'timeout', '300', RbConfig.ruby, *args, chdir: ROOT)
    [out, err, status]
  end
end
