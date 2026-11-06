# frozen_string_literal: true

require 'bundler/setup'
require 'json'

module LinalgBench
  class Error < StandardError; end

  ROOT = File.expand_path('..', __dir__)
  DATA_DIR = ENV.fetch('LINALG_BENCH_DATA') { File.join(ROOT, 'data') }
end

require_relative 'linalg_bench/backend'
require_relative 'linalg_bench/fixtures'
require_relative 'linalg_bench/gates'
require_relative 'linalg_bench/known_failures'
require_relative 'linalg_bench/timer'
require_relative 'linalg_bench/problem'
Dir[File.join(__dir__, 'linalg_bench/problems/*.rb')].each { |path| require path }
