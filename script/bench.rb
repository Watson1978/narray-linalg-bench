#!/usr/bin/env ruby
# frozen_string_literal: true

# Times one problem at one size in one dtype and prints a single TSV line:
#
#   impl  backend  problem  size  dtype  ms/step  best_seconds  all_seconds
#
# bench/run.sh reads the sixth column. Fixture loading and process start-up are
# outside the timed region.
#
#   GPU=1 DTYPE=fp32 ruby script/bench.rb --problem gaussian_process --size mid --inner 20

require 'optparse'
require_relative '../lib/linalg_bench'

options = { problem: nil, size: 'small', variant: nil, rounds: 3, warmup: 1, inner: 1 }
OptionParser.new do |parser|
  parser.banner = 'Usage: ruby script/bench.rb --problem NAME [--size NAME] [--rounds N] [--inner N]'
  parser.on('--problem NAME', '題材') { |v| options[:problem] = v }
  parser.on('--size NAME', 'fixture のサイズ (既定 small)') { |v| options[:size] = v }
  parser.on('--variant NAME', '題材の中で測るもの (既定は題材ごとの先頭)') { |v| options[:variant] = v }
  parser.on('--rounds N', Integer, 'best-of-N の N (既定 3)') { |v| options[:rounds] = v }
  parser.on('--warmup N', Integer, 'ウォームアップの区間数 (既定 1)') { |v| options[:warmup] = v }
  parser.on('--inner N', Integer, '1 区間で回すステップ数 (既定 1)') { |v| options[:inner] = v }
end.parse!
abort 'bench.rb: --problem is required' unless options[:problem]

problem = LinalgBench::Problem.fetch(options[:problem]).new
step = problem.bench_step(problem.fixture(options[:size]), options[:variant])
result = LinalgBench::Timer.measure(rounds: options[:rounds], warmup: options[:warmup], inner: options[:inner]) do
  step.call
end

backend = LinalgBench.gpu? ? "cumo #{Cumo::VERSION} #{LinalgBench.cumo_so}" : "numo-linalg-alt #{Numo::Linalg::VERSION}"
name = [problem.name, options[:variant]].compact.join(':')
puts [LinalgBench.backend_name, backend, name, options[:size], LinalgBench.dtype_name,
      format('%.4f', result.per_step * 1000), format('%.6f', result.best),
      result.times.map { |t| format('%.6f', t) }.join(',')].join("\t")
