require 'rake/testtask'

ROOT = __dir__
DATA_DIR = ENV.fetch('LINALG_BENCH_DATA') { File.join(ROOT, 'data') }
PYTHON = ENV.fetch('PYTHON') { File.join(ROOT, 'python/.venv/bin/python') }
PROBLEMS = %w[hartree_fock gaussian_process kalman_riccati procrustes_pca vandermonde_lstsq markov_diffusion kkt].freeze

Rake::TestTask.new(:test) do |t|
  t.libs << 'lib' << 'test'
  t.test_files = FileList['test/test_*.rb']
  t.warning = false
end

PROBLEMS.each do |problem|
  file(File.join(DATA_DIR, "#{problem}_small.bin")) { ruby 'script/make_fixtures.rb', '--size', 'small', problem }
end

desc 'Build the small fixtures that the gates and tests read'
task fixtures: PROBLEMS.map { |problem| File.join(DATA_DIR, "#{problem}_small.bin") }

task test: :fixtures

desc 'Run every gate on the current backend (GPU=1 for Cumo)'
task(gate: :fixtures) { ruby 'script/gate.rb' }

desc 'Build every fixture size, including the large ones the benches read'
task('fixtures:all') { ruby 'script/make_fixtures.rb', '--size', 'all' }

desc 'Run bench/conditions.tsv (GPU: Cumo / CuPy) into tmp/bench'
task bench: 'fixtures:all' do
  sh 'bench/run.sh', 'bench/conditions.tsv', 'tmp/bench'
  sh PYTHON, 'bench/aggregate.py', 'bench/conditions.tsv', 'tmp/bench'
end

desc 'Run bench/conditions_cpu.tsv (CPU: Numo / numpy) into tmp/bench_cpu'
task 'bench:cpu' => 'fixtures:all' do
  sh 'bench/run.sh', 'bench/conditions_cpu.tsv', 'tmp/bench_cpu'
  sh PYTHON, 'bench/aggregate.py', 'bench/conditions_cpu.tsv', 'tmp/bench_cpu', '--labels', 'Numo,numpy'
end

task default: :test
