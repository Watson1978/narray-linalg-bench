#!/usr/bin/env ruby
# frozen_string_literal: true

# Builds data/<problem>_<size>.{bin,json} on the host, in fp64, from a fixed
# seed. This is the only place random numbers are drawn: Numo, Cumo and numpy
# do not share a generator, so every implementation reads these bytes instead.
#
#   ruby script/make_fixtures.rb                     # every problem, small
#   ruby script/make_fixtures.rb --size small gaussian_process
#   ruby script/make_fixtures.rb --force             # rewrite existing files

require 'optparse'
require_relative '../lib/linalg_bench'

module FixtureGenerators
  DFloat = HM::DFloat
  Linalg = HM::Linalg

  SEEDS = {
    'hartree_fock' => 1001, 'gaussian_process' => 1002, 'kalman_riccati' => 1003,
    'procrustes_pca' => 1004, 'vandermonde_lstsq' => 1005, 'markov_diffusion' => 1006, 'kkt' => 1007
  }.freeze

  SIZES = {
    'hartree_fock' => {
      'small' => {},
      **[128, 512, 1024, 2048, 4096].to_h { |n| ["n#{n}", { n: n, bench: true }] }
    },
    'gaussian_process' => {
      'small' => { n: 64, d: 2, n_test: 36 },
      **[512, 1024, 2048, 4096, 8192].to_h { |n| ["n#{n}", { n: n, d: 2, n_test: 36 }] }
    },
    'kalman_riccati' => {
      'small' => { n: 8, m: 3, steps: 2000 },
      **[64, 256, 1024, 4096].to_h { |n| ["n#{n}", { n: n, m: n / 4, bench: true }] }
    },
    'procrustes_pca' => {
      'small' => { points: 1000, d: 16, rank: 10 },
      **[256, 512, 1024, 2048].to_h { |d| ["N8192_d#{d}", { points: 8192, d: d, bench: true }] }
    },
    'vandermonde_lstsq' => {
      'small' => { m: 256, n: 8 },
      **[4096, 16_384, 65_536].product([16, 64, 256]).to_h { |m, n| ["m#{m}_n#{n}", { m: m, n: n, bench: true }] }
    },
    'markov_diffusion' => {
      'small' => { n: 16 },
      **[256, 1024, 4096].to_h { |n| ["n#{n}", { n: n }] }
    },
    'kkt' => {
      'small' => { m: 32, n: 16, p: 4 },
      **[512, 1024, 2048, 4096].to_h { |size| ["n#{size}", { m: 2 * (size - (size / 8)), n: size - (size / 8), p: size / 8 }] }
    }
  }.freeze

  module_function

  def randn(*shape) = DFloat.new(*shape).rand_norm
  def rand(*shape) = DFloat.new(*shape).rand

  # Haar-ish orthogonal matrix: Q from QR with the signs of R's diagonal
  # folded in, so the result does not depend on LAPACK's sign convention.
  def orthogonal(n)
    q, r = Linalg.qr(randn(n, n))
    q * r.diagonal.sign
  end

  def rbf(x1, x2, ell, sf)
    d2 = (x1 * x1).sum(axis: 1).expand_dims(1) + (x2 * x2).sum(axis: 1).expand_dims(0) - (2 * x1.dot(x2.transpose))
    sf**2 * HM::NMath.exp(-d2 / (2 * ell**2))
  end

  # The bench sizes keep the shape of HF without integrals: a symmetric H
  # and an SPD S with a unit diagonal, like an overlap matrix.
  def hartree_fock(spec)
    if spec[:bench]
      n = spec[:n]
      g = randn(n, n)
      h = (g + g.transpose) / 2
      w = randn(n, n)
      s = (w.dot(w.transpose) / n) + DFloat.eye(n)
      scale = 1 / HM::NMath.sqrt(s.diagonal)
      return [{ 'H' => h, 'S' => s * scale.expand_dims(1) * scale.expand_dims(0) }, {}]
    end

    source = JSON.parse(File.read(File.join(__dir__, 'hartree_fock_sto3g.json')))
    arrays = {}
    attrs = { 'source' => source['source'] }
    source['molecules'].each do |name, mol|
      arrays["#{name}_S"] = DFloat[*mol['S']]
      arrays["#{name}_H"] = DFloat[*mol['H']]
      arrays["#{name}_eri"] = DFloat[*mol['eri']]
      %w[e_nuc n_electrons e_total].each { |key| attrs["#{name}_#{key}"] = mol[key] }
    end
    [arrays, attrs]
  end

  def gaussian_process(spec)
    n = spec[:n]
    d = spec[:d]
    ell = 0.3
    sf = 1.0
    sn = 0.1
    x = rand(n, d)
    k = rbf(x, x, ell, sf) + (sn**2 * DFloat.eye(n))
    y = Linalg.cholesky(k, uplo: 'L').dot(randn(n))
    side = Math.sqrt(spec[:n_test]).ceil
    grid = DFloat.new(side).seq / (side - 1)
    x_test = DFloat.zeros(side * side, 2)
    x_test[true, 0] = grid.expand_dims(1).tile(1, side).flatten
    x_test[true, 1] = grid.tile(side)
    [{ 'X' => x, 'y' => y, 'X_test' => x_test }, { 'ell' => ell, 'sf' => sf, 'sn' => sn }]
  end

  def kalman_riccati(spec)
    n = spec[:n]
    m = spec[:m]
    dt = 0.1
    g = randn(n, n)
    s = randn(n, n)
    # A negative definite symmetric part makes every eigenvalue's real part negative.
    a_c = -((g.dot(g.transpose) / n) + (0.5 * DFloat.eye(n))) + ((s - s.transpose) / 2)
    c = randn(m, n)
    h = randn(n, n)
    q = (0.1 * h.dot(h.transpose) / n) + (0.01 * DFloat.eye(n))
    r = 0.1 * DFloat.eye(m)
    return [{ 'A_c' => a_c, 'C' => c, 'Q' => q, 'R' => r }, { 'dt' => dt }] if spec[:bench]

    a_d = Linalg.expm(a_c * dt)
    lq = Linalg.cholesky(q, uplo: 'L')
    lr = Linalg.cholesky(r, uplo: 'L')
    xs = DFloat.zeros(spec[:steps] + 1, n)
    ys = DFloat.zeros(spec[:steps], m)
    xs[0, true] = randn(n)
    spec[:steps].times do |k|
      ys[k, true] = c.dot(xs[k, true]) + lr.dot(randn(m))
      xs[k + 1, true] = a_d.dot(xs[k, true]) + lq.dot(randn(n))
    end
    # Small entries but a large 1-norm: what expm's scaling has to handle.
    # Symmetric, so that eigh gives an independent reference. Drawn last so
    # that the arrays above keep their bytes.
    uniform = (rand(256, 256) * 2) - 1
    wide = (uniform + uniform.transpose) / 2
    [{ 'A_c' => a_c, 'C' => c, 'Q' => q, 'R' => r, 'xs' => xs, 'ys' => ys, 'E_wide' => wide }, { 'dt' => dt }]
  end

  def procrustes_pca(spec)
    points = spec[:points]
    d = spec[:d]
    if spec[:bench]
      spectrum = 2.0**-(DFloat.new(d).seq / (d / 8.0))
      return [{ 'X' => (randn(points, d) * spectrum).dot(orthogonal(d)) }, {}]
    end

    rank = spec[:rank]
    noise = 0.01
    pts = randn(points, 3)
    r0 = orthogonal(3)
    r0[true, 2] *= -1 if Linalg.det(r0).negative?
    t = randn(3)
    reflect = DFloat[[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    q_clean = pts.dot(r0.transpose) + t
    q_noisy = q_clean + (noise * randn(points, 3))
    q_reflected = pts.dot(r0.dot(reflect).transpose) + t
    spectrum = 2.0**-(DFloat.new(d).seq / 2)
    x = (randn(points, d) * spectrum).dot(orthogonal(d))
    x_low = randn(points, rank).dot(randn(rank, d))
    arrays = { 'P' => pts, 'R0' => r0, 't' => t, 'Q_clean' => q_clean, 'Q_noisy' => q_noisy,
               'Q_reflected' => q_reflected, 'X' => x, 'X_low' => x_low }
    [arrays, { 'noise' => noise, 'rank' => rank }]
  end

  def vandermonde_lstsq(spec)
    m = spec[:m]
    n = spec[:n]
    noise = 1e-3
    x = (DFloat.new(m).seq * 2 / (m - 1)) - 1
    v = DFloat.zeros(m, n)
    n.times { |j| v[true, j] = x**j }
    coef = randn(n)
    y = v.dot(coef) + (noise * randn(m))
    return [{ 'x' => x, 'V' => v, 'coef' => coef, 'y' => y }, { 'noise' => noise }] if spec[:bench]

    duplicated = [[n - 1, 2], [n - 2, 1]]
    v_dup = v.dup
    duplicated.each { |dst, src| v_dup[true, dst] = v[true, src] }
    [{ 'x' => x, 'V' => v, 'V_dup' => v_dup, 'coef' => coef, 'y' => y },
     { 'noise' => noise, 'duplicated' => duplicated, 'rank_dup' => n - duplicated.size }]
  end

  def markov_diffusion(spec)
    n = spec[:n]
    weights = rand(n, n) + 0.05
    p = weights / weights.sum(axis: 1).expand_dims(1)
    # Q = P − I is a generator with the same stationary distribution as P.
    q = p - DFloat.eye(n)
    # Neumann Laplacian: rows and columns sum to zero, so I - r L conserves Σu.
    lap = (-2 * DFloat.eye(n))
    (n - 1).times do |i|
      lap[i, i + 1] = 1
      lap[i + 1, i] = 1
    end
    lap[0, 0] = -1
    lap[n - 1, n - 1] = -1
    r = 0.5
    m = DFloat.eye(n) - (r * lap)
    u0 = HM::NMath.exp(-(((DFloat.new(n).seq / (n - 1)) - 0.3)**2) / 0.01)
    [{ 'P' => p, 'Q' => q, 'M' => m, 'u0' => u0 }, { 'r' => r }]
  end

  def kkt(spec)
    [{ 'A' => randn(spec[:m], spec[:n]), 'b' => randn(spec[:m]),
       'B' => randn(spec[:p], spec[:n]), 'c' => randn(spec[:p]) }, {}]
  end

  def build(problem, size)
    spec = SIZES.fetch(problem).fetch(size)
    HM::NArray.srand(SEEDS.fetch(problem))
    arrays, attrs = send(problem, spec)
    attrs = attrs.merge('seed' => SEEDS.fetch(problem), 'spec' => spec.transform_keys(&:to_s))
    LinalgBench::Fixtures.write(problem, size, arrays, attrs: attrs)
  end
end

if $PROGRAM_NAME == __FILE__
  options = { size: 'small', force: false }
  OptionParser.new do |parser|
    parser.banner = 'Usage: ruby script/make_fixtures.rb [--size small] [--force] [problem ...]'
    parser.on('--size NAME', 'small (既定)、n512 など。all で題材のサイズすべて') { |v| options[:size] = v }
    parser.on('--force', '既にあるファイルも作り直す') { options[:force] = true }
  end.parse!
  problems = ARGV.empty? ? FixtureGenerators::SIZES.keys : ARGV
  problems.each do |problem|
    sizes = FixtureGenerators::SIZES.fetch(problem)
    names = options[:size] == 'all' ? sizes.keys : [options[:size]]
    names.each do |size|
      unless sizes.key?(size)
        warn "#{problem}: no #{size} size yet"
        next
      end
      if !options[:force] && LinalgBench::Fixtures.exist?(problem, size)
        puts "#{problem}_#{size}: exists"
        next
      end
      manifest = FixtureGenerators.build(problem, size)
      puts "#{problem}_#{size}: #{manifest['bytes']} bytes, #{manifest['sha256'][0, 16]}"
    end
  end
end
