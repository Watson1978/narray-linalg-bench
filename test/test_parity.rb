# frozen_string_literal: true

require_relative 'test_helper'

# Numo::Linalg against Cumo::Linalg on the same fixture bytes, function by
# function. Only results that are unique up to rounding are compared directly;
# factors with a sign or ordering freedom are compared through what they rebuild.
class TestParity < Test::Unit::TestCase
  include TestHelper

  TOL = { HM::DFloat => 1e-12, HM::SFloat => 1e-4 }.freeze

  def setup
    LinalgBench.add_cumo_load_path
    require 'cumo/narray'
    require 'cumo/linalg'
  rescue LoadError, LinalgBench::Error => e
    omit("cumo is not available: #{e.message}")
  end

  def numo(host, type) = type.cast(host)
  def cumo(host, type) = Cumo.const_get(type.name.split('::').last).cast(Cumo::DFloat.from_binary(host.to_binary, host.shape))
  def back(array) = HM::DFloat.from_binary(Cumo::DFloat.cast(array).to_binary, array.shape)

  def assert_close(expected, actual, type, message)
    expected = HM::DFloat.cast(expected)
    actual = back(actual)
    diff = Math.sqrt((expected - actual).square.sum) / Math.sqrt(expected.square.sum)
    assert_operator(diff, :<=, TOL[type], "#{message} (#{type}): relative difference #{diff}")
  end

  def assert_scalar_close(expected, actual, type, message)
    expected = LinalgBench.scalar(expected)
    actual = LinalgBench.scalar(actual)
    assert_in_delta(expected, actual, TOL[type] * [expected.abs, 1].max, "#{message} (#{type})")
  end

  def gaussian_process_kernel
    fixture = require_fixture('gaussian_process', 'small')
    x = fixture['X']
    d2 = ((x.expand_dims(1) - x.expand_dims(0))**2).sum(axis: 2)
    k = (fixture.attr('sf')**2) * HM::NMath.exp(-d2 / (2 * (fixture.attr('ell')**2)))
    [k + ((fixture.attr('sn')**2) * HM::DFloat.eye(x.shape[0])), fixture['y']]
  end

  [HM::DFloat, HM::SFloat].each do |type|
    short = type.name.split('::').last

    define_method("test_cholesky_family_#{short}") do
      k_host, y_host = gaussian_process_kernel
      k_n = numo(k_host, type)
      k_c = cumo(k_host, type)
      y_n = numo(y_host, type)
      y_c = cumo(y_host, type)
      %w[U L].each do |uplo|
        l_n = HM::Linalg.cholesky(k_n, uplo: uplo)
        l_c = Cumo::Linalg.cholesky(k_c, uplo: uplo)
        assert_close(l_n, l_c, type, "cholesky uplo=#{uplo}")
        f_n = HM::Linalg.cho_fact(k_n, uplo: uplo)
        f_c = Cumo::Linalg.cho_fact(k_c, uplo: uplo)
        assert_close(f_n, f_c, type, "cho_fact uplo=#{uplo}")
        assert_close(HM::Linalg.cho_solve(f_n, y_n, uplo: uplo), Cumo::Linalg.cho_solve(f_c, y_c, uplo: uplo),
                     type, "cho_solve uplo=#{uplo}")
        inv_n = HM::Linalg.cho_inv(f_n, uplo: uplo)
        inv_c = Cumo::Linalg.cho_inv(f_c, uplo: uplo)
        part = uplo == 'U' ? :triu : :tril
        assert_close(inv_n.send(part), inv_c.send(part), type, "cho_inv uplo=#{uplo}")
      end
    end

    define_method("test_least_squares_family_#{short}") do
      fixture = require_fixture('vandermonde_lstsq', 'small')
      v_n = numo(fixture['V'], type)
      v_c = cumo(fixture['V'], type)
      vd_n = numo(fixture['V_dup'], type)
      vd_c = cumo(fixture['V_dup'], type)
      y_n = numo(fixture['y'], type)
      y_c = cumo(fixture['y'], type)
      rcond = v_n.shape.max * type::EPSILON
      x_n, _, rank_n, s_n = HM::Linalg.lstsq(v_n, y_n, rcond: rcond)
      x_c, _, rank_c, s_c = Cumo::Linalg.lstsq(v_c, y_c, rcond: rcond)
      assert_close(x_n, x_c, type, 'lstsq x')
      assert_equal(rank_n, LinalgBench.scalar(rank_c), 'lstsq rank')
      assert_close(s_n, s_c, type, 'lstsq singular values')
      assert_close(HM::Linalg.pinv(v_n), Cumo::Linalg.pinv(v_c), type, 'pinv')
      assert_close(HM::Linalg.svdvals(v_n), Cumo::Linalg.svdvals(v_c), type, 'svdvals')
      assert_close(HM::Linalg.svd(v_n, job: 'S')[0], Cumo::Linalg.svd(v_c, job: 'S')[0], type, 'svd s')
      assert_scalar_close(HM::Linalg.cond(v_n), Cumo::Linalg.cond(v_c), type, 'cond')
      assert_scalar_close(HM::Linalg.norm(v_n, 'fro'), Cumo::Linalg.norm(v_c, 'fro'), type, 'norm fro')
      assert_scalar_close(HM::Linalg.norm(v_n, 2), Cumo::Linalg.norm(v_c, 2), type, 'norm 2')
      assert_equal(HM::Linalg.matrix_rank(vd_n), LinalgBench.scalar(Cumo::Linalg.matrix_rank(vd_c)), 'matrix_rank')
      %w[economic reduce].each do |mode|
        q_n, = HM::Linalg.qr(v_n, mode: mode)
        q_c, = Cumo::Linalg.qr(v_c, mode: mode)
        k = v_n.shape[1]
        assert_close(q_n[true, 0...k].dot(q_n[true, 0...k].transpose), q_c[true, 0...k].dot(q_c[true, 0...k].transpose),
                     type, "qr #{mode} projector")
      end
      assert_close(HM::Linalg.qr(v_n, mode: 'r').abs, Cumo::Linalg.qr(v_c, mode: 'r').abs, type, 'qr r up to signs')
      z_n = HM::Linalg.null_space(vd_n, rcond: rcond)
      z_c = Cumo::Linalg.null_space(vd_c, rcond: rcond)
      assert_equal(z_n.shape, z_c.shape, 'null_space shape')
      assert_close(z_n.dot(z_n.transpose), z_c.dot(z_c.transpose), type, 'null_space projector')
    end

    define_method("test_spectral_family_#{short}") do
      fixture = require_fixture('procrustes_pca', 'small')
      x_host = fixture['X'] - fixture['X'].mean(axis: 0)
      cov_host = x_host.transpose.dot(x_host) / (x_host.shape[0] - 1)
      cov_n = numo(cov_host, type)
      cov_c = cumo(cov_host, type)
      x_n = numo(x_host, type)
      x_c = cumo(x_host, type)
      w_n, v_n = HM::Linalg.eigh(cov_n)
      w_c, v_c = Cumo::Linalg.eigh(cov_c)
      assert_close(w_n, w_c, type, 'eigh values')
      top_n = v_n[true, -3..].dup
      top_c = v_c[true, -3..].dup
      assert_close(top_n.dot(top_n.transpose), top_c.dot(top_c.transpose), type, 'eigh top-3 projector')
      assert_close(HM::Linalg.eigvalsh(cov_n), Cumo::Linalg.eigvalsh(cov_c), type, 'eigvalsh')
      assert_scalar_close(HM::Linalg.norm(x_n, 'nuc'), Cumo::Linalg.norm(x_c, 'nuc'), type, 'norm nuc')
      low_n = numo(fixture['X_low'], type)
      low_c = cumo(fixture['X_low'], type)
      o_n = HM::Linalg.orth(low_n)
      o_c = Cumo::Linalg.orth(low_c)
      assert_equal(o_n.shape, o_c.shape, 'orth shape')
      assert_close(o_n.dot(o_n.transpose), o_c.dot(o_c.transpose), type, 'orth projector')
      r0_n = numo(fixture['R0'], type)
      r0_c = cumo(fixture['R0'], type)
      assert_scalar_close(HM::Linalg.det(r0_n), Cumo::Linalg.det(r0_c), type, 'det of a rotation')
      sign_n, logdet_n = HM::Linalg.slogdet(r0_n)
      sign_c, logdet_c = Cumo::Linalg.slogdet(r0_c)
      assert_equal(sign_n, LinalgBench.scalar(sign_c), 'slogdet sign of a rotation')
      assert_in_delta(logdet_n, LinalgBench.scalar(logdet_c), TOL[type], 'slogdet logdet of a rotation')
    end

    define_method("test_markov_family_#{short}") do
      fixture = require_fixture('markov_diffusion', 'small')
      p_n = numo(fixture['P'], type)
      p_c = cumo(fixture['P'], type)
      q_n = numo(fixture['Q'], type)
      q_c = cumo(fixture['Q'], type)
      m_n = numo(fixture['M'], type)
      m_c = cumo(fixture['M'], type)
      u_n = numo(fixture['u0'], type)
      u_c = cumo(fixture['u0'], type)
      moduli = ->(w) { HM::DFloat.cast(w.is_a?(HM::NArray) ? w.abs : back(w.abs)).sort }
      assert_close(moduli.call(HM::Linalg.eigvals(p_n)), moduli.call(Cumo::Linalg.eigvals(p_c)), type, 'eigvals moduli')
      w_n, = HM::Linalg.eig(p_n)
      w_c, = Cumo::Linalg.eig(p_c)
      assert_close(moduli.call(w_n), moduli.call(w_c), type, 'eig moduli')
      assert_close(HM::Linalg.matrix_power(p_n, 8), Cumo::Linalg.matrix_power(p_c, 8), type, 'matrix_power')
      assert_close(HM::Linalg.expm(q_n), Cumo::Linalg.expm(q_c), type, 'expm')
      lu_n, piv_n = HM::Linalg.lu_fact(m_n)
      lu_c, piv_c = Cumo::Linalg.lu_fact(m_c)
      assert_close(lu_n, lu_c, type, 'lu_fact')
      assert_equal(piv_n.to_a, piv_c.to_a, 'lu_fact pivots')
      assert_close(HM::Linalg.lu_solve(lu_n, piv_n, u_n), Cumo::Linalg.lu_solve(lu_c, piv_c, u_c), type, 'lu_solve')
      assert_close(HM::Linalg.lu_inv(lu_n, piv_n), Cumo::Linalg.lu_inv(lu_c, piv_c), type, 'lu_inv')
      HM::Linalg.lu(m_n).zip(Cumo::Linalg.lu(m_c), %w[P L U]).each do |a, b, name|
        assert_close(a, b, type, "lu #{name}")
      end
      z_n = HM::Linalg.null_space(p_n.transpose - type.eye(16))
      z_c = Cumo::Linalg.null_space(p_c.transpose - Cumo.const_get(type.name.split('::').last).eye(16))
      assert_close(z_n.dot(z_n.transpose), z_c.dot(z_c.transpose), type, 'null_space projector')
    end

    define_method("test_control_family_#{short}") do
      fixture = require_fixture('kalman_riccati', 'small')
      a_host = fixture['A_c'] * fixture.attr('dt')
      a_n = numo(a_host, type)
      a_c = cumo(a_host, type)
      e_n = HM::Linalg.expm(a_n)
      e_c = Cumo::Linalg.expm(a_c)
      assert_close(e_n, e_c, type, 'expm of a non-normal matrix')
      assert_close(HM::Linalg.matrix_power(e_n, 16), Cumo::Linalg.matrix_power(e_c, 16), type, 'matrix_power')
      moduli = ->(w) { HM::DFloat.cast(w.is_a?(HM::NArray) ? w.abs : back(w.abs)).sort }
      assert_close(moduli.call(HM::Linalg.eigvals(e_n)), moduli.call(Cumo::Linalg.eigvals(e_c)), type, 'eigvals moduli')
      s_host = fixture['C'].dot(fixture['Q']).dot(fixture['C'].transpose) + fixture['R']
      s_n = numo(s_host, type)
      s_c = cumo(s_host, type)
      assert_scalar_close(HM::Linalg.cond(s_n), Cumo::Linalg.cond(s_c), type, 'cond')
      assert_scalar_close(HM::Linalg.norm(s_n, 2), Cumo::Linalg.norm(s_c, 2), type, 'norm 2')
    end

    define_method("test_symmetric_indefinite_family_#{short}") do
      fixture = require_fixture('kkt', 'small')
      a = fixture['A']
      bc = fixture['B']
      n = a.shape[1]
      p = bc.shape[0]
      k_host = HM::DFloat.zeros(n + p, n + p)
      k_host[0...n, 0...n] = a.transpose.dot(a)
      k_host[0...n, n...(n + p)] = bc.transpose
      k_host[n...(n + p), 0...n] = bc
      k_n = numo(k_host, type)
      k_c = cumo(k_host, type)
      %w[U L].each do |uplo|
        u_n, d_n, perm_n = HM::Linalg.ldl(k_n, uplo: uplo)
        u_c, d_c, perm_c = Cumo::Linalg.ldl(k_c, uplo: uplo)
        assert_close(u_n, u_c, type, "ldl factor uplo=#{uplo}")
        assert_close(d_n, d_c, type, "ldl d uplo=#{uplo}")
        assert_equal(perm_n.to_a, perm_c.to_a, "ldl perm uplo=#{uplo}")
      end
      assert_close(HM::Linalg.eigvalsh(k_n), Cumo::Linalg.eigvalsh(k_c), type, 'eigvalsh of an indefinite matrix')
      assert_close(HM::Linalg.inv(k_n, driver: 'sysv'), Cumo::Linalg.inv(k_c, driver: 'sysv'), type, 'inv driver sysv')
      rhs = numo(HM::DFloat.new(n + p).seq, type)
      %w[sym gen].each do |driver|
        assert_close(HM::Linalg.solve(k_n, rhs, driver: driver),
                     Cumo::Linalg.solve(k_c, cumo(HM::DFloat.new(n + p).seq, type), driver: driver),
                     type, "solve driver #{driver}")
      end
    end

    define_method("test_generalized_eigen_family_#{short}") do
      fixture = require_fixture('hartree_fock', 'small')
      %w[h2 heh].each do |name|
        h_n = numo(fixture["#{name}_H"], type)
        h_c = cumo(fixture["#{name}_H"], type)
        s_n = numo(fixture["#{name}_S"], type)
        s_c = cumo(fixture["#{name}_S"], type)
        w_n, v_n = HM::Linalg.eigh(h_n, s_n)
        w_c, v_c = Cumo::Linalg.eigh(h_c, s_c)
        assert_close(w_n, w_c, type, "#{name} eigh(H, S) values")
        assert_close(v_n.abs, v_c.abs, type, "#{name} eigh(H, S) vectors up to sign")
        assert_close(HM::Linalg.eigvalsh(h_n, s_n), Cumo::Linalg.eigvalsh(h_c, s_c), type, "#{name} eigvalsh(H, S)")
        assert_close(HM::Linalg.eigh(h_n, s_n, turbo: true)[0], Cumo::Linalg.eigh(h_c, s_c, turbo: true)[0], type,
                     "#{name} eigh(H, S, turbo: true)")
      end
    end

    define_method("test_solve_inv_and_determinants_#{short}") do
      k_host, y_host = gaussian_process_kernel
      k_n = numo(k_host, type)
      k_c = cumo(k_host, type)
      assert_close(HM::Linalg.solve(k_n, numo(y_host, type)), Cumo::Linalg.solve(k_c, cumo(y_host, type)), type, 'solve')
      assert_close(HM::Linalg.inv(k_n), Cumo::Linalg.inv(k_c), type, 'inv')
      sign_n, logdet_n = HM::Linalg.slogdet(k_n)
      sign_c, logdet_c = Cumo::Linalg.slogdet(k_c)
      assert_equal(sign_n, LinalgBench.scalar(sign_c), 'slogdet sign')
      assert_scalar_close(logdet_n, logdet_c, type, 'slogdet logdet')
      det_n = HM::Linalg.det(k_n)
      det_c = LinalgBench.scalar(Cumo::Linalg.det(k_c))
      assert_operator((det_c - det_n).abs, :<=, TOL[type] * det_n.abs, "det (#{type}): #{det_n} vs #{det_c}")
    end
  end
end
