# frozen_string_literal: true

module LinalgBench
  module Problems
    # E. Polynomial least squares on a Vandermonde matrix. Four routes to the
    # same solution (lstsq, QR, pinv, normal equations) have to agree to within
    # what the condition number allows, and a copy with duplicated columns has
    # to give the minimum-norm solution and a null space of the right size.
    class VandermondeLstsq < Problem
      register 'vandermonde_lstsq'

      DRIVERS = %w[lsd lss lsy].freeze
      TOL = { 'fp64' => 2e-14 }.freeze
      VARIANTS = %w[lstsq qr].freeze

      # numo-linalg-alt, numpy and CuPy default to different cut-offs for
      # small singular values, so every call that has one gets this.
      def self.rcond(a)
        a.shape.max * a.class::EPSILON
      end

      def run(fixture)
        v = fixture.device('V')
        y = fixture.device('y')
        coef = fixture.device('coef')
        { 'solutions' => solutions(v, y, coef), 'qr' => qr(v), 'spectrum' => spectrum(v),
          'deficient' => deficient(fixture.device('V_dup'), y), 'rank_dup' => fixture.attr('rank_dup'),
          'n_dup' => fixture.attr('duplicated').size }.tap do |result|
          result['values'] = {
            'residual_norm' => result['solutions']['residual_norm'],
            'cond' => result['spectrum']['cond'],
            'coef_error' => result['solutions']['coef_error'],
            'min_norm' => result['deficient']['min_norm']
          }
        end
      end

      def gate(result)
        tol = TOL.fetch(LinalgBench.dtype_name) { Gates.tol }
        sol = result['solutions']
        qr = result['qr']
        sp = result['spectrum']
        dd = result['deficient']
        # Solutions of a least-squares problem agree to about cond · eps; the
        # normal equations square the condition number.
        cond = sp['cond']
        [
          Gates.le('lstsq_drivers_agree', sol['drivers'], tol),
          Gates.le('qr_vs_lstsq', sol['qr'] / cond, tol),
          Gates.le('pinv_vs_lstsq', sol['pinv'] / cond, tol),
          Gates.le('normal_equations_vs_lstsq', sol['normal'] / (cond**2), tol),
          Gates.le('coef_error', sol['coef_error'], 1e-2),
          Gates.le('q_complete_orthogonality', qr['q_orthogonality'], tol),
          Gates.le('q_economic_orthogonality', qr['qe_orthogonality'], tol),
          Gates.le('q_r_rebuilds_v', qr['rebuild'], tol),
          Gates.le('q_r_economic_rebuilds_v', qr['rebuild_economic'], tol),
          Gates.le('mode_r_vs_reduce', qr['mode_r'], tol),
          Gates.le('mode_raw_vs_economic', qr['mode_raw'], tol),
          Gates.ge('cond_grows_with_degree', sp['cond_growth_min'], 1.0),
          Gates.le('cond_vs_svd_ratio', sp['cond_vs_svd'], tol),
          Gates.le('svdvals_vs_svd', sp['svdvals_vs_svd'], tol),
          Gates.le('u_s_vt_rebuilds_v', sp['svd_rebuild'], tol),
          Gates.le('norm_fro', sp['norm_fro'], tol),
          Gates.le('norm_2_vs_largest_singular_value', sp['norm_2'], tol),
          Gates.eq('matrix_rank_dup', dd['matrix_rank'], result['rank_dup']),
          Gates.eq('lstsq_rank_dup', dd['lstsq_rank'], result['rank_dup']),
          Gates.eq('null_space_columns', dd['null_columns'], result['n_dup']),
          Gates.le('null_space_annihilates', dd['null_residual'], tol),
          Gates.le('null_space_orthogonality', dd['null_orthogonality'], tol),
          Gates.le('min_norm_lstsq_vs_pinv', dd['pinv'], tol),
          Gates.le('min_norm_beats_shifted', dd['norm_ratio'], 1.0 - 1e-6),
          Gates.le('shifted_keeps_residual', dd['shifted_residual'], tol)
        ]
      end

      def bench_step(fixture, variant = nil)
        v = fixture.device('V')
        y = fixture.device('y')
        case variant || VARIANTS.first
        when 'lstsq'
          rcond = self.class.rcond(v)
          -> { XL.lstsq(v, y, rcond: rcond) }
        when 'qr'
          lambda do
            q, r = XL.qr(v, mode: 'economic')
            XL.solve(r, q.transpose.dot(y))
          end
        else raise Error, "unknown variant #{variant.inspect}; use #{VARIANTS.join(' or ')}"
        end
      end

      private

      def solutions(v, y, coef)
        rcond = self.class.rcond(v)
        by_driver = DRIVERS.map { |driver| XL.lstsq(v, y, driver: driver, rcond: rcond)[0] }
        x = by_driver.first
        q, r = XL.qr(v, mode: 'economic')
        vt = v.transpose
        {
          'drivers' => by_driver.map { |other| Gates.rel_diff(other, x) }.max,
          'qr' => Gates.rel_diff(XL.solve(r, q.transpose.dot(y)), x),
          'pinv' => Gates.rel_diff(XL.pinv(v, rcond: rcond).dot(y), x),
          'normal' => Gates.rel_diff(XL.solve(vt.dot(v), vt.dot(y)), x),
          'coef_error' => Gates.rel_diff(x, coef),
          'residual_norm' => Gates.fro(v.dot(x) - y)
        }
      end

      def qr(v)
        n = v.shape[1]
        q, r = XL.qr(v, mode: 'reduce')
        qe, re = XL.qr(v, mode: 'economic')
        raw, = XL.qr(v, mode: 'raw')
        {
          'q_orthogonality' => Gates.orthogonality(q),
          'qe_orthogonality' => Gates.orthogonality(qe),
          'rebuild' => Gates.rel_diff(q.dot(r), v),
          'rebuild_economic' => Gates.rel_diff(qe.dot(re), v),
          'mode_r' => Gates.rel_diff(XL.qr(v, mode: 'r'), r),
          'mode_raw' => Gates.rel_diff(raw[0...n, true].triu, re)
        }
      end

      def spectrum(v)
        n = v.shape[1]
        s, u, vt = XL.svd(v, job: 'S')
        conds = (2..n).map { |k| LinalgBench.scalar(XL.cond(v[true, 0...k].dup)) }
        cond = LinalgBench.scalar(XL.cond(v))
        s0 = LinalgBench.scalar(s[0])
        {
          'cond' => cond,
          'cond_growth_min' => conds.each_cons(2).map { |a, b| b / a }.min,
          'cond_vs_svd' => ((cond - (s0 / LinalgBench.scalar(s[-1]))) / cond).abs,
          'svdvals_vs_svd' => Gates.max_rel_diff_sorted(XL.svdvals(v), s),
          'svd_rebuild' => Gates.rel_diff((u * s).dot(vt), v),
          'norm_fro' => ((LinalgBench.scalar(XL.norm(v, 'fro')) - Gates.fro(v)) / Gates.fro(v)).abs,
          'norm_2' => ((LinalgBench.scalar(XL.norm(v, 2)) - s0) / s0).abs
        }
      end

      def deficient(vd, y)
        rcond = self.class.rcond(vd)
        x, _resids, rank, = XL.lstsq(vd, y, rcond: rcond)
        z = XL.null_space(vd, rcond: rcond)
        shifted = x + z[true, 0]
        residual = Gates.fro(vd.dot(x) - y)
        {
          'matrix_rank' => LinalgBench.scalar(XL.matrix_rank(vd)).to_i,
          'lstsq_rank' => LinalgBench.scalar(rank).to_i,
          'null_columns' => z.shape[1],
          'null_residual' => Gates.max_abs(vd.dot(z)) / Gates.fro(vd),
          'null_orthogonality' => Gates.orthogonality(z),
          'pinv' => Gates.rel_diff(XL.pinv(vd, rcond: rcond).dot(y), x),
          'min_norm' => Gates.fro(x),
          'norm_ratio' => Gates.fro(x) / Gates.fro(shifted),
          'shifted_residual' => (Gates.fro(vd.dot(shifted) - y) - residual).abs / residual
        }
      end
    end
  end
end
