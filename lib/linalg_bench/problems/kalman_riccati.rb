# frozen_string_literal: true

module LinalgBench
  module Problems
    # C. A stable continuous-time system, discretized with expm, filtered with
    # the steady-state Kalman predictor from the discrete Riccati iteration.
    # The iteration has to reach its fixed point by two routes (solve and inv),
    # the closed loop has to be stable, and the filter's prediction errors on
    # the simulated trajectory have to have the covariance the iteration claims.
    class KalmanRiccati < Problem
      register 'kalman_riccati'

      TOL = { 'fp64' => 5e-14 }.freeze
      EXPM_TOL = 1e-10
      ITERATIONS = 500
      SQUARINGS = 4
      BURN_IN = 100
      VARIANTS = %w[riccati expm].freeze

      # One step of P ← A P Aᵀ − A P Cᵀ (C P Cᵀ + R)⁻¹ C P Aᵀ + Q. Without
      # the symmetrization, the antisymmetric part of the rounding error grows
      # every step and the iteration blows up after about 100 steps.
      def self.riccati_step(a, c, q, r, p, route)
        apct = a.dot(p).dot(c.transpose)
        s = c.dot(p).dot(c.transpose) + r
        correction = route == :inv ? apct.dot(XL.inv(s)).dot(apct.transpose) : apct.dot(XL.solve(s, apct.transpose))
        nxt = a.dot(p).dot(a.transpose) - correction + q
        (nxt + nxt.transpose) / 2
      end

      def self.riccati(a, c, q, r, route)
        p = q
        ITERATIONS.times { p = riccati_step(a, c, q, r, p, route) }
        p
      end

      def run(fixture)
        a_c = fixture.device('A_c')
        c = fixture.device('C')
        q = fixture.device('Q')
        r = fixture.device('R')
        dt = fixture.attr('dt')
        a = XL.expm(a_c * dt)
        p = self.class.riccati(a, c, q, r, :solve)
        p_inv = self.class.riccati(a, c, q, r, :inv)
        s = c.dot(p).dot(c.transpose) + r
        gain = a.dot(p).dot(c.transpose).dot(XL.inv(s))
        result = {
          'discretization' => discretization(a_c, dt, a).merge('wide' => expm_wide(fixture.device('E_wide'))),
          'riccati' => { 'residual' => Gates.rel_diff(self.class.riccati_step(a, c, q, r, p, :solve), p),
                         'solve_vs_inv' => Gates.rel_diff(p_inv, p),
                         'norm_fro' => rel(XL.norm(p, 'fro'), Gates.fro(p)) },
          'stability' => stability(a, a - gain.dot(c)),
          'conditioning' => conditioning(s),
          'filter' => filter(a, c, gain, fixture.device('xs'), fixture.device('ys'), p)
        }
        result['values'] = { 'trace_p' => LinalgBench.scalar(p.diagonal.sum),
                             'closed_loop_radius' => result['stability']['closed_loop_radius'],
                             'cond_s' => result['conditioning']['cond'] }
        result
      end

      def gate(result)
        tol = TOL.fetch(LinalgBench.dtype_name) { Gates.tol }
        disc = result['discretization']
        ric = result['riccati']
        st = result['stability']
        cn = result['conditioning']
        fl = result['filter']
        [
          Gates.le('expm_vs_matrix_power_of_expm', disc['squaring'], tol),
          Gates.le('expm_vs_eigh_wide_symmetric', disc['wide'], EXPM_TOL),
          Gates.le('open_loop_radius_below_one', st['open_loop_radius'], 1 - 1e-6),
          Gates.le('eig_vs_eigvals', st['eig_vs_eigvals'], tol),
          Gates.le('riccati_fixed_point', ric['residual'], tol),
          Gates.le('riccati_solve_vs_inv', ric['solve_vs_inv'], tol),
          Gates.le('norm_fro', ric['norm_fro'], tol),
          Gates.le('closed_loop_radius_below_one', st['closed_loop_radius'], 1 - 1e-6),
          Gates.le('cond_vs_norm_product', cn['cond_vs_norms'], tol),
          Gates.le('filter_error_variance_vs_p', fl['variance_ratio_error'], 0.2)
        ]
      end

      def bench_step(fixture, variant = nil)
        case variant || VARIANTS.first
        when 'riccati'
          a = XL.expm(fixture.device('A_c') * fixture.attr('dt'))
          c = fixture.device('C')
          q = fixture.device('Q')
          r = fixture.device('R')
          p = q
          -> { p = self.class.riccati_step(a, c, q, r, p, :solve) }
        when 'expm'
          a_c = fixture.device('A_c')
          -> { XL.expm(a_c) }
        else raise Error, "unknown variant #{variant.inspect}; use #{VARIANTS.join(' or ')}"
        end
      end

      private

      def discretization(a_c, dt, a)
        small = XL.expm(a_c * (dt / (2**SQUARINGS)))
        { 'squaring' => Gates.rel_diff(XL.matrix_power(small, 2**SQUARINGS), a) }
      end

      # E is symmetric with entries in [−1, 1] but a 1-norm near 100, which a
      # scaling that looks only at the largest entry gets wrong. V diag(e^λ) Vᵀ
      # from eigh is an independent reference.
      def expm_wide(e)
        w, v = XL.eigh(e)
        Gates.rel_diff(XL.expm(e), (v * XM::NMath.exp(w)).dot(v.transpose))
      end

      def stability(a, closed)
        w, = XL.eig(a)
        open_moduli = Gates.descending(XL.eigvals(a).abs)
        {
          'open_loop_radius' => open_moduli.first,
          'eig_vs_eigvals' => Gates.max_rel_diff_sorted(w.abs, XL.eigvals(a).abs),
          'closed_loop_radius' => Gates.descending(XL.eigvals(closed).abs).first
        }
      end

      def conditioning(s)
        cond = LinalgBench.scalar(XL.cond(s))
        product = LinalgBench.scalar(XL.norm(s, 2)) * LinalgBench.scalar(XL.norm(XL.inv(s), 2))
        { 'cond' => cond, 'cond_vs_norms' => ((cond - product) / product).abs }
      end

      # Runs the steady-state predictor x̂ₖ₊₁ = A x̂ₖ + L (yₖ − C x̂ₖ) along the
      # simulated trajectory and compares the prediction errors' variances
      # with diag(P).
      def filter(a, c, gain, xs, ys, p)
        steps = ys.shape[0]
        n = a.shape[0]
        x_hat = xs.class.zeros(n)
        errors = xs.class.zeros(steps, n)
        steps.times do |k|
          errors[k, true] = xs[k, true] - x_hat
          x_hat = a.dot(x_hat) + gain.dot(ys[k, true] - c.dot(x_hat))
        end
        tail = errors[BURN_IN...steps, true]
        variances = LinalgBench.host((tail * tail).mean(axis: 0))
        expected = LinalgBench.host(p.diagonal)
        { 'variance_ratio_error' => ((variances / expected) - 1).abs.max }
      end

      def rel(actual, expected)
        ((LinalgBench.scalar(actual) - expected) / expected).abs
      end
    end
  end
end
