# frozen_string_literal: true

module LinalgBench
  module Problems
    # F. A Markov chain and an implicit diffusion step. The stationary
    # distribution of P comes out of eig and out of null_space and has to agree;
    # P^k has to converge to it at the rate of the second eigenvalue, and so does
    # expm of the generator Q = P − I. The implicit step M u' = u, solved with one
    # LU factorization, has to conserve the total and never raise the maximum.
    class MarkovDiffusion < Problem
      register 'markov_diffusion'

      TOL = { 'fp64' => 5e-14 }.freeze
      POWERS = [4, 8].freeze
      TIMES = [0.5, 1.0].freeze
      STEPS = 100
      VARIANTS = %w[lu_solve expm matrix_power eig].freeze
      BENCH_POWER = 64

      # Eigenvalues come back complex from geev on every backend.
      def self.stationary_from_eig(pt)
        w, _vl, vr = XL.eig(pt)
        index = LinalgBench.scalar((w.real - 1).abs.min_index).to_i
        pi = vr[true, index].real
        [pi / pi.sum, w]
      end

      def run(fixture)
        p = fixture.device('P')
        q = fixture.device('Q')
        n = p.shape[0]
        eye = p.class.eye(n)
        pi, w = self.class.stationary_from_eig(p.transpose)
        z = XL.null_space(p.transpose - eye)
        pi_null = z[true, 0] / z[true, 0].sum
        moduli = Gates.descending(XL.eigvals(p).abs)
        result = {
          'stationary' => stationary(p, pi, pi_null, z.shape[1]),
          'spectrum' => { 'largest' => moduli[0], 'second' => moduli[1],
                          'eig_vs_eigvals' => Gates.max_rel_diff_sorted(XL.eigvals(p).abs, w.abs) },
          'powers' => powers(p, pi),
          'generator' => generator(q, pi),
          'diffusion' => diffusion(fixture.device('M'), fixture.device('u0'))
        }
        result['values'] = { 'pi_max' => LinalgBench.scalar(pi.max), 'lambda2' => moduli[1],
                             'diffusion_max' => result['diffusion']['final_max'] }
        result
      end

      def gate(result)
        tol = TOL.fetch(LinalgBench.dtype_name) { Gates.tol }
        st = result['stationary']
        sp = result['spectrum']
        pw = result['powers']
        gen = result['generator']
        df = result['diffusion']
        [
          Gates.ge('pi_nonnegative', st['min'], -tol),
          Gates.le('pi_sums_to_one', st['sum_error'], tol),
          Gates.le('pi_is_stationary', st['residual'], tol),
          Gates.eq('null_space_columns', st['null_columns'], 1),
          Gates.le('pi_eig_vs_null_space', st['eig_vs_null'], tol),
          Gates.le('largest_modulus_is_one', (sp['largest'] - 1).abs, tol),
          Gates.le('second_modulus_below_one', sp['second'], 1 - 1e-3),
          Gates.le('eig_vs_eigvals', sp['eig_vs_eigvals'], tol),
          Gates.le('matrix_power_vs_products', pw['vs_products'], tol),
          Gates.le('power_convergence_rate_vs_lambda2', (pw['rate'] / sp['second'] - 1).abs, 0.5),
          Gates.le('expm_rows_sum_to_one', gen['row_sum_error'], tol),
          Gates.ge('expm_nonnegative', gen['min'], -tol),
          Gates.le('expm_semigroup', gen['semigroup'], tol),
          Gates.le('expm_vs_matrix_power_of_expm', gen['squaring'], tol),
          Gates.le('expm_keeps_pi', gen['pi_residual'], tol),
          Gates.le('diffusion_conserves_total', df['total_drift'], tol),
          Gates.le('diffusion_max_never_rises', df['max_rise'], tol),
          Gates.le('p_l_u_rebuilds_m', df['plu'], tol),
          Gates.le('lu_inv_vs_inv', df['lu_inv'], tol),
          Gates.le('lu_solve_vs_solve', df['lu_solve'], tol)
        ]
      end

      def bench_step(fixture, variant = nil)
        case variant || VARIANTS.first
        when 'lu_solve'
          m = fixture.device('M')
          u = fixture.device('u0')
          lu, ipiv = XL.lu_fact(m)
          -> { XL.lu_solve(lu, ipiv, u) }
        when 'expm'
          q = fixture.device('Q')
          -> { XL.expm(q) }
        when 'matrix_power'
          p = fixture.device('P')
          -> { XL.matrix_power(p, BENCH_POWER) }
        when 'eig'
          pt = fixture.device('P').transpose.dup
          -> { XL.eig(pt) }
        else raise Error, "unknown variant #{variant.inspect}; use #{VARIANTS.join(', ')}"
        end
      end

      private

      def stationary(p, pi, pi_null, null_columns)
        {
          'min' => LinalgBench.scalar(pi.min),
          'sum_error' => (LinalgBench.scalar(pi.sum) - 1).abs,
          'residual' => Gates.max_abs(pi.dot(p) - pi),
          'null_columns' => null_columns,
          'eig_vs_null' => Gates.rel_diff(pi_null, pi)
        }
      end

      # Each row of P^k approaches π; the error should shrink by |λ₂| per step.
      def powers(p, pi)
        errors = POWERS.map { |k| Gates.max_abs(XL.matrix_power(p, k) - pi) }
        {
          'vs_products' => Gates.rel_diff(XL.matrix_power(p, 3), p.dot(p).dot(p)),
          'rate' => (errors[1] / errors[0])**(1.0 / (POWERS[1] - POWERS[0]))
        }
      end

      def generator(q, pi)
        e1, e2 = TIMES.map { |t| XL.expm(q * t) }
        ones = q.class.ones(q.shape[0])
        {
          'row_sum_error' => Gates.max_abs(e2.dot(ones) - 1),
          'min' => LinalgBench.scalar(e2.min),
          'semigroup' => Gates.rel_diff(e1.dot(e1), e2),
          'squaring' => Gates.rel_diff(XL.matrix_power(XL.expm(q * (TIMES[1] / 8)), 8), e2),
          'pi_residual' => Gates.max_abs(pi.dot(e2) - pi)
        }
      end

      def diffusion(m, u0)
        lu, ipiv = XL.lu_fact(m)
        total = LinalgBench.scalar(u0.sum)
        u = u0
        max_rise = 0.0
        previous_max = LinalgBench.scalar(u.max)
        STEPS.times do
          u = XL.lu_solve(lu, ipiv, u)
          current = LinalgBench.scalar(u.max)
          max_rise = [max_rise, current - previous_max].max
          previous_max = current
        end
        pm, l, up = XL.lu(m)
        {
          'total_drift' => (LinalgBench.scalar(u.sum) - total).abs / total,
          'max_rise' => max_rise,
          'final_max' => previous_max,
          'plu' => Gates.rel_diff(pm.dot(l).dot(up), m),
          'lu_inv' => Gates.rel_diff(XL.lu_inv(lu, ipiv), XL.inv(m)),
          'lu_solve' => Gates.rel_diff(XL.lu_solve(lu, ipiv, u0), XL.solve(m, u0))
        }
      end
    end
  end
end
