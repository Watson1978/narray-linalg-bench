# frozen_string_literal: true

module LinalgBench
  module Problems
    # G. Equality-constrained least squares, min ‖A x − b‖² s.t. B x = c,
    # through its KKT system K = [[AᵀA, Bᵀ], [B, 0]], which is symmetric and
    # indefinite. The solution has to satisfy the constraints and the Lagrange
    # condition, agree with the null-space elimination route, and K has to have
    # inertia (n, p, 0), the same as the D of its LDLᵀ factorization.
    class Kkt < Problem
      register 'kkt'

      TOL = { 'fp64' => 2e-14 }.freeze
      VARIANTS = %w[solve eigvalsh ldl].freeze

      def self.system(a, b, bc, c)
        n = a.shape[1]
        p = bc.shape[0]
        k = a.class.zeros(n + p, n + p)
        k[0...n, 0...n] = a.transpose.dot(a)
        k[0...n, n...(n + p)] = bc.transpose
        k[n...(n + p), 0...n] = bc
        rhs = a.class.zeros(n + p)
        rhs[0...n] = a.transpose.dot(b)
        rhs[n...(n + p)] = c
        [k, rhs]
      end

      def self.inertia(values, scale)
        host = LinalgBench.host(values).to_a
        cut = scale * host.size * values.class::EPSILON
        [host.count { |v| v > cut }, host.count { |v| v < -cut }, host.count { |v| v.abs <= cut }]
      end

      def run(fixture)
        a = fixture.device('A')
        b = fixture.device('b')
        bc = fixture.device('B')
        c = fixture.device('c')
        n = a.shape[1]
        k, rhs = self.class.system(a, b, bc, c)
        sym = XL.solve(k, rhs, driver: 'sym')
        gen = XL.solve(k, rhs, driver: 'gen')
        x = sym[0...n]
        lambda = sym[n..]
        eliminated = eliminate(a, b, bc, c)
        scale = Gates.max_abs(k)
        result = {
          'n' => n, 'p' => bc.shape[0],
          'constraint' => Gates.fro(bc.dot(x) - c) / Gates.fro(c),
          'lagrange' => Gates.fro(a.transpose.dot(a.dot(x) - b) + bc.transpose.dot(lambda)) / Gates.fro(a.transpose.dot(b)),
          'sym_vs_gen' => Gates.rel_diff(gen, sym),
          'elimination' => Gates.rel_diff(eliminated, x),
          'inertia' => self.class.inertia(XL.eigvalsh(k), scale),
          'inv_vs_solve' => Gates.rel_diff(XL.inv(k, driver: 'sysv'), XL.solve(k, k.class.eye(k.shape[0]))),
          'ldl' => ldl(k, scale)
        }
        result['values'] = { 'x_norm' => Gates.fro(x), 'lambda_norm' => Gates.fro(lambda),
                             'objective' => Gates.fro(a.dot(x) - b) }
        result
      end

      def gate(result)
        tol = TOL.fetch(LinalgBench.dtype_name) { Gates.tol }
        expected = [result['n'], result['p'], 0]
        ldl = result['ldl']
        [
          Gates.le('constraint_residual', result['constraint'], tol),
          Gates.le('lagrange_residual', result['lagrange'], tol),
          Gates.le('solve_sym_vs_gen', result['sym_vs_gen'], tol),
          Gates.le('kkt_vs_null_space_elimination', result['elimination'], tol),
          Gates.eq('inertia_from_eigvalsh', result['inertia'], expected),
          Gates.le('inv_vs_solve_identity', result['inv_vs_solve'], tol),
          Gates.le('ldl_rebuilds_k', ldl['rebuild'], tol),
          Gates.le('ldl_factor_is_triangular_after_perm', ldl['off_triangle'], 0),
          Gates.eq('inertia_from_ldl_d', ldl['inertia'], expected)
        ]
      end

      def bench_step(fixture, variant = nil)
        k, rhs = self.class.system(fixture.device('A'), fixture.device('b'), fixture.device('B'), fixture.device('c'))
        case variant || VARIANTS.first
        when 'solve' then -> { XL.solve(k, rhs) }
        when 'eigvalsh' then -> { XL.eigvalsh(k) }
        when 'ldl' then -> { XL.ldl(k) }
        else raise Error, "unknown variant #{variant.inspect}; use #{VARIANTS.join(', ')}"
        end
      end

      private

      # x = x₀ + Z y with B x₀ = c and the columns of Z spanning the null space
      # of B; y is the least-squares solution of (A Z) y = b − A x₀.
      def eliminate(a, b, bc, c)
        x0 = XL.lstsq(bc, c)[0]
        z = XL.null_space(bc)
        y = XL.lstsq(a.dot(z), b - a.dot(x0))[0]
        x0 + z.dot(y)
      end

      def ldl(k, scale)
        u, d, perm = XL.ldl(k)
        permuted = u[LinalgBench.host(perm).to_a.map(&:to_i), true]
        {
          'rebuild' => Gates.rel_diff(u.dot(d).dot(u.transpose), k),
          'off_triangle' => Gates.max_abs(permuted.tril(-1)),
          'inertia' => self.class.inertia(XL.eigvalsh(d), scale)
        }
      end
    end
  end
end
