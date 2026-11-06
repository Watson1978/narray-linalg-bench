# frozen_string_literal: true

module LinalgBench
  module Problems
    # B. Gaussian process regression with an RBF kernel. The fixture is a draw
    # from the GP with known (ell, sf, sn), so the marginal likelihood has to
    # peak next to those values on a grid around them.
    class GaussianProcess < Problem
      register 'gaussian_process'

      GRID_STEP = 1.5
      GRID_HALF = 3
      LOG_2PI = Math.log(2 * Math::PI)
      TOL = { 'fp64' => 5e-13 }.freeze

      def self.kernel(x1, x2, ell, sf)
        sq1 = (x1 * x1).sum(axis: 1)
        sq2 = (x2 * x2).sum(axis: 1)
        d2 = sq1.expand_dims(1) + sq2.expand_dims(0) - (2 * x1.dot(x2.transpose))
        (sf**2) * XM::NMath.exp(d2 * (-0.5 / (ell**2)))
      end

      def self.noisy_kernel(x, ell, sf, sn)
        kernel(x, x, ell, sf) + ((sn**2) * x.class.eye(x.shape[0]))
      end

      # −½ yᵀα − Σ log L_ii − (n/2) log 2π, left on the device.
      def self.log_marginal_likelihood(x, y, ell, sf, sn)
        l = XL.cholesky(noisy_kernel(x, ell, sf, sn), uplo: 'L')
        alpha = XL.cho_solve(l, y, uplo: 'L')
        (-0.5 * y.dot(alpha)) - XM::NMath.log(l.diagonal).sum - (0.5 * y.shape[0] * LOG_2PI)
      end

      def self.grid(center)
        (-GRID_HALF..GRID_HALF).map { |k| center * (GRID_STEP**k) }
      end

      def run(fixture)
        x = fixture.device('X')
        y = fixture.device('y')
        x_test = fixture.device('X_test')
        ell, sf, sn = %w[ell sf sn].map { |key| fixture.attr(key) }
        likelihood = scan(x, y, ell, sf, sn)
        prediction = predict(x, y, x_test, ell, sf, sn)
        { 'likelihood' => likelihood, 'paths' => paths(x, y, ell, sf, sn), 'prediction' => prediction, 'sn' => sn,
          'values' => { 'log_likelihood_peak' => likelihood['peak'], 'train_rmse' => prediction['train_rmse'],
                        'test_variance_min' => prediction['variance_min'] } }
      end

      def gate(result)
        tol = TOL.fetch(LinalgBench.dtype_name) { Gates.tol }
        lik = result['likelihood']
        paths = result['paths']
        pred = result['prediction']
        [
          Gates.le('likelihood_peak_grid_steps', lik['peak_offset'].map(&:abs).max, 1),
          Gates.le('alpha_solve_vs_cho_solve', paths['alpha_solve'], tol),
          Gates.le('alpha_inv_vs_cho_solve', paths['alpha_inv'], tol),
          Gates.le('cho_fact_vs_cholesky', paths['cho_fact'], tol),
          Gates.le('cho_inv_vs_inv', paths['cho_inv'], tol),
          Gates.le('logdet_cholesky_vs_slogdet', paths['logdet_slogdet'], tol),
          Gates.le('logdet_cholesky_vs_det', paths['logdet_det'], tol),
          Gates.eq('slogdet_sign', paths['slogdet_sign'], 1.0),
          Gates.le('train_rmse_over_sn', pred['train_rmse'] / result['sn'], 1.0),
          Gates.ge('test_variance_min', pred['variance_min'], 0.0),
          Gates.le('test_variance_max_over_sf2', pred['variance_max_over_sf2'], 1.0)
        ]
      end

      def bench_step(fixture, _variant = nil)
        x = fixture.device('X')
        y = fixture.device('y')
        ell, sf, sn = %w[ell sf sn].map { |key| fixture.attr(key) }
        -> { self.class.log_marginal_likelihood(x, y, ell, sf, sn) }
      end

      private

      # Reads back one scalar per grid point, which is fine for a gate.
      def scan(x, y, ell, sf, sn)
        best = nil
        grids = [self.class.grid(ell), self.class.grid(sf), self.class.grid(sn)]
        grids[0].each_with_index do |e, i|
          grids[1].each_with_index do |f, j|
            grids[2].each_with_index do |s, k|
              value = LinalgBench.scalar(self.class.log_marginal_likelihood(x, y, e, f, s))
              best = [value, [i, j, k]] if best.nil? || value > best[0]
            end
          end
        end
        { 'peak' => best[0], 'peak_offset' => best[1].map { |index| index - GRID_HALF } }
      end

      def paths(x, y, ell, sf, sn)
        k = self.class.noisy_kernel(x, ell, sf, sn)
        l = XL.cholesky(k, uplo: 'L')
        alpha = XL.cho_solve(l, y, uplo: 'L')
        logdet = 2 * LinalgBench.scalar(XM::NMath.log(l.diagonal).sum)
        sign, slog = XL.slogdet(k)
        inv = XL.inv(k)
        {
          'alpha_solve' => Gates.rel_diff(XL.solve(k, y), alpha),
          'alpha_inv' => Gates.rel_diff(inv.dot(y), alpha),
          'cho_fact' => Gates.rel_diff(XL.cho_fact(k, uplo: 'L').tril, l),
          'cho_inv' => Gates.rel_diff(XL.cho_inv(l, uplo: 'L').tril, inv.tril),
          'logdet_slogdet' => ((LinalgBench.scalar(slog) - logdet) / logdet).abs,
          'logdet_det' => ((Math.log(LinalgBench.scalar(XL.det(k))) - logdet) / logdet).abs,
          'slogdet_sign' => LinalgBench.scalar(sign).to_f
        }
      end

      def predict(x, y, x_test, ell, sf, sn)
        k = self.class.noisy_kernel(x, ell, sf, sn)
        l = XL.cholesky(k, uplo: 'L')
        alpha = XL.cho_solve(l, y, uplo: 'L')
        train_mean = self.class.kernel(x, x, ell, sf).dot(alpha)
        k_star = self.class.kernel(x, x_test, ell, sf)
        v = XL.cho_solve(l, k_star, uplo: 'L')
        variance = (sf**2) - (k_star * v).sum(axis: 0)
        { 'train_rmse' => Gates.fro(train_mean - y) / Math.sqrt(y.shape[0]),
          'variance_min' => LinalgBench.scalar(variance.min),
          'variance_max_over_sf2' => LinalgBench.scalar(variance.max) / (sf**2) }
      end
    end
  end
end
