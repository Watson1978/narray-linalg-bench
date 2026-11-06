# frozen_string_literal: true

module LinalgBench
  module Problems
    # D. Kabsch alignment of a point cloud moved by a known rotation, and PCA
    # of a higher-dimensional cloud by eigh of the covariance and by svd of
    # the data. The rotation must come back, a reflection must be caught by
    # the sign of det, and the two PCA routes must span the same subspace.
    class ProcrustesPca < Problem
      register 'procrustes_pca'

      TOL = { 'fp64' => 5e-14 }.freeze
      TOP_K = 3
      VARIANTS = %w[svd eigh].freeze

      # Returns R and t such that q_i ≈ R p_i + t, and the sign d that turned
      # a reflection into a proper rotation.
      def self.kabsch(p, q)
        p_mean = p.mean(axis: 0)
        q_mean = q.mean(axis: 0)
        h = (p - p_mean).transpose.dot(q - q_mean)
        _s, u, vt = XL.svd(h)
        v = vt.transpose
        d = LinalgBench.scalar(XL.det(v.dot(u.transpose))).negative? ? -1.0 : 1.0
        r = (v * p.class[1, 1, d]).dot(u.transpose)
        [r, q_mean - r.dot(p_mean), d]
      end

      def self.rmsd(p, q, r, t)
        diff = p.dot(r.transpose) + t - q
        Math.sqrt(LinalgBench.scalar((diff * diff).sum) / p.shape[0])
      end

      def self.centered(x)
        x - x.mean(axis: 0)
      end

      def self.projector(basis)
        basis.dot(basis.transpose)
      end

      def run(fixture)
        p = fixture.device('P')
        r0 = fixture.device('R0')
        noise = fixture.attr('noise')
        result = {
          'clean' => align(p, fixture.device('Q_clean'), r0),
          'noisy' => align(p, fixture.device('Q_noisy'), r0),
          'reflected' => align(p, fixture.device('Q_reflected'), r0),
          'pca' => pca(fixture.device('X')),
          'low' => low_rank(fixture.device('X_low')),
          'noise' => noise, 'rank' => fixture.attr('rank'), 'd' => fixture['X'].shape[1]
        }
        result['values'] = { 'rmsd_noisy' => result['noisy']['rmsd'], 'rmsd_reflected' => result['reflected']['rmsd'],
                             'top_variance' => result['pca']['top_variance'],
                             'nuclear_norm' => result['pca']['nuclear_norm'] }
        result
      end

      def gate(result)
        tol = TOL.fetch(LinalgBench.dtype_name) { Gates.tol }
        clean = result['clean']
        noisy = result['noisy']
        refl = result['reflected']
        pca = result['pca']
        low = result['low']
        expected_rmsd = result['noise'] * Math.sqrt(3)
        [
          Gates.le('clean_rmsd', clean['rmsd'], tol),
          Gates.le('clean_rotation_error', clean['rotation_error'], tol),
          Gates.le('clean_det_is_one', clean['det_error'], tol),
          Gates.eq('clean_sign', clean['sign'], 1.0),
          Gates.le('noisy_rmsd_vs_noise', (noisy['rmsd'] / expected_rmsd - 1).abs, 0.1),
          Gates.le('noisy_rotation_error', noisy['rotation_error'], 1e-3),
          Gates.eq('reflected_sign', refl['sign'], -1.0),
          Gates.le('reflected_det_is_one', refl['det_error'], tol),
          Gates.le('reflected_slogdet_sign', (refl['slogdet_sign'] - 1).abs, 0),
          Gates.ge('reflected_rmsd_is_large', refl['rmsd'] / expected_rmsd, 10.0),
          Gates.le('pca_subspace_eigh_vs_svd', pca['subspace'], tol),
          Gates.le('pca_variances_eigh_vs_svd', pca['variances'], tol),
          Gates.le('pca_eigvalsh_vs_eigh', pca['eigvalsh'], tol),
          Gates.le('pca_eigenvectors_orthogonality', pca['orthogonality'], tol),
          Gates.le('norm_fro_vs_svdvals', pca['norm_fro'], tol),
          Gates.le('norm_2_vs_svdvals', pca['norm_2'], tol),
          Gates.le('norm_nuc_vs_svdvals', pca['norm_nuc'], tol),
          Gates.eq('matrix_rank_full', pca['rank'], result['d']),
          Gates.eq('matrix_rank_low', low['rank'], result['rank']),
          Gates.eq('orth_columns_low', low['orth_columns'], result['rank']),
          Gates.le('orth_column_space_vs_eigh', low['column_space'], tol),
          Gates.le('orth_orthogonality', low['orthogonality'], tol)
        ]
      end

      def bench_step(fixture, variant = nil)
        x = fixture.device('X')
        case variant || VARIANTS.first
        when 'svd' then -> { XL.svd(x, job: 'S') }
        # turbo: picks syevd on Numo, which is what Cumo, numpy and CuPy run.
        when 'eigh' then -> { XL.eigh(x.transpose.dot(x), turbo: true) }
        else raise Error, "unknown variant #{variant.inspect}; use #{VARIANTS.join(' or ')}"
        end
      end

      private

      def align(p, q, r0)
        r, t, sign = self.class.kabsch(p, q)
        slog_sign, = XL.slogdet(r)
        {
          'rmsd' => self.class.rmsd(p, q, r, t),
          'rotation_error' => Gates.max_abs(r - r0),
          'det_error' => (LinalgBench.scalar(XL.det(r)) - 1).abs,
          'slogdet_sign' => LinalgBench.scalar(slog_sign).to_f,
          'sign' => sign
        }
      end

      def pca(x)
        k = TOP_K
        xc = self.class.centered(x)
        n, d = xc.shape
        cov = xc.transpose.dot(xc) / (n - 1)
        w, vecs = XL.eigh(cov)
        top_eigh = vecs[true, (d - k)...d].dup
        s, _u, vt = XL.svd(xc, job: 'S')
        top_svd = vt[0...k, true].transpose.dup
        variances = (s * s) / (n - 1)
        svals = XL.svdvals(xc)
        sv_host = Gates.descending(svals)
        {
          'subspace' => Gates.max_abs(self.class.projector(top_eigh) - self.class.projector(top_svd)),
          'variances' => Gates.max_rel_diff_sorted(w, variances),
          'eigvalsh' => Gates.max_rel_diff_sorted(XL.eigvalsh(cov), w),
          'orthogonality' => Gates.orthogonality(vecs),
          'norm_fro' => rel(XL.norm(xc, 'fro'), Math.sqrt(sv_host.sum { |v| v * v })),
          'norm_2' => rel(XL.norm(xc, 2), sv_host.first),
          'norm_nuc' => rel(XL.norm(xc, 'nuc'), sv_host.sum),
          'rank' => LinalgBench.scalar(XL.matrix_rank(xc)).to_i,
          'top_variance' => LinalgBench.scalar(w[-1]),
          'nuclear_norm' => sv_host.sum
        }
      end

      # The column space two ways: orth, and u_i = X v_i / sqrt(λ_i) from the
      # top eigenpairs of XᵀX. orth of the wide Xᵀ would be the row space, but
      # numo-linalg-alt 0.10.1 raises on a wide matrix there (docs/results).
      def low_rank(x)
        rank = LinalgBench.scalar(XL.matrix_rank(x)).to_i
        cols = XL.orth(x)
        d = x.shape[1]
        w, vecs = XL.eigh(x.transpose.dot(x))
        top = x.dot(vecs[true, (d - rank)...d]) / XM::NMath.sqrt(w[(d - rank)...d])
        {
          'rank' => rank,
          'orth_columns' => cols.shape[1],
          'column_space' => Gates.max_abs(self.class.projector(cols) - self.class.projector(top)),
          'orthogonality' => Gates.orthogonality(cols)
        }
      end

      def rel(actual, expected)
        ((LinalgBench.scalar(actual) - expected) / expected).abs
      end
    end
  end
end
