# frozen_string_literal: true

module LinalgBench
  module Problems
    # A. Restricted Hartree-Fock for H₂ and HeH⁺ in STO-3G, from integrals
    # PySCF computed once. The SCF energy has to reach PySCF's to 1e-6 Ha, and
    # the Roothaan equations F C = S C ε have to give the same orbital energies
    # whether solved as a generalized problem (eigh(F, S)) or reduced to a
    # standard one through S^(-1/2) from eigh(S) or through the Cholesky factor.
    class HartreeFock < Problem
      register 'hartree_fock'

      TOL = { 'fp64' => 1e-14 }.freeze
      ENERGY_TOL = 1e-6
      ITERATIONS = 50
      MOLECULES = %w[h2 heh].freeze
      VARIANTS = %w[eigh_general cholesky_solve].freeze

      # G(P) = J − ½ K with J_{μν} = Σ P_{λσ} (μν|λσ) and K_{μν} = Σ P_{λσ} (μλ|νσ).
      def self.two_electron(eri, p)
        n = p.shape[0]
        j = eri.reshape(n * n, n * n).dot(p.flatten).reshape(n, n)
        k = eri.transpose(0, 2, 1, 3).dup.reshape(n * n, n * n).dot(p.flatten).reshape(n, n)
        j - (0.5 * k)
      end

      def self.density(c, occupied)
        occ = c[true, 0...occupied]
        2.0 * occ.dot(occ.transpose)
      end

      def self.electronic_energy(h, f, p)
        0.5 * LinalgBench.scalar((p * (h + f)).sum)
      end

      def self.scf(h, s, eri, occupied)
        p = h.class.zeros(*h.shape)
        energies = []
        f = h
        eps = c = nil
        ITERATIONS.times do
          f = h + two_electron(eri, p)
          eps, c = XL.eigh(f, s)
          energies << electronic_energy(h, f, p)
          p = density(c, occupied)
        end
        { 'f' => h + two_electron(eri, p), 'c' => c, 'eps' => eps, 'p' => p, 'energies' => energies }
      end

      def run(fixture)
        molecules = MOLECULES.to_h { |name| [name, molecule(fixture, name)] }
        { 'molecules' => molecules,
          'values' => molecules.to_h { |name, m| ["#{name}_energy", m['energy']] } }
      end

      def gate(result)
        tol = TOL.fetch(LinalgBench.dtype_name) { Gates.tol }
        result['molecules'].flat_map do |name, m|
          [
            Gates.le("#{name}_energy_vs_pyscf", m['energy_error'], ENERGY_TOL),
            Gates.le("#{name}_scf_converged", m['last_change'], 1e-10),
            Gates.le("#{name}_scf_energy_never_rises", m['max_rise'], 1e-12),
            Gates.le("#{name}_eps_general_vs_symmetric", m['eps_symmetric'], tol),
            Gates.le("#{name}_eps_general_vs_cholesky", m['eps_cholesky'], tol),
            Gates.le("#{name}_eigvalsh_vs_eigh", m['eigvalsh'], tol),
            Gates.le("#{name}_c_s_orthonormal", m['s_orthonormal'], tol),
            Gates.le("#{name}_cho_solve_vs_solve", m['cho_solve'], tol),
            Gates.le("#{name}_inv_x_vs_solve", m['inv'], tol)
          ]
        end + [Gates.le('h2_energy_vs_szabo_ostlund', (result['molecules']['h2']['energy'] + 1.1167).abs, 5e-5)]
      end

      def bench_step(fixture, variant = nil)
        h = fixture.device('H')
        s = fixture.device('S')
        case variant || VARIANTS.first
        when 'eigh_general' then -> { XL.eigh(h, s, turbo: true) }
        when 'cholesky_solve'
          lambda do
            l = XL.cholesky(s, uplo: 'L')
            XL.solve(l, XL.solve(l, h).transpose)
          end
        else raise Error, "unknown variant #{variant.inspect}; use #{VARIANTS.join(' or ')}"
        end
      end

      private

      def molecule(fixture, name)
        h = fixture.device("#{name}_H")
        s = fixture.device("#{name}_S")
        eri = fixture.device("#{name}_eri")
        occupied = fixture.attr("#{name}_n_electrons") / 2
        out = self.class.scf(h, s, eri, occupied)
        f = out['f']
        energy = self.class.electronic_energy(h, f, out['p']) + fixture.attr("#{name}_e_nuc")
        eps, c = XL.eigh(f, s)
        w, u = XL.eigh(s)
        x = (u / XM::NMath.sqrt(w)).dot(u.transpose)
        l = XL.cholesky(s, uplo: 'L')
        x_chol = XL.inv(l).transpose
        n = s.shape[0]
        {
          'energy' => energy,
          'energy_error' => (energy - fixture.attr("#{name}_e_total")).abs,
          'last_change' => (out['energies'][-1] - out['energies'][-2]).abs,
          'max_rise' => out['energies'][1..].each_cons(2).map { |a, b| b - a }.max,
          'eps_symmetric' => Gates.max_rel_diff_sorted(XL.eigh(x.dot(f).dot(x))[0], eps),
          'eps_cholesky' => Gates.max_rel_diff_sorted(XL.eigh(x_chol.transpose.dot(f).dot(x_chol))[0], eps),
          'eigvalsh' => Gates.max_rel_diff_sorted(XL.eigvalsh(f, s), eps),
          's_orthonormal' => Gates.max_abs(c.transpose.dot(s).dot(c) - s.class.eye(n)),
          'cho_solve' => Gates.rel_diff(XL.cho_solve(l, f, uplo: 'L'), XL.solve(s, f)),
          'inv' => Gates.rel_diff(x_chol.transpose.dot(f), XL.solve(l, f))
        }
      end
    end
  end
end
