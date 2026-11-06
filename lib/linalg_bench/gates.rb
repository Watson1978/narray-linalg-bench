# frozen_string_literal: true

module LinalgBench
  # Gate measures are built from elementwise operations and dot only, never
  # from the Linalg function under test.
  module Gates
    DEFAULT_TOL = { 'fp64' => 1e-10, 'fp32' => 1e-4 }.freeze

    Gate = Struct.new(:name, :value, :tol, :op) do
      def pass?
        case op
        when :le then value <= tol
        when :ge then value >= tol
        when :eq then value == tol
        else raise Error, "unknown gate op #{op.inspect}"
        end
      end

      def to_h
        { 'value' => value, 'tol' => tol, 'op' => op.to_s, 'pass' => pass? }
      end
    end

    module_function

    def tol(dtype = LinalgBench.dtype_name)
      DEFAULT_TOL.fetch(dtype)
    end

    def le(name, value, tol) = Gate.new(name.to_s, Float(value), Float(tol), :le)
    def ge(name, value, tol) = Gate.new(name.to_s, Float(value), Float(tol), :ge)
    def eq(name, value, expected) = Gate.new(name.to_s, value, expected, :eq)

    def fro(a)
      Math.sqrt(LinalgBench.scalar((a * a).sum))
    end

    def max_abs(a)
      LinalgBench.scalar(a.abs.max)
    end

    # ‖A x − b‖ / (‖A‖ ‖x‖ + ‖b‖), Frobenius throughout.
    def solve_residual(a, x, b)
      fro(a.dot(x) - b) / ((fro(a) * fro(x)) + fro(b))
    end

    def rel_diff(actual, expected)
      fro(actual - expected) / fro(expected)
    end

    # max |QᵀQ − I| over the columns of q.
    def orthogonality(q)
      k = q.shape[1]
      max_abs(q.transpose.dot(q) - q.class.eye(k))
    end

    # Eigenvalues and singular values come back in different orders from
    # different libraries; compare them as descending lists.
    def descending(values)
      LinalgBench.host(values).to_a.sort.reverse
    end

    def max_rel_diff_sorted(actual, expected)
      a = descending(actual)
      e = descending(expected)
      raise Error, "length differs: #{a.size} vs #{e.size}" if a.size != e.size

      scale = e.map(&:abs).max
      a.zip(e).map { |x, y| (x - y).abs }.max / scale
    end

    def report(gates)
      gates.to_h { |gate| [gate.name, gate.to_h] }
    end
  end
end
