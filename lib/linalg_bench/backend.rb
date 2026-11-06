# frozen_string_literal: true

require 'numo/narray'
require 'numo/linalg'
HM = Numo

module LinalgBench
  # Cumo is not in the Gemfile, so bundle install works without CUDA. GPU=1
  # loads the build in vendor/cumo-0.15.0 (or wherever CUMO_LIB points), which
  # also lets builds of other commits sit next to it.
  def self.add_cumo_load_path
    lib = ENV.fetch('CUMO_LIB') { Dir[File.expand_path('../../vendor/cumo-0.15.0/gems/cumo-*/lib', __dir__)].first }
    raise Error, 'no cumo build found; see AGENTS.md (cumo の版) or set CUMO_LIB' unless lib && File.directory?(lib)

    $LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
    lib
  end

  def self.cumo_so
    $LOADED_FEATURES.grep(%r{/cumo\.so\z}).first
  end
end

if ENV['GPU'].to_s =~ /\A(1|on|true)\z/i
  LinalgBench.add_cumo_load_path
  require 'cumo/narray'
  require 'cumo/linalg'
  XM = Cumo
else
  XM = Numo
end

XL = XM::Linalg

XF =
  case (dtype = ENV['DTYPE'].to_s.downcase)
  when '', 'fp32', 'sfloat' then XM::SFloat
  when 'fp64', 'dfloat' then XM::DFloat
  else raise LinalgBench::Error, "unknown DTYPE #{dtype.inspect}; use fp32 or fp64"
  end

module LinalgBench
  module_function

  def gpu?
    XM.name == 'Cumo'
  end

  def fp64?
    XF == XM::DFloat
  end

  def dtype_name
    fp64? ? 'fp64' : 'fp32'
  end

  def backend_name
    gpu? ? 'cumo' : 'numo'
  end

  # Cumo reductions return a 0-d NArray where Numo returns a Ruby Numeric.
  # Reading one back drains the queue, so never call this inside a hot loop.
  def scalar(value)
    return value if value.is_a?(Numeric)

    value.respond_to?(:extract_cpu) ? value.extract_cpu : value.extract
  end

  def synchronize
    XM::CUDA::Runtime.cudaDeviceSynchronize if gpu?
  end

  # Fixtures are always read as Numo::DFloat on the host. This is the one place
  # they cross to the backend and to the working dtype.
  def device(host, dtype = XF)
    array = gpu? ? XM::DFloat.from_binary(host.to_binary, host.shape) : host
    array.is_a?(dtype) ? array : dtype.cast(array)
  end

  def host(array)
    return array unless gpu?

    HM::DFloat.from_binary(XM::DFloat.cast(array).to_binary, array.shape)
  end
end
