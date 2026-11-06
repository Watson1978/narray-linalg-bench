# frozen_string_literal: true

require 'digest'

module LinalgBench
  # A fixture is two files under data/: <problem>_<size>.bin holds every array
  # as little-endian fp64 in row-major order, back to back, and
  # <problem>_<size>.json says where each one starts and what shape it has.
  # Ruby and Python both read the .bin and must see the same bytes.
  module Fixtures
    Fixture = Struct.new(:problem, :size, :arrays, :attrs, :manifest) do
      def [](name)
        arrays.fetch(name.to_s) { raise Error, "#{problem}_#{size} has no array #{name}" }
      end

      def attr(name)
        attrs.fetch(name.to_s) { raise Error, "#{problem}_#{size} has no attribute #{name}" }
      end

      def device(name, dtype = XF)
        LinalgBench.device(self[name], dtype)
      end
    end

    module_function

    def base_path(problem, size, dir: DATA_DIR)
      File.join(dir, "#{problem}_#{size}")
    end

    def exist?(problem, size, dir: DATA_DIR)
      base = base_path(problem, size, dir: dir)
      File.exist?("#{base}.bin") && File.exist?("#{base}.json")
    end

    def write(problem, size, arrays, attrs: {}, dir: DATA_DIR)
      raise Error, 'fixtures are written little-endian; this host is big-endian' if [1].pack('S') != [1].pack('S<')

      base = base_path(problem, size, dir: dir)
      offset = 0
      entries = []
      File.open("#{base}.bin", 'wb') do |f|
        arrays.each do |name, array|
          bytes = HM::DFloat.cast(array).to_binary
          f.write(bytes)
          entries << { 'name' => name.to_s, 'shape' => array.shape, 'offset' => offset,
                       'sha256' => Digest::SHA256.hexdigest(bytes) }
          offset += bytes.bytesize
        end
      end
      manifest = { 'problem' => problem.to_s, 'size' => size.to_s, 'dtype' => '<f8', 'order' => 'C',
                   'bytes' => offset, 'sha256' => Digest::SHA256.file("#{base}.bin").hexdigest,
                   'arrays' => entries, 'attrs' => attrs.transform_keys(&:to_s) }
      File.write("#{base}.json", "#{JSON.pretty_generate(manifest)}\n")
      manifest
    end

    def read(problem, size, dir: DATA_DIR)
      base = base_path(problem, size, dir: dir)
      raise Error, "#{base}.bin not found; run `rake fixtures`" unless exist?(problem, size, dir: dir)

      manifest = JSON.parse(File.read("#{base}.json"))
      arrays = manifest['arrays'].to_h do |entry|
        count = entry['shape'].inject(1, :*)
        bytes = File.binread("#{base}.bin", count * 8, entry['offset'])
        raise Error, "#{base}.bin is truncated at #{entry['name']}" if bytes.nil? || bytes.bytesize != count * 8

        [entry['name'], HM::DFloat.from_binary(bytes, entry['shape'])]
      end
      Fixture.new(problem.to_s, size.to_s, arrays, manifest['attrs'], manifest)
    end
  end
end
