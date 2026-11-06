# frozen_string_literal: true

module LinalgBench
  # Gates that are recorded to fail on a backend (known_failures.json). A
  # problem whose only failing gates are recorded ones is a known failure,
  # not a failure. A recorded gate that passes is a failure too, so the
  # record cannot outlive the bug.
  module KnownFailures
    PATH = File.join(ROOT, 'known_failures.json')

    module_function

    def table
      @table ||= File.exist?(PATH) ? JSON.parse(File.read(PATH)) : {}
    end

    def known?(problem, gate, backend)
      table.dig(problem, gate, 'backends')&.include?(backend) || false
    end

    def classify(problem, gates, backend)
      known, unexpected = gates.reject(&:pass?).partition { |gate| known?(problem, gate.name, backend) }
      passed = gates.select { |gate| gate.pass? && known?(problem, gate.name, backend) }
      status = if unexpected.any? || passed.any? then 'fail'
               elsif known.any? then 'known_failure'
               else 'pass'
               end
      notes = { 'known_failures' => known.map(&:name), 'unexpected_passes' => passed.map(&:name) }
      [status, notes.reject { |_, names| names.empty? }]
    end
  end
end
