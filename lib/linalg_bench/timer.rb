# frozen_string_literal: true

module LinalgBench
  # best-of-N with warm-up. Nothing synchronizes inside the timed region; the
  # device is drained once at its end, so every interval includes all the work
  # it queued.
  module Timer
    Result = Struct.new(:times, :inner) do
      def best = times.min
      def per_step = best / inner
    end

    module_function

    def measure(rounds: 3, warmup: 1, inner: 1, &step)
      raise Error, 'rounds must be at least 1' if rounds < 1

      warmup.times { inner.times(&step) }
      LinalgBench.synchronize
      times = Array.new(rounds) do
        started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        inner.times(&step)
        LinalgBench.synchronize
        Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
      end
      Result.new(times, inner)
    end
  end
end
