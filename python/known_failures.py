"""Gates that are recorded to fail on a backend, the Python side of
lib/linalg_bench/known_failures.rb. Same file, same rules."""

import json
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "known_failures.json")


def table():
    if not os.path.exists(PATH):
        return {}
    with open(PATH) as f:
        return json.load(f)


def known(problem, gate, backend):
    return backend in table().get(problem, {}).get(gate, {}).get("backends", [])


def classify(problem, gates, backend):
    failed = [g for g in gates if not g.passed()]
    expected = [g.name for g in failed if known(problem, g.name, backend)]
    unexpected = [g.name for g in failed if not known(problem, g.name, backend)]
    passed = [g.name for g in gates if g.passed() and known(problem, g.name, backend)]
    if unexpected or passed:
        status = "fail"
    elif expected:
        status = "known_failure"
    else:
        status = "pass"
    notes = {"known_failures": expected, "unexpected_passes": passed}
    return status, {k: v for k, v in notes.items() if v}
