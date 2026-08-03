#!/usr/bin/env python3
"""Score a results.json against the PAVE baseline using the public metric.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import metric  # noqa: E402

REFERENCE = {"reward0": -156.1981, "smooth0": 0.340948}


def main() -> None:
    results_path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else results_path.as_posix()
    output = json.loads(results_path.read_text(encoding="utf-8"))
    print(f"=== {label} ===")
    print(metric.diagnostics(output, REFERENCE))
    m = metric.metrics(output, REFERENCE)
    print(f"reward: {m['reward']:.4f}")
    print(f"smoothness: {m['smoothness']:.6f}")
    print(f"score: {metric.score(output, REFERENCE):.4f}")


if __name__ == "__main__":
    main()
