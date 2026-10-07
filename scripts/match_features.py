"""Hydra matching entry point using shared cached features."""

import sys

from run_experiment import main

if __name__ == "__main__":
    if not any(a.startswith("stage=") for a in sys.argv[1:]):
        sys.argv.append("stage=matches")
    main()
