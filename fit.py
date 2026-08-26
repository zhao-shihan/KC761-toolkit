#!/usr/bin/env python3
"""Fit simulated spectrum/spectra to background-subtracted experimental data.

Thin wrapper around :mod:`kc761fit.cli`; see there for the full usage,
examples and options.  Equivalent to the installed ``kc761-fit`` entry point.
"""

import sys

from kc761fit.cli import main

if __name__ == "__main__":
    sys.exit(main())
