#!/usr/bin/env python3
"""Deprecated wrapper for BumbleBox tracking parameter optimization."""

from __future__ import annotations

import warnings

from bumblebox_v2.tracking_optimizer import legacy_entrypoint


def main() -> int:
    warnings.warn(
        (
            "tracking-optimization.0.6.py is deprecated. "
            "Use 'python3 bbx.py optimize-tracking ...' instead."
        ),
        category=DeprecationWarning,
        stacklevel=2,
    )
    return legacy_entrypoint()


if __name__ == "__main__":
    raise SystemExit(main())
