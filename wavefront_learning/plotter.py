"""Backwards compatibility - import from plotting/ instead.

This module re-exports all functions from the plotting subpackage for
backwards compatibility. New code should import directly from plotting/.
"""

from plotting import *  # noqa: F401, F403
