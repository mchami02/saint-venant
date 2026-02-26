"""Utility functions for shock detection post-processing.

Provides connected component filtering to remove small isolated shock cells
from binary shock masks produced by the Lax entropy condition.
"""

import numpy as np


def filter_small_components(
    is_shock: np.ndarray, min_component_size: int = 5
) -> np.ndarray:
    """Remove small connected components from a binary shock mask.

    Uses scipy's ndimage.label to identify connected components in the
    2D boolean shock mask and removes any component with fewer than
    `min_component_size` cells.

    Args:
        is_shock: Boolean array of shape (nt, nx-1) indicating shock interfaces.
        min_component_size: Minimum number of cells for a component to be kept.
            If <= 0, filtering is disabled and the input is returned unchanged.

    Returns:
        Filtered boolean array of the same shape with small components removed.
    """
    if min_component_size <= 0:
        return is_shock

    from scipy.ndimage import label

    filtered = is_shock.copy()
    labeled, n_components = label(filtered)

    for comp_id in range(1, n_components + 1):
        component_mask = labeled == comp_id
        if component_mask.sum() < min_component_size:
            filtered[component_mask] = False

    return filtered
