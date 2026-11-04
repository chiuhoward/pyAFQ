"""Private coordinate conversion for the optional FireANTs backend."""

import numpy as np


def _normalized_grid_to_world_displacement(coordinates, image_shape, affine):
    """Decode one shared-grid FireANTs sampling map into RAS-mm displacements.

    Parameters
    ----------
    coordinates : array-like, shape (1, Z, Y, X, 3)
        Full normalized moving-image sampling coordinates, with vector
        components (x, y, z) and ``align_corners=True``. Convert backend tensors
        to CPU NumPy arrays before calling. Raw warp parameters and halfway
        fields are not full sampling coordinates.
    image_shape : sequence of 3 ints
        Shared fixed/moving image shape in nibabel (X, Y, Z) order. Each axis
        must contain at least two voxels. Different grids or a downsampled
        coordinate map are not supported by this helper.
    affine : array-like, shape (4, 4)
        Shared image voxel-to-RAS-mm affine.

    Returns
    -------
    displacement : ndarray, shape (X, Y, Z, 3)
        Contiguous float32 RAS-mm displacement, sampled on the shared grid.
        Coordinates outside [-1, 1] are preserved. Affine translation cancels
        when subtracting positions; no additional LPS/RAS sign flip is used.

    Notes
    -----
    This returns the full residual sampling displacement, not a pyAFQ mapping
    or its inverse. It imports neither torch nor FireANTs and does not modify
    the input array.
    """
    image_shape = np.asarray(image_shape)
    if (
        image_shape.shape != (3,)
        or not np.issubdtype(image_shape.dtype, np.integer)
        or np.any(image_shape < 2)
    ):
        raise ValueError("image_shape must contain three integer sizes >= 2")

    coordinates = np.asarray(coordinates)
    expected_shape = (1, *image_shape[::-1], 3)
    if coordinates.shape != expected_shape:
        raise ValueError(f"coordinates must have shape {expected_shape}")
    affine = np.asarray(affine)
    if affine.shape != (4, 4):
        raise ValueError("affine must have shape (4, 4)")

    # Reorder spatial axes only; vector components remain (x, y, z).
    displacement = np.array(
        coordinates[0].transpose(2, 1, 0, 3), dtype=np.float64, copy=True
    )
    for axis, size in enumerate(image_shape):
        displacement[..., axis] = (displacement[..., axis] + 1) * (size - 1) / 2
        index_shape = [1, 1, 1]
        index_shape[axis] = size
        displacement[..., axis] -= np.arange(size).reshape(index_shape)

    # Transform vectors, not positions: the origin cancels in the difference.
    return np.ascontiguousarray(displacement @ affine[:3, :3].T, dtype=np.float32)
