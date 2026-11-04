"""Analytic decoder tests: no FireANTs, torch, GPU, or subject data required."""

import numpy as np
import numpy.testing as npt
import pytest

from AFQ.utils._fireants import _normalized_grid_to_world_displacement


def _identity_coordinates(shape, dtype=np.float64):
    """Construct backend-order coordinates from independent 1D axis ramps."""
    nx, ny, nz = shape
    coordinates = np.empty((1, nz, ny, nx, 3), dtype=dtype)
    coordinates[0, ..., 0] = np.linspace(-1, 1, nx)[None, None, :]
    coordinates[0, ..., 1] = np.linspace(-1, 1, ny)[None, :, None]
    coordinates[0, ..., 2] = np.linspace(-1, 1, nz)[:, None, None]
    return coordinates


def _affine(linear, origin=(11, -12, 13)):
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = origin
    return affine


ANGLE = np.pi / 6
ROTATION = np.array(
    [
        [np.cos(ANGLE), -np.sin(ANGLE), 0],
        [np.sin(ANGLE), np.cos(ANGLE), 0],
        [0, 0, 1],
    ]
)
AFFINES = [
    np.eye(4),
    _affine(np.diag([2, 3, 4])),
    _affine(np.diag([-2, 3, -4])),
    _affine(ROTATION @ np.diag([2, 3, -4])),
]


@pytest.mark.parametrize(
    "affine", AFFINES, ids=["identity", "spacing", "flip", "oblique"]
)
def test_identity_coordinates(affine):
    shape = (5, 7, 9)
    displacement = _normalized_grid_to_world_displacement(
        _identity_coordinates(shape), shape, affine
    )

    npt.assert_allclose(displacement, 0, atol=1e-6)
    assert displacement.shape == shape + (3,)
    assert displacement.dtype == np.float32
    assert displacement.flags.c_contiguous


@pytest.mark.parametrize(
    "affine", AFFINES, ids=["identity", "spacing", "flip", "oblique"]
)
@pytest.mark.parametrize("axis", range(3), ids=["x", "y", "z"])
@pytest.mark.parametrize("direction", [-1, 1], ids=["negative", "positive"])
def test_known_axis_translations(affine, axis, direction):
    shape = (5, 7, 9)
    coordinates = _identity_coordinates(shape)
    shift_voxels = direction * 0.75
    coordinates[..., axis] += 2 * shift_voxels / (shape[axis] - 1)

    displacement = _normalized_grid_to_world_displacement(
        coordinates, shape, affine
    )

    # A voxel-axis step is the corresponding affine column in physical space.
    expected_vector = shift_voxels * affine[:3, axis]
    expected = np.broadcast_to(expected_vector, shape + (3,))
    npt.assert_allclose(displacement, expected, atol=1e-6)


def test_spatial_axes_and_components_at_distinct_locations():
    shape = (5, 7, 9)
    coordinates = _identity_coordinates(shape)
    # Backend indices are (z, y, x), but vector components are (x, y, z).
    coordinates[0, 3, 2, 1] = (1, 0.5, -0.5)
    coordinates[0, 7, 1, 3] = (0, -1, 1)
    affine = _affine([[2, 0.25, 0.5], [-0.1, 3, 0.75], [0.2, -0.5, 4]])

    displacement = _normalized_grid_to_world_displacement(
        coordinates, shape, affine
    )

    expected = np.zeros(shape + (3,))
    # Sampled voxel positions are (4, 4.5, 2) and (2, 0, 8), respectively.
    # Their voxel displacements are (3, 2.5, -1) and (-1, -1, 1).
    expected[1, 2, 3] = (6.125, 6.45, -4.65)
    expected[3, 1, 7] = (-1.75, -2.15, 4.3)
    npt.assert_allclose(displacement, expected, atol=1e-6)


def test_corner_samples_outside_normalized_range_are_not_clipped():
    shape = (5, 7, 9)
    coordinates = _identity_coordinates(shape)
    coordinates[0, 0, 0, 0] = (1.5, -1.5, 2)
    coordinates[0, -1, -1, -1] = (-2, 2, -1.5)

    displacement = _normalized_grid_to_world_displacement(
        coordinates, shape, np.eye(4)
    )

    expected = np.zeros(shape + (3,))
    expected[0, 0, 0] = (5, -1.5, 12)
    expected[-1, -1, -1] = (-6, 3, -10)
    npt.assert_allclose(displacement, expected, atol=1e-6)


def test_affine_origin_cancels_and_input_is_unchanged():
    shape = (5, 7, 9)
    coordinates = _identity_coordinates(shape)
    coordinates[..., 0] += 0.25
    original = coordinates.copy()
    affine = AFFINES[-1].copy()
    translated_affine = affine.copy()
    translated_affine[:3, 3] = (10000, -20000, 30000)

    displacement = _normalized_grid_to_world_displacement(
        coordinates, shape, affine
    )
    translated_displacement = _normalized_grid_to_world_displacement(
        coordinates, shape, translated_affine
    )

    npt.assert_array_equal(displacement, translated_displacement)
    npt.assert_array_equal(coordinates, original)


def test_float32_coordinates_at_realistic_image_size():
    shape = (65, 73, 81)
    displacement = _normalized_grid_to_world_displacement(
        _identity_coordinates(shape, dtype=np.float32), shape, AFFINES[-1]
    )

    # Normalized float32 coordinates introduce small voxel-rounding errors.
    npt.assert_allclose(displacement, 0, atol=2e-5)


def test_noncontiguous_coordinate_array():
    shape = (5, 7, 9)
    coordinates = _identity_coordinates(shape)[:, :, :, ::-1, :]
    assert not coordinates.flags.c_contiguous

    displacement = _normalized_grid_to_world_displacement(
        coordinates, shape, np.eye(4)
    )

    expected = np.zeros(shape + (3,))
    expected[..., 0] = (4 - 2 * np.arange(5))[:, None, None]
    npt.assert_allclose(displacement, expected, atol=1e-6)


@pytest.mark.parametrize(
    "coordinate_shape",
    [(9, 7, 5, 3), (2, 9, 7, 5, 3), (1, 9, 7, 5, 2), (1, 8, 7, 5, 3)],
)
def test_coordinate_shape_must_match_single_shared_grid(coordinate_shape):
    with pytest.raises(ValueError, match="coordinates"):
        _normalized_grid_to_world_displacement(
            np.zeros(coordinate_shape), (5, 7, 9), np.eye(4)
        )


@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 1), (5, 7, 9.5)])
def test_image_shape_requires_three_integer_axes_larger_than_one(shape):
    with pytest.raises(ValueError, match="image_shape"):
        _normalized_grid_to_world_displacement(
            _identity_coordinates((5, 7, 9)), shape, np.eye(4)
        )


def test_affine_shape():
    with pytest.raises(ValueError, match="affine"):
        _normalized_grid_to_world_displacement(
            _identity_coordinates((5, 7, 9)), (5, 7, 9), np.eye(3)
        )
