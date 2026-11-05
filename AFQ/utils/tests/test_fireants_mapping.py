"""Adapter tests with physical-coordinate oracles, without backend optimization."""

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from AFQ.utils._fireants import _mapping_from_displacement


def _points(shape):
    return np.moveaxis(np.indices(shape, dtype=float), 0, -1)


def _ramp(points):
    return points[..., 0] + 2 * points[..., 1] + 3 * points[..., 2]


@pytest.fixture
def grids():
    # Different shapes/spacing/origins and an oblique, flipped world basis.
    angle = np.pi / 6
    template_affine = np.array(
        [
            [-2 * np.cos(angle), -3 * np.sin(angle), 0, 23],
            [-2 * np.sin(angle), 3 * np.cos(angle), 0, -19],
            [0, 0, 4, 11],
            [0, 0, 0, 1],
        ]
    )
    relative = np.diag([0.7, 0.8, 0.9, 1])
    relative[:3, 3] = 4
    subject = nib.Nifti1Image(_ramp(_points((15, 17, 19))), template_affine @ relative)
    template = nib.Nifti1Image(_ramp(_points((25, 29, 33))), template_affine)
    return subject, template


@pytest.mark.parametrize("shift", [(0, 0, 0), (0.4, -0.3, 0.2)])
@pytest.mark.parametrize("same_grid", [False, True], ids=["unequal", "identical"])
def test_native_mapping_image_and_point_directions(grids, shift, same_grid):
    from dipy.io.stateful_tractogram import Space, StatefulTractogram

    from AFQ.utils.streamlines import move_streamlines

    subject, template = grids
    if same_grid:
        subject = template
    shift = np.asarray(shift)
    residual = np.empty(template.shape + (3,), dtype=np.float32)
    residual[:] = template.affine[:3, :3] @ shift
    mapping = _mapping_from_displacement(residual, subject, template)

    subject_points = _points(subject.shape)[3:-3, 3:-3, 3:-3].reshape(-1, 3)
    expected_template = (
        nib.affines.apply_affine(
            np.linalg.inv(template.affine) @ subject.affine, subject_points
        )
        - shift
    )
    npt.assert_allclose(
        mapping.transform_points(subject_points), expected_template, atol=1e-4
    )
    npt.assert_allclose(
        mapping.transform_points_inverse(expected_template), subject_points, atol=1e-4
    )

    # Independent voxel ramps check which image gets sampled on which grid.
    warped_template = mapping.transform(template.get_fdata())
    npt.assert_allclose(
        warped_template[3:-3, 3:-3, 3:-3].ravel(), _ramp(expected_template), atol=1e-4
    )
    template_points = _points(template.shape)
    expected_subject = nib.affines.apply_affine(
        np.linalg.inv(subject.affine) @ template.affine,
        template_points + shift,
    )
    valid = np.all(
        (expected_subject > 2) & (expected_subject < np.array(subject.shape) - 3),
        axis=-1,
    )
    warped_subject = mapping.transform_inverse(subject.get_fdata())
    npt.assert_allclose(
        warped_subject[valid], _ramp(expected_subject[valid]), atol=1e-4
    )
    assert mapping.forward.shape == subject.shape + (3,)
    assert mapping.backward.shape == template.shape + (3,)

    tractogram = StatefulTractogram([subject_points[:4]], subject, Space.VOX)
    moved = move_streamlines(
        tractogram, "template", mapping, template, to_space=Space.RASMM
    )
    npt.assert_allclose(
        moved.streamlines[0],
        nib.affines.apply_affine(template.affine, expected_template[:4]),
        atol=1e-4,
    )
    restored = move_streamlines(moved, "subject", mapping, subject, to_space=Space.VOX)
    npt.assert_allclose(restored.streamlines[0], subject_points[:4], atol=1e-4)


def test_nonlinear_inverse_and_prealignment_order(grids, tmp_path):
    from AFQ.registration import read_syn_mapping

    subject, template = grids

    # Two successive shears have an explicit inverse that is not -u.
    def forward(points):
        out = points.copy()
        out[..., 0] += 1.6 * np.sin(points[..., 1] / 5)
        out[..., 1] += 1.2 * np.sin(out[..., 0] / 5)
        return out

    def inverse(points):
        out = points.copy()
        out[..., 1] -= 1.2 * np.sin(points[..., 0] / 5)
        out[..., 0] -= 1.6 * np.sin(out[..., 1] / 5)
        return out

    voxel_prealign = np.array(
        [[1.05, 0.1, 0, -0.7], [0, 0.95, 0, 0.9], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    prealign = template.affine @ voxel_prealign @ np.linalg.inv(template.affine)
    template_points = _points(template.shape)
    residual = (
        (forward(template_points) - template_points) @ template.affine[:3, :3].T
    ).astype(np.float32)
    mapping = _mapping_from_displacement(residual, subject, template, prealign)

    subject_points = _points(subject.shape)[3:-3, 3:-3, 3:-3].reshape(-1, 3)
    prealigned = nib.affines.apply_affine(
        np.linalg.inv(template.affine) @ np.linalg.inv(prealign) @ subject.affine,
        subject_points,
    )
    expected_template = inverse(prealigned)
    # Every declared interior point is evaluated; no error-based exclusion.
    # Express both compositions in template voxels, as in the real-data check.
    mapped_template = mapping.transform_points(subject_points)
    mapped_subject = mapping.transform_points_inverse(expected_template)
    subject_to_template = np.linalg.inv(template.affine) @ subject.affine
    for actual, expected in (
        (mapped_template, expected_template),
        (
            nib.affines.apply_affine(subject_to_template, mapped_subject),
            nib.affines.apply_affine(subject_to_template, subject_points),
        ),
        (
            nib.affines.apply_affine(
                subject_to_template, mapping.transform_points_inverse(mapped_template)
            ),
            nib.affines.apply_affine(subject_to_template, subject_points),
        ),
        (mapping.transform_points(mapped_subject), expected_template),
    ):
        error_voxels = np.linalg.norm(actual - expected, axis=-1)
        assert np.percentile(error_voxels, 95) <= 0.1
        assert error_voxels.max() <= 0.5

    fields = []
    for name, field, image in (
        ("forward", mapping.forward, subject),
        ("backward", mapping.backward, template),
    ):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(field, image.affine), path)
        fields.append(str(path))
    restored = read_syn_mapping(*fields)
    for method, points in (
        ("transform_points", subject_points),
        ("transform_points_inverse", expected_template),
    ):
        npt.assert_allclose(
            getattr(restored, method)(points),
            getattr(mapping, method)(points),
            atol=1e-6,
        )
    for method, image in (("transform", template), ("transform_inverse", subject)):
        npt.assert_allclose(
            getattr(restored, method)(image.get_fdata()),
            getattr(mapping, method)(image.get_fdata()),
            atol=1e-6,
        )
