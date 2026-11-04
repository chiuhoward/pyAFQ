"""Provider boundaries; registration mathematics is tested in utils/tests."""

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from AFQ.definitions.mapping import FireANTsMap, SynMap
from AFQ.registration import read_syn_mapping
from AFQ.utils import _fireants


def test_lazy_definition_without_backend():
    # A clean process catches eager imports even when another test used torch.
    code = """
import builtins
original_import = builtins.__import__
def no_backend(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'fireants'):
        raise AssertionError('Eager backend import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_backend
from AFQ.definitions.mapping import FireANTsMap
mapping = FireANTsMap()
assert mapping.device == 'cpu'
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


def test_backend_filenames_persistence_and_cache(tmp_path, monkeypatch):
    subject = nib.Nifti1Image(np.zeros((9, 11, 13)), np.diag([2, 3, 4, 1]))
    template = nib.Nifti1Image(np.zeros((15, 17, 19)), np.eye(4))
    affine_options = {"level_iters": [1]}
    backend_options = {"iterations": [1], "scales": [1]}
    definition = FireANTsMap(
        affine_kwargs=affine_options, fireants_kwargs=backend_options
    )
    affine_options["unused"] = True
    backend_options["unused"] = True
    assert "unused" not in definition.affine_kwargs
    assert "unused" not in definition.fireants_kwargs

    base = str(tmp_path / "sub-01")
    baseline_files = []
    for source, target in (("subject", "mni"), ("mni", "subject")):
        actual = definition.get_fnames(".nii.gz", base, source, target)
        inherited = SynMap().get_fnames(".nii.gz", base, source, target)
        for path, reference in zip(actual, inherited):
            assert path == reference.replace("_desc-mapping_", "_desc-fireants_")
            Path(reference).write_bytes(b"SynMap output must remain untouched")
            baseline_files.append(reference)

    prealign = np.eye(4)
    prealign[0, 3] = 2
    calls = []

    def affine(moving, fixed):
        assert moving is subject and fixed is template
        return prealign

    def compute(moving, fixed, matrix, device, options):
        assert moving is subject and fixed is template
        npt.assert_array_equal(matrix, prealign)
        assert device == "cpu" and options == {"iterations": [1], "scales": [1]}
        calls.append(True)
        forward = np.zeros(subject.shape + (3,), dtype=np.float32)
        backward = np.zeros(template.shape + (3,), dtype=np.float32)
        forward[..., 0] = 1
        backward[..., 0] = -1
        mapping = read_syn_mapping(
            nib.Nifti1Image(forward, subject.affine),
            nib.Nifti1Image(backward, template.affine),
        )
        return mapping, {"device": device, "torch_threads": 1}

    monkeypatch.setattr(definition, "prealign", affine)
    monkeypatch.setattr(_fireants, "_register_fireants", compute)
    args = (base, subject, "sub-01_dwi.nii.gz", subject, template, "mni")
    first = definition.get_for_subses(*args)
    assert len(calls) == 1

    def forbidden(*args, **kwargs):
        pytest.fail("A cache hit must not recompute prealignment or registration")

    monkeypatch.setattr(definition, "prealign", forbidden)
    monkeypatch.setattr(_fireants, "_register_fireants", forbidden)
    second = definition.get_for_subses(*args)
    npt.assert_array_equal(first.forward, second.forward)
    npt.assert_array_equal(first.backward, second.backward)
    npt.assert_allclose(
        second.transform_points(np.array([[3.0, 3.0, 3.0]])), [[4, 3, 3]]
    )
    for source, target, image in (
        ("subject", "mni", subject),
        ("mni", "subject", template),
    ):
        field_path, meta_path = definition.get_fnames(".nii.gz", base, source, target)
        npt.assert_allclose(nib.load(field_path).affine, image.affine)
        meta = json.loads(Path(meta_path).read_text())
        assert meta["device"] == "cpu" and meta["torch_threads"] == 1
        assert meta["fireants_kwargs"] == {"iterations": [1], "scales": [1]}
    for path in baseline_files:
        assert Path(path).read_bytes() == b"SynMap output must remain untouched"


def test_missing_backend_error_on_computation(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_backend(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "fireants"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backend)
    image = nib.Nifti1Image(np.zeros((9, 11, 13)), np.eye(4))
    with pytest.raises(ImportError, match="[Ii]nstall.*fireants"):
        FireANTsMap(use_prealign=False).get_for_subses(
            str(tmp_path / "sub-01"), image, "sub-01_dwi.nii.gz", image, image, "mni"
        )
