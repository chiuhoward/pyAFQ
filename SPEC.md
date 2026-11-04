# FireANTs registration backend for pyAFQ

**Status:** Integration planned; coordinate decoder merged; end-to-end correctness and performance unmeasured

**Owner:** Howard Chiu · **Updated:** 2026-11-04

**Target repository:** [tractometry/pyAFQ](https://github.com/tractometry/pyAFQ)

## 1. Question and pitch

**Integrate FireANTs as an optional pyAFQ registration backend with minimal additional code, correct image and streamline transforms, and a measured runtime benefit.** Integration is the project objective; validation establishes whether the implementation meets those acceptance criteria.

Implement an optional `FireANTsMap`, preserving the existing `SynMap` default. CPU execution is required for accessibility and debugging. Measure complete usable-map time against `SynMap`; use GPU execution to demonstrate the intended acceleration when CPU offers no benefit. A working integration without a measured speedup is partial completion, not evidence of faster registration.

## 2. Two tiers and working rules

**Must-have for v1:** the §6 image/point/serialization contract; focused tests of new adapter behavior under §10; a minimal Python mapping definition; backend-specific filenames using pyAFQ's existing persistence pattern; lazy backend imports; CPU execution; and the one-subject correctness and performance validation in §11. Leave the default mapping and shared `SynMap` behavior unchanged.

**Follow-up:** config/CLI integration, package extra and release compatibility matrix, extended device/thread APIs, RNG scoping, detailed broken-import diagnostics, general geometry policy (including shear rejection), custom cache invalidation/integrity machinery, broad CI coverage, and comprehensive docs. These are not v1 acceptance conditions. Small changes needed to package an eventual first PR can be agreed with maintainers when preparing submission.

Record exact versions, settings, results, and unresolved issues in `NOTES_fireants.md`. Develop implementation and focused tests together; use one Stanford HARDI harness for integration checks, anatomical validation, and timing. Correctness failures block accepting the affected behavior; a slow CPU result guides optimization/GPU work rather than blocking integration. Discuss evidence requiring changes to §6 or existing public behavior. Do not spend the hackathon building follow-up features. Preparing a branch/report does not authorize publishing an upstream PR or contacting maintainers.

Choose the smallest existing FireANTs/DIPY/pyAFQ path that satisfies the consumer contract before adding helpers. The decoder merged in PR #6 is available implementation, not a mandatory architecture: retain it if used by that path, otherwise remove or replace it together with its obsolete tests. Preserve the externally required image/point behavior, not a particular intermediate representation.

## 3. Source baseline and evidence

The design was checked against these sources on 2026-10-26:

| Component | Inspected revision | Meaning |
|---|---|---|
| pyAFQ | [`65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8`](https://github.com/tractometry/pyAFQ/tree/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8) | Implementation target for this document |
| DIPY | [`407e9319cbfe3d05304682d317363ecd09a83fdc`](https://github.com/dipy/dipy/tree/407e9319cbfe3d05304682d317363ecd09a83fdc), tag `1.12.0` | Target pyAFQ requires `dipy>=1.12.0,<1.13.0`; do not design against DIPY development HEAD |
| FireANTs | [`f8a9cc259ab9113fc43bd99797000fa8fc34eb59`](https://github.com/rohitrango/FireANTs/tree/f8a9cc259ab9113fc43bd99797000fa8fc34eb59) | Source metadata says `1.5.1`; equivalence to a published wheel has not been established |

Important source findings:

1. [`GeneratedMapMixin`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/definitions/mapping.py) supplies `get_fnames` and affine estimation, not automatic nonlinear computation/caching. Its filenames do not identify the backend. `SynMap.get_for_subses` owns its own computation and persistence; it has no `gen_mapping` hook.
2. `SynMap` simplifies its transform with [`AFQ._fixes.get_simplified_transform`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/_fixes.py), writes separate subject-grid and template-grid fields, and reloads them through [`read_syn_mapping`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/registration.py). The returned map operates in voxel coordinates, with its affines and prealignment set to identity/`None`.
3. [`move_streamlines`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/utils/streamlines.py) converts streamlines to voxel coordinates, then calls `transform_points` or `transform_points_inverse`. Returning an arbitrary world-space `DiffeomorphicMap` would not meet this contract.
4. The mapping task uses `mapping_definition`, not the illustrative `mapping=` argument found in some mapping docstrings. See [`AFQ/tasks/mapping.py`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/tasks/mapping.py).
5. Config parsing imports mapping classes using `__all__` in [`AFQ/utils/bin.py`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/utils/bin.py). [`Definition.str_formal`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/definitions/utils.py) inspects constructor variable names, including local variables, and expects corresponding attributes. Account for that behavior if config support is added; config integration is follow-up work.
6. FireANTs [`Image`](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/io/image.py) defaults to CUDA. Pass an explicit device at image creation. Its metadata uses `(size-1)/2`, consistent with `align_corners=True`.
7. [`AbstractRegistration.get_warped_coordinates`](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/registration/abstract.py) composes the normalized affine and displacement into a full sampling grid. Use that method; `get_warp_parameters()['grid']` alone is not the full transform.
8. [`SyNRegistration.get_inverse_warp_parameters`](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/registration/syn.py) raises `NotImplementedError`. Symmetric optimization does not provide an exposed full inverse. `get_warp_parameters` numerically inverts the reverse halfway warp internally; extraction can be expensive.
9. SyN scales use `max(int(size/scale), MIN_IMG_SIZE)` with `MIN_IMG_SIZE=32`. This is clamping, not simply a rejection of every coarse grid smaller than 32. The inverse path also has size assumptions. Test noncubic shapes and avoid custom output shapes in v1.
10. FireANTs [`pyproject.toml`](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/pyproject.toml) already requires torch and SimpleITK. Its [`interpolator`](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/interpolator/__init__.py) has PyTorch fallbacks for CPU. CPU operation still needs a runtime smoke test.
11. Target pyAFQ declares extras in [`setup.cfg`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/setup.cfg). It supports Python `>=3.11,<3.14`.
12. [`GroupAFQ.combine_bundle`](https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/api/group.py) explicitly restricts mappings to `SynMap`. Do not claim universal support wherever `SynMap` is accepted. That separate restriction remains outside v1.

Before implementation, check for applicable repository instructions and record the actual checkout. If working against another revision, reconcile these findings first. A package version string alone does not establish source equivalence.

<a id="4-early-feasibility-gate-m0"></a>

## 4. Shared real-data harness: M0

Use one small harness on **Stanford HARDI at its actual registration/template resolution**, using the same contrast/template and affine settings as `SynMap`. It should call the adapter implementation as it becomes available, rather than duplicate loading, extraction, or inversion code. Begin data/environment setup and the reference run alongside implementation. M0 is a shared validation task, not permission to start M1/M2. No separate synthetic optimizer project or standalone go/no-go experiment is required.

1. Record hardware, thread allocation, exact versions, image shapes, settings, and a predeclared runtime budget. Set the same job-script thread limits below for both backends before Python starts. Run the `SynMap` reference without a registration cache.
2. Run the focused analytic tests for the conversion path actually used. Reuse the merged decoder checks if that helper is retained; do not rebuild upstream coordinate or registration suites.
3. Run FireANTs SyN on CPU. Measure affine/resampling, optimization, full coordinate extraction (which includes internal inversion), full residual inversion, native-map conversion, and total usable bidirectional registration time. A short optimization run may screen failures, but cannot establish full-settings speed or quality; label it accordingly.
4. Before judging registration quality, perform the **§7.3 comparison against FireANTs' own `evaluate` output**, using the same prealigned moving image, output grid, and interpolation/padding. Record the comparison error on a predefined valid interior. If it fails, diagnose extraction/decoding/resampling first; a bad adapter overlay is not evidence against FireANTs. Inspect the native `evaluate` image separately: if native alignment is poor too, investigate registration/input/settings rather than decoding.
5. Once those checks pass, inspect forward alignment overlays, inverse overlays, and bidirectional point-composition errors on a predefined valid interior/brain mask. Report coverage and error in both mm and voxel units. Do not evaluate only the small region that happens to pass.
6. Use the provisional inverse acceptance limits: 95th-percentile composition error ≤0.1 template voxel and maximum ≤0.5 template voxel, evaluating both compositions in template voxel coordinates. Report corresponding mm errors and valid coverage. These are engineering targets to review against interpolation evidence, not anatomical accuracy claims.
7. Record correctness and performance separately. Fix decoding, inverse, or geometry failures before calling that path correct; investigate settings if FireANTs' native result is poor. A slow CPU result directs performance work and GPU measurement. It does not cancel the integration task. Implement the §7.4 fallback only if the selected inverse fails its accuracy/cost requirements.

Use this allocation/thread setup in the pilot's SLURM job script, before importing numerical libraries, for **both** the `SynMap` and FireANTs runs:

```bash
#!/usr/bin/env bash
#SBATCH --cpus-per-task=4

: "${SLURM_CPUS_PER_TASK:?Submit with an explicit CPU allocation}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Launch each backend's pilot in a fresh Python process here.
# Run them sequentially with the same allocation and these exports.
```

For a local pilot, set both variables to the same explicit CPU budget instead. Record the environment and FireANTs' observed `torch.get_num_threads()`; check that it matches the intended budget before timing. These variables do not control every possible numerical runtime, so record any backend-specific overrides. Run one subject at a time during timing to avoid multiplying this budget by subject workers. This is pilot setup, not a new thread-control API in `FireANTsMap`.

Compare the same timing boundary for both methods: either include the same affine work in each total or report a shared affine stage separately. Similar-looking iteration counts are not equivalent settings across optimizers; assess runtime alongside alignment quality, not in isolation.

Set a practical run budget before launching; a timeout is an incomplete measurement, not a speed estimate. Report the stage and elapsed time. Slower CPU performance does not require abandoning implementation or matching GPU performance on CPU. Demonstrating acceleration still requires a comparable complete-map measurement at useful registration quality.

M0 deliverable: a runnable harness/job script, settings, and a short timing/error table with overlays as the adapter becomes runnable. Report passed, failed, and unrun checks and remaining work. Reuse these results in §11; no separate go decision is needed to continue implementation.

## 5. Minimal Python integration

```python
class FireANTsMap(GeneratedMapMixin, Definition):
    def __init__(
        self,
        use_prealign=True,
        affine_kwargs=None,
        fireants_kwargs=None,
        device="cpu",
    ): ...
```

Use CPU as the explicit v1 default. Support an explicit CUDA device for the planned GPU experiment. Automatic device selection can follow later. Copy caller dictionaries; import torch/FireANTs only when computation is needed. A missing backend should produce a straightforward install message; sophisticated diagnostics are follow-up. Use an exact tested development environment until a package extra is agreed.

Mirror the target `get_for_subses` signature and implement computation/cache handling there or in private helpers; there is no inherited nonlinear `gen_mapping` hook. Do not refactor neighboring mapping classes.

```python
from AFQ.api.group import GroupAFQ
from AFQ.definitions.mapping import FireANTsMap

afq = GroupAFQ(
    bids_path="/path/to/bids",
    mapping_definition=FireANTsMap(device="cpu"),
)
```

Add the new definition to mapping `__all__`. V1 uses the Python API; config support is follow-up. Registration images must be scalar 3D images on the grids expected downstream. Confirm the subject registration grid matches the DWI grid and that loading preserves image geometry. Do not add a general shear detector or resampling framework for v1. If the actual loader changes geometry in the pilot, stop and resolve that concrete failure rather than silently returning a misregistered map.

## 6. Mathematical contract: images, points, and fields

This section is authoritative. Names such as “forward” are not sufficient evidence of direction.

Use voxel-center indices in nibabel `(i,j,k)` order. Let:

- `A_S`, `A_T`: subject and template voxel-to-RAS-mm affines.
- `P`: the affine **pull** transform returned by `affine_registration(subject, template)`: template RAS → subject RAS. Confirm using an analytic affine fixture.
- `S`, `T`: subject and template image arrays.
- `S_pre(x) = S(A_S^-1 P x)`, where `x` is a template-space RAS position.
- `F(x) = x + u(x)`: FireANTs' full template → prealigned-subject sampling map, expressed in template RAS mm. Both images now have the template grid and affine.
- `G = F^-1`: validated inverse of the full residual map on that same grid; see §7.4 for two candidate construction methods.

The complete physical maps are:

```text
template → subject: H_TS(x) = P F(x)
subject → template: H_ST(y) = G(P^-1 y)
```

The nonlinear transform and affine generally do not commute. Do not reverse their order, negate a field to approximate an inverse, or treat SyN's two halfway fields as a full forward/inverse pair.

The returned pyAFQ map must satisfy:

| Operation | Input | Output |
|---|---|---|
| `mapping.transform(T)` | Template image | Subject-grid image, sampling `T` using subject → template coordinates |
| `mapping.transform_inverse(S)` | Subject image | Template-grid image, sampling `S` using template → subject coordinates |
| `mapping.transform_points(subject_vox)` | Subject voxel points | Template voxel points |
| `mapping.transform_points_inverse(template_vox)` | Template voxel points | Subject voxel points |

Build the native-grid coordinate maps:

```text
C_ST(i) = A_T^-1 G(P^-1 A_S i)   # i ranges over subject voxels
C_TS(j) = A_S^-1 P F(A_T j)      # j ranges over template voxels

forward[i]  = C_ST(i) - i       # shape subject_shape + (3,)
backward[j] = C_TS(j) - j       # shape template_shape + (3,)
```

These are **voxel-coordinate displacements**, incorporating both image geometries and prealignment. They are not world-mm residuals. Their component values can be nonzero even when the anatomical physical map is identity, if the two voxel grids differ.

The recommended construction mirrors `read_syn_mapping`:

```python
mapping = DiffeomorphicMap(
    dim=3,
    disp_shape=template_shape,
    disp_grid2world=None,
    domain_shape=subject_shape,
    domain_grid2world=None,
    codomain_shape=template_shape,
    codomain_grid2world=None,
    prealign=None,
)
mapping.forward = forward.astype(np.float32)
mapping.backward = backward.astype(np.float32)
```

Mirror the target `read_syn_mapping` construction and inherit its constructor defaults. The inspected DIPY 1.12.0 attribute is **`is_inverse`**, not `is_inverted`; no manual flag assignment is needed here. Recheck the reader and DIPY constructor if the target version changes.

This follows pyAFQ's simplified-map convention, including separately sampled forward/backward fields. Test the specific methods pyAFQ uses; do not assume every DIPY diagnostic designed for a common displacement grid applies unchanged. Never simplify this already simplified map a second time. An alternative intermediate world-space map is acceptable only if it demonstrably produces the same native-grid result and serialization.

## 7. Minimal computation pipeline

### 7.1 Prealignment

Reuse `GeneratedMapMixin.prealign` when enabled; otherwise use identity. It estimates the template-RAS → subject-RAS pull affine `P`; confirm direction with an analytic fixture. Use DIPY `AffineMap` with template domain and subject codomain to sample the subject on the template grid. The helper has no independent prealignment cache.

### 7.2 FireANTs registration

Write `T` and the prealigned subject as temporary float32 NIfTI images with template affine `A_T`. Load with an explicit device through `Image.load_file`, without orientation/center overrides, then wrap in `BatchedImages`. Use a managed temporary directory. Do not pass `P` as FireANTs' initial affine as well; it has already been applied.

Provisional starting configuration: `scales=[4,2,1]`, `iterations=[100,50,20]`, `optimizer="Adam"`, `optimizer_lr=0.2`, `deformation_type="compositive"`, `loss_type="cc"`, float32. Scale/iteration/optimizer values follow the inspected example embedded in `syn.py`; remaining defaults follow the pinned constructor. Resolve and record settings once. Do not call them validated until the real-data pilot passes. Thread allocation is controlled by the pilot's launch environment and recorded; an in-process thread management API is follow-up.

Use full-resolution extraction, no custom output shape, and dimensions ≥32 for the initial supported experiment. FireANTs clamps coarse scale dimensions to at least 32; record effective shapes if diagnosing a scale failure. Do not implement a general scale-selection framework.

### 7.3 Full sampling coordinates

Call `reg.get_warped_coordinates(fixed_batch, moving_batch)` once after optimization. Do not use raw `get_warp_parameters()['grid']` as the complete transform. Detach and copy to CPU.

The inspected coordinate tensor has shape `[B,Z,Y,X,3]`, with `(x,y,z)` components. Decode each normalized component for its corresponding moving-image size `N`:

```text
voxel_coordinate = (normalized_coordinate + 1) * (N - 1) / 2
```

Transpose spatial axes to nibabel `(X,Y,Z,3)` without reversing the component axis. Subtract the identity voxel grid and multiply vectors by `A_T[:3,:3]` to obtain the template-grid RAS-mm residual `u`. Both images share the same grid. No additional LPS/RAS sign flip is applied under this design; analytic and actual-loader checks must establish that correspondence. Do not clip coordinates to `[-1,1]`.

Check extracted-grid sampling against FireANTs `evaluate` on valid interior points with matching padding/interpolation. Perform this in M0 before using adapter overlays to judge the backend. Record the error and a justified numerical tolerance; systematic axis/sign discrepancies must not be dismissed as interpolation differences. This is a focused correctness check, not a new resampling subsystem.

### 7.4 Inverse: primary method and fallback

**Primary:** numerically invert the full composed residual `u` on the template grid using DIPY 1.12.0 [`invert_vector_field_fixed_point_3d`](https://github.com/dipy/dipy/blob/407e9319cbfe3d05304682d317363ecd09a83fdc/dipy/align/vector_fields.pyx). Use contiguous float32 fields, float64 world-to-grid/spacing arrays, and explicit iteration/tolerance settings. Start provisionally at 100 iterations and tolerance `1e-3`; acceptance uses measured composition error. This pinned function has no `num_threads` argument.

**Fallback if accuracy or cost is unacceptable:** derive the opposite full transform from FireANTs' two halfway maps by composing them in reverse order with the appropriate inverse. For maps `Q` and `R`, if the extracted residual map is exactly `F = Q ∘ R^-1`, then `G = R ∘ Q^-1`. The inspected `syn.py` and [`compose_warp`](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/utils/util.py) provide the starting point for checking that order. Match all actual smoothing, interpolation, coordinate frames, and resolution handling; the raw tensors alone are not a proven inverse pair. An existing inverse of `R` used to extract `F` does not eliminate the need to obtain `Q^-1` for `G`.

Treat this as a candidate consistent with symmetric-registration construction, not a guaranteed better or free inverse. Implement only the chosen method; investigate the fallback if necessary. Derive against the pinned source, test on analytic fixtures, and compare error and total cost on the same real subject. Use the same error/coverage limits whichever method is selected. An inverse that fails leaves bidirectional integration incomplete; never substitute identity. Record the chosen method and any private-API reliance.

Evaluate both composition directions and count invalid/out-of-support points. For residual sampling outside the template support, use zero residual extension (affine-only extrapolation); image sampling outside FOV uses zero background. Exclude invalid compositions from error summaries while reporting their number/coverage. Central anatomical coverage failure blocks validation.

## 8. Lazy imports and runtime scope

The mapping module and class construction must not introduce eager backend imports. Keep imports inside computation; retain CPU operation without CUDA or compiled FireANTs extensions. Fail clearly for an unavailable explicitly requested GPU rather than silently changing the experiment. Existing `SynMap` behavior stays unchanged.

V1 does not expose seed/thread management or guarantee bitwise reproducibility. Record the pilot environment and reuse upstream tractography/scalars to control the main comparison. Scoped RNG state, thread restoration, extended device policies, detailed broken-install diagnostics, and exhaustive exception tests belong in follow-up work.

## 9. Cache: follow pyAFQ's existing pattern

Use backend-specific filenames such as `desc-fireants` while retaining the existing from/to-space naming convention. Save forward/backward native-grid fields as NIfTI with subject/template reference affines, respectively, and reload through `AFQ.registration.read_syn_mapping`. Return the reloaded representation on first run too. A matching pair of existing files is reused using the same basic existence-check pattern as `SynMap`.

### Dataset placement and BIDS regression check

**V1 requirement: introduce no BIDS validation regressions relative to existing pyAFQ outputs. Full BIDS compliance of pyAFQ's existing private mapping convention is not a requirement of this feature.** Keep the existing suffix, field representation, directory layout, and sidecar convention; change the description to `desc-fireants` for backend separation. These are private pyAFQ caches, not standardized spatial-transform exports.

Keep outputs in a derivative pipeline directory, for example:

```text
derivatives/afq/sub-01/ses-01/dwi/
  sub-01_ses-01_desc-fireants_from-subject_to-mni_xform.nii.gz
  sub-01_ses-01_desc-fireants_from-subject_to-mni_xform.json
  sub-01_ses-01_desc-fireants_from-mni_to-subject_xform.nii.gz
  sub-01_ses-01_desc-fireants_from-mni_to-subject_xform.json
```

The labels `subject`/`mni` follow pyAFQ's convention; record the actual reference images/template in metadata. Preserve relevant source entities such as acquisition/run to avoid collisions, and use pyAFQ's existing derivative placement, never raw `sub-*/dwi` paths. Do not add ignore rules, relocate the outputs, or change shared dataset metadata solely to eliminate preexisting pyAFQ validation issues. Record those issues separately as upstream/follow-up concerns.

Compare matched `SynMap` and `FireANTsMap` output fixtures with the same validator, schema, validation scope, metadata, and existing ignore configuration. Normalize only the expected backend description token in reported paths. Existing `NOT_INCLUDED` findings caused by the shared `xform` convention are acceptable when reproduced in the baseline; new malformed labels, lost source entities, misplaced files, invalid sidecars, or other feature-specific findings are regressions to fix. Do not add exclusions to make this comparison pass. If both backends' caches coexist, account for the expected extra files with the same known findings rather than requiring equal absolute counts.

**Measured filename check:** Validator 3.0.2 with BIDS 1.11.2 on matched synthetic fixtures, without exclusions, reports zero raw-root errors and four derivative `NOT_INCLUDED` errors for each backend. All issue signatures match after normalizing only `desc-mapping` versus `desc-fireants`. This supports the proposed naming change, but does not replace checking the actual generated files during implementation. See `BIDS_FILENAME_REVIEW.md` for evidence and the earlier exclusion experiments; those workarounds are not v1 requirements.

Record settings, affine, versions, inverse method, timings, **device and thread allocation as provenance** in JSON sidecars. Changing a SLURM allocation or moving between CPU and GPU must not itself invalidate the cache. There is no device/thread cache key. Use separate fresh output directories for intentional CPU/GPU comparisons so the GPU experiment actually computes a registration.

V1 does not introduce content hashes, completion manifests, checksums, custom locking, or atomic pair commits. Document the existing limitation: cached results can become stale when inputs or algorithm settings change. Use fresh output directories or the existing pyAFQ invalidation workflow for those experiments, and remove incomplete pairs before retrying. Test backend separation and ordinary save/reload; do not build a stronger cache framework for this one backend. Any richer invalidation should be discussed as shared pyAFQ infrastructure later.

## 10. Required correctness tests

Reuse existing pyAFQ regressions and DIPY/FireANTs algorithm tests. Add tests that invoke the new adapter or provider boundary. A related upstream test that never calls our code cannot detect a regression in that boundary. Use the coverage audit to identify existing assertions before adding tests.

The following are behavior obligations, not six separate test suites or an exhaustive cross-product. Combine fixtures where one independent oracle covers several obligations. Remove or replace a test when retained coverage protects the same behavior through our implementation. Do not retain tests for an intermediate helper that the integration no longer uses.

### A. Adapter mathematics, without FireANTs optimization

Use independent analytic coordinate maps as the oracle, rather than deriving expected results with the helper under test. Cover the following through the selected implementation:

1. Physical identity with identical **and different** voxel grids; test image and point directions and unequal shapes.
2. Positive/negative translations on each axis, anisotropic spacing, nonzero origins, axis flips, and an oblique orthogonal affine; check physical sign and magnitude.
3. Nonidentity affine prealignment plus a smooth invertible spatially varying residual; distinguish `P F` from `F P` and inversion from field negation.
4. If the implementation decodes normalized grids, cover noncubic dimensions, corners and out-of-range coordinates; distinguish spatial-axis order from component order. Reuse the existing decoder tests when that helper is retained.
5. Bidirectional composition and actual `move_streamlines` on an artificial tractogram; verify destination voxel coordinates and world positions.
6. Image/point save-reload equivalence on unequal grids, field shapes, and reference affines. Array-returning transforms do not return affines themselves.

For well-conditioned exact affine/translation fixtures, target maximum point error ≤`1e-4` destination voxel. For interpolated nonlinear compositions, target 95th-percentile error ≤`0.1` voxel and maximum ≤`0.5` voxel on predefined valid interior points. Report mm errors and coverage. Explain evidence before changing tolerances; do not weaken them just to pass a failing adapter.

### B. Minimal integration checks

Check lazy import/construction without FireANTs, an informative error when absent on computation, ordinary cache reuse, and separation from `SynMap` files. Check actual output placement/names for validation regressions against the existing pyAFQ baseline under §9, using the same validator configuration. Only actual backend tests use `pytest.importorskip`; analytic/import checks must still run without FireANTs. One real CPU run exercises optimization, full-grid extraction, inverse, map construction, and reload. If CUDA is unavailable, report “not tested,” not a passing GPU test.

Additional synthetic optimizer suites, exhaustive import/cache/device mocking, config round trips, RNG restoration, and CI matrices are follow-up. They must not delay the working integration or real-data validation.

## 11. One-subject validation and performance evidence

Use Stanford HARDI through `AFQ.data.fetch.organize_stanford_data`, as in target tests. Record actual subject/session and exact registration contrast/template. Reuse the M0 harness/results where applicable; do not redo identical runs solely for a milestone label.

Reuse the same tractography, scalar maps, bundle definitions, prealignment settings, and downstream settings for `SynMap` and FireANTs. Use separate output directories. Hold the prealignment matrix fixed in the controlled harness where practical; otherwise record both matrices. This limits downstream variability without adding an RNG framework.

Produce a runnable harness/config and `fireants_validation.md` containing:

| Evidence | Minimum report |
|---|---|
| Timing | `SynMap` CPU and FireANTs CPU; FireANTs GPU when available. Include optimization, extraction, inverse, and total usable-map time. State settings and avoid comparing a smoke budget against a full run as if equivalent. |
| Inverse | Both composition directions, p95/max in voxel and mm units, evaluated/total coverage, chosen inverse method |
| Visual quality | Template/subject and representative ROI overlays in both spaces, reviewed for gross displacement and hemisphere mistakes |
| ROI agreement | Dice for a few named bilateral/small/large ROIs, nearest-neighbor label sampling on a common grid |
| Downstream result | Per-bundle presence and streamline counts, plus FA profile correlations/absolute differences; missing/constant profiles marked undefined |
| Provenance | Hardware, versions/revisions, effective settings, device, CPU allocation, memory if readily measurable |

Agreement with `SynMap` is not ground-truth anatomical accuracy. Investigate missing bundles and gross discrepancies rather than hiding them. An inverse/coverage failure or visibly incorrect registration blocks “validated.” Extensive Jacobian diagnostics and larger benchmarks are useful follow-up; finite fields and the required directional/inverse tests remain essential now.

**Before upstream submission, actively try to obtain at least one GPU run.** Schedule/request access early in the project; run the same subject/settings through the entire adapter, synchronizing the selected GPU at timing boundaries. Report GPU model and full bidirectional-map cost, not optimization alone. This spec does not authorize sending access requests on the owner's behalf.

If CPU is slower and no GPU result is available, report correctness and integration progress separately from the unmet acceleration objective. Continue integration and seek GPU evidence; defer a performance-motivated upstream submission until benefit is demonstrated. A speedup on this subject is scoped evidence, not a broad benchmark claim.

## 12. First PR and follow-up boundaries

Prepare the smallest complete integration PR: mapping class/required conversion, focused tests, ordinary persistence with backend-specific names, one-subject evidence, and concise usage/limitations. Implementation and its relevant tests belong together in atomic single-issue commits. Keep experiment data, generated outputs and unused helpers out of the upstream patch. Run applicable existing regression checks and report unrun checks. No default change or neighboring refactor.

Resolve minimal optional packaging and documentation needs with maintainers for that PR. The target extras file is `setup.cfg`; FireANTs already requires torch. Select release constraints only after testing the actual published package. Config/CLI exposure and its serialization tests, extended runtime controls, cache hardening, general geometry handling, and broad CI/release coverage stay in follow-ups unless maintainers explicitly require them for merge. Keep the development environment reproducible without inventing a compatibility matrix.

## 13. Milestones and handoffs

| Milestone | Exit condition |
|---|---|
| M0: shared harness | Reproducible Stanford HARDI reference and adapter runs, matched-thread timing, correctness measurements and overlays under §4; developed alongside M1/M2 |
| M1: adapter correctness | Smallest conversion/inverse implementation with focused §10A coverage of the §6 contract |
| M2: minimal integration | Python mapping class, lazy imports, backend filenames, ordinary native save/reload |
| M3: correctness and speed | One-subject downstream comparison, reviewed overlays, and measured complete-map speedup; report missing GPU/performance evidence explicitly |
| M4: submission decision | Evidence supports practical benefit; concise PR prepared, owner reviews whether to submit |

M0 setup and M1/M2 implementation can proceed together; M2 consumes the tested M1 behavior as it becomes available. Agree on the §6 consumer contract before dividing implementation. A known coordinate/inverse defect blocks accepting that behavior, not unrelated implementation work. If time runs out, report a tested partial result and remaining correctness/performance evidence. Do not build duplicate experimental and production adapters.

## 14. V1 acceptance checklist

- [ ] Exact source/runtime versions and pilot settings recorded.
- [ ] M0 decoding and native-evaluate checks pass before judging adapter overlays; both timing runs use the same explicit job-script thread budget.
- [ ] §6 behavior has focused §10 coverage through the selected implementation, including actual streamline movement, unequal grids and inverse composition with coverage.
- [ ] CPU end-to-end computation and native save/reload work; `SynMap` defaults/behavior remain unchanged.
- [ ] Backend filenames are separate; imports/construction stay lazy; settings/device/threads are recorded as provenance.
- [ ] Outputs follow pyAFQ's placement/naming conventions and introduce no feature-specific BIDS validation findings relative to matched `SynMap` outputs; preexisting upstream findings are documented, not blockers.
- [ ] One-subject overlays, ROI agreement, bundle counts, and profiles reported with discrepancies explained.
- [ ] GPU access/run was pursued; actual result or unavailability and its effect on submission are stated.
- [ ] Comparable total usable-map timing demonstrates a speedup at useful registration quality on at least one supported device; otherwise the performance objective is explicitly incomplete and no speed claim is made.

## 15. Follow-up backlog

Config/CLI and serialization, optional packaging/version support, comprehensive docs/CI, automatic device selection, seed/thread APIs and state scoping, advanced import diagnostics, cache integrity/invalidation as shared infrastructure, shear/general geometry policy, expanded deformation diagnostics, Greedy, FireANTs affine, MPS, `combine_bundle`, and larger backend/device/preset benchmarks.

These items are intentionally outside the hackathon's completion criteria. Revisit them only after a correct registration and a plausible performance benefit have been demonstrated.
